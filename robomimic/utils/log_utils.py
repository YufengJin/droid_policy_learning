"""
Logging utilities for robomimic: stdout/stderr redirection, TensorBoard, and W&B.

The main entry point is `DataLogger`, a thin facade that fans out a single
`record(...)` call to TensorBoard (via tensorboardX) and/or Weights & Biases.
Both backends are optional and independent.
"""
from __future__ import annotations

import os
import sys
import time
import textwrap
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np
from termcolor import colored
from tqdm import tqdm

import robomimic

# Global warning buffer; populated via @log_warning, drained via @flush_warnings.
WARNINGS_BUFFER: list[str] = []


class PrintLogger:
    """Tee `print()` output to both terminal and a file."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        # Required for python3 stdio compatibility; intentional no-op.
        pass


class DataLogger:
    """
    Unified TB + W&B logging facade.

    Both backends are optional. Disabling both makes record() a no-op aside
    from running-stats accumulation.

    Public API:
        record(key, value, step, data_type='scalar', log_stats=False)
        record_dict({key: value, ...}, step)
        log_text(key, message, step)
        get_stats(key)
        close()
    """

    def __init__(
        self,
        log_dir: str,
        config: Any,
        log_tb: bool = True,
        log_wandb: bool = False,
        wandb_init_retries: int = 10,
        wandb_init_backoff_seconds: float = 30.0,
        wandb_run_id: str = None,
    ):
        self._tb_logger = None
        self._wandb_logger = None
        self._wandb_run_id = None
        self._data: dict[str, list[float]] = {}

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            self._init_wandb(
                log_dir=log_dir,
                config=config,
                retries=wandb_init_retries,
                backoff=wandb_init_backoff_seconds,
                wandb_run_id=wandb_run_id,
            )

    # ---------- wandb init ----------

    def _init_wandb(self, log_dir: str, config: Any, retries: int, backoff: float, wandb_run_id: str = None) -> None:
        import wandb
        import robomimic.macros as Macros

        if Macros.WANDB_API_KEY is not None:
            os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

        assert Macros.WANDB_ENTITY is not None, (
            "WANDB_ENTITY macro is set to None.\n"
            "Set this macro in {base}/macros_private.py\n"
            "If this file does not exist, first run python {base}/scripts/setup_macros.py".format(
                base=robomimic.__path__[0],
            )
        )

        # 若用户显式设置 WANDB_MODE（offline/disabled/online），则尊重之并跳过 online 重试，
        # 避免无 key/无网时白白 backoff 重试 retries 次。未设置时维持原 online→offline 兜底逻辑。
        forced_mode = os.environ.get("WANDB_MODE", "").strip().lower() or None

        for attempt in range(retries):
            try:
                self._wandb_logger = wandb
                # resume: when wandb_run_id is provided (e.g. continuing a checkpoint),
                # reattach to the same run so logging appends to the original curves.
                init_kwargs = dict(
                    entity=Macros.WANDB_ENTITY,
                    project=config.experiment.logging.wandb_proj_name,
                    name=config.experiment.name,
                    dir=log_dir,
                    mode=(forced_mode if forced_mode is not None
                          else ("offline" if attempt == retries - 1 else "online")),
                )
                if wandb_run_id is not None:
                    init_kwargs["id"] = wandb_run_id
                    init_kwargs["resume"] = "allow"
                self._wandb_logger.init(**init_kwargs)
                # remember the active run id so it can be persisted into checkpoints
                try:
                    self._wandb_run_id = self._wandb_logger.run.id
                except Exception:
                    self._wandb_run_id = wandb_run_id
                self._sync_wandb_config(config)
                break
            except Exception as e:
                log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                self._wandb_logger = None
                time.sleep(backoff)

    def _sync_wandb_config(self, config: Any) -> None:
        """Push experiment config into wandb.config for searchable comparisons."""
        if self._wandb_logger is None:
            return
        try:
            if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
                self._wandb_logger.config.update(config.to_dict(), allow_val_change=True)
                return
        except Exception as e:
            log_warning("wandb config.to_dict() failed: {}; falling back to meta+hp".format(e))

        # Legacy fallback: meta + hp_keys/hp_values + algo
        wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
        for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
            wandb_config[k] = v
        if "algo" not in wandb_config:
            wandb_config["algo"] = config.algo_name
        self._wandb_logger.config.update(wandb_config)

    @property
    def wandb_run_id(self):
        """Active wandb run id (None if wandb logging is disabled/failed). Persist into
        checkpoints so a later resume can reattach to the same run."""
        return self._wandb_run_id

    # ---------- record API ----------

    def record(
        self,
        k: str,
        v: Any,
        step: Optional[int] = None,
        data_type: str = 'scalar',
        log_stats: bool = False,
        *,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Args:
            k: metric key.
            v: scalar float, or HxWxC / NxHxWxC ndarray for images.
            step: global step (iteration / epoch). Wins over `epoch` if both given.
            data_type: 'scalar' | 'image'.
            log_stats: scalar-only — also write running mean/std/min/max.
            epoch: deprecated alias for `step`; preserved for backward compatibility.
        """
        if step is None:
            step = epoch
        assert step is not None, "record() requires `step` (or legacy `epoch`)"
        assert data_type in ('scalar', 'image'), "unknown data_type: {}".format(data_type)

        if data_type == 'scalar':
            self._cache_if_needed(k, v, log_stats)
            self._log_scalar(k, v, step)
            if log_stats:
                self._log_stats(k, step)
        else:  # image
            self._log_image(k, v, step)

    def record_dict(
        self,
        metrics: Mapping[str, Any],
        step: int,
        data_type: str = 'scalar',
        log_stats: bool = False,
    ) -> None:
        """Batch-record many keys at the same step."""
        for k, v in metrics.items():
            self.record(k, v, step=step, data_type=data_type, log_stats=log_stats)

    def log_text(self, k: str, msg: str, step: int) -> None:
        """Log a text annotation (e.g. release notes, command, eval breakdown)."""
        if self._tb_logger is not None:
            self._tb_logger.add_text(k, msg, global_step=step)
        if self._wandb_logger is not None:
            try:
                self._wandb_logger.log({k: msg}, step=step)
            except Exception as e:
                log_warning("wandb text log: {}".format(e))

    # ---------- private writers ----------

    def _cache_if_needed(self, k: str, v: float, log_stats: bool) -> None:
        # Once a key has been requested with stats, keep accumulating it forever
        # (matches legacy semantics so get_stats(k) can be called any time later).
        if log_stats or k in self._data:
            self._data.setdefault(k, []).append(v)

    def _log_scalar(self, k: str, v: float, step: int) -> None:
        if self._tb_logger is not None:
            self._tb_logger.add_scalar(k, v, step)
        if self._wandb_logger is not None:
            try:
                self._wandb_logger.log({k: v}, step=step)
            except Exception as e:
                log_warning("wandb scalar log: {}".format(e))

    def _log_stats(self, k: str, step: int) -> None:
        for stat_k, stat_v in self.get_stats(k).items():
            full_key = "{}-{}".format(k, stat_k)
            if self._tb_logger is not None:
                self._tb_logger.add_scalar(full_key, stat_v, step)
            if self._wandb_logger is not None:
                try:
                    self._wandb_logger.log({full_key: stat_v}, step=step)
                except Exception as e:
                    log_warning("wandb stats log: {}".format(e))

    def _log_image(self, k: str, v: np.ndarray, step: int) -> None:
        if v.ndim == 3:
            v = v[None, ...]
        if self._tb_logger is not None:
            self._tb_logger.add_images(k, img_tensor=v, global_step=step, dataformats="NHWC")
        if self._wandb_logger is not None:
            try:
                import wandb
                self._wandb_logger.log({k: wandb.Image(v)}, step=step)
            except Exception as e:
                log_warning("wandb image log: {}".format(e))

    # ---------- stats / lifecycle ----------

    def get_stats(self, k: str) -> dict[str, float]:
        arr = self._data[k]
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
        }

    def close(self) -> None:
        if self._tb_logger is not None:
            self._tb_logger.close()
        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class custom_tqdm(tqdm):
    """tqdm that writes to stdout (default writes stderr) so it tees with PrintLogger."""

    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super().__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """Temporarily redirect stdout to /dev/null."""
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def log_warning(message: str, color: str = "yellow", print_now: bool = True) -> None:
    """Buffer a colored warning; optionally also print immediately."""
    global WARNINGS_BUFFER
    buffer_message = colored(
        "ROBOMIMIC WARNING(\n{}\n)".format(textwrap.indent(message, "    ")),
        color,
    )
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings() -> None:
    """Print and clear the warning buffer."""
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []
