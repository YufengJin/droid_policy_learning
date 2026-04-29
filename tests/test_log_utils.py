"""
Smoke test for robomimic.utils.log_utils — uses mocks so no real W&B / TB
installation or network access is required.

Run:
    python tests/test_log_utils.py
"""
import io
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


# --- Pre-import mocks: must be installed in sys.modules BEFORE log_utils import ---

def _install_mocks():
    sys.modules.setdefault("tensorboardX", MagicMock(name="tensorboardX_mock"))
    sys.modules.setdefault("wandb", MagicMock(name="wandb_mock"))
    sys.modules.setdefault("wandb.integration", MagicMock())
    sys.modules.setdefault("wandb.integration.sb3", MagicMock())

    fake_macros = ModuleType("robomimic.macros")
    fake_macros.WANDB_API_KEY = "FAKE_KEY"
    fake_macros.WANDB_ENTITY = "fake-entity"
    sys.modules["robomimic.macros"] = fake_macros


def _make_fake_config():
    cfg = SimpleNamespace()
    cfg.experiment = SimpleNamespace(
        name="smoke_run",
        logging=SimpleNamespace(wandb_proj_name="smoke_proj"),
    )
    cfg.algo_name = "bc"
    cfg.meta = {"hp_keys": ["lr"], "hp_values": [1e-3], "extra": "x"}
    cfg.to_dict = lambda: {"experiment": {"name": "smoke_run"}, "lr": 1e-3}
    return cfg


# --- Test cases ---

def test_print_logger():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        path = f.name
    try:
        # Capture the constructor's banner so it doesn't pollute test output.
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            pl = log_utils.PrintLogger(path)
        finally:
            sys.stdout = old_stdout
        pl.write("hello\n")
        pl.flush()
        assert "hello" in open(path).read()
        print("  PrintLogger: OK")
    finally:
        os.unlink(path)


def test_data_logger_scalar_and_stats():
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=True, log_wandb=True,
                                  wandb_init_retries=1, wandb_init_backoff_seconds=0)
        dl._tb_logger.reset_mock(); dl._wandb_logger.reset_mock()  # isolate from prior tests

        # log_stats=True on first call seeds the cache; subsequent writes append
        # because the key is now present in _data (legacy semantics preserved).
        dl.record("loss", 0.5, step=0, log_stats=True)
        dl.record("loss", 0.3, step=1)
        dl.record("loss", 0.4, step=2, log_stats=True)

        # TB scalar writes
        tb_calls = dl._tb_logger.add_scalar.call_args_list
        loss_writes = [c for c in tb_calls if c.args[0] == "loss"]
        assert len(loss_writes) == 3, "expected 3 TB scalar writes for 'loss', got {}".format(len(loss_writes))

        # TB stats writes: 4 stats × 2 calls where log_stats=True = 8
        stats_keys = {"loss-mean", "loss-std", "loss-min", "loss-max"}
        stats_writes = [c for c in tb_calls if c.args[0] in stats_keys]
        assert len(stats_writes) == 8, "expected 8 TB stats writes, got {}".format(len(stats_writes))

        # wandb scalar writes — find the loss-mean log among many calls
        wandb_calls = dl._wandb_logger.log.call_args_list
        assert any(c.args[0] == {"loss": 0.5} for c in wandb_calls), "wandb missing scalar"
        mean_calls = [c.args[0] for c in wandb_calls if list(c.args[0].keys()) == ["loss-mean"]]
        # Last loss-mean reflects all 3 values; allow float tolerance
        expected = (0.5 + 0.3 + 0.4) / 3
        assert mean_calls and abs(mean_calls[-1]["loss-mean"] - expected) < 1e-9, \
            "wandb mean mismatch: {}".format(mean_calls)

        # get_stats sanity
        stats = dl.get_stats("loss")
        assert abs(stats["mean"] - expected) < 1e-9
        assert stats["min"] == 0.3 and stats["max"] == 0.5

        dl.close()
        print("  DataLogger scalar + log_stats: OK")


def test_data_logger_image():
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=True, log_wandb=True,
                                  wandb_init_retries=1, wandb_init_backoff_seconds=0)
        dl._tb_logger.reset_mock(); dl._wandb_logger.reset_mock()  # isolate from prior tests
        # 3D image — code should auto-unsqueeze to 4D NHWC
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        dl.record("frame", img, step=10, data_type="image")

        # TB image write happened
        assert dl._tb_logger.add_images.called, "TB add_images not called"
        kwargs = dl._tb_logger.add_images.call_args.kwargs
        assert kwargs["dataformats"] == "NHWC"
        assert kwargs["img_tensor"].ndim == 4 and kwargs["img_tensor"].shape == (1, 4, 4, 3)

        # wandb image write happened (wrapped in wandb.Image)
        assert dl._wandb_logger.log.called, "wandb log not called"
        dl.close()
        print("  DataLogger image: OK")


def test_data_logger_text():
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=True, log_wandb=True,
                                  wandb_init_retries=1, wandb_init_backoff_seconds=0)
        dl._tb_logger.reset_mock(); dl._wandb_logger.reset_mock()  # isolate from prior tests
        dl.log_text("note", "first epoch complete", step=0)

        assert dl._tb_logger.add_text.called, "TB add_text not called"
        args, kwargs = dl._tb_logger.add_text.call_args
        assert args[0] == "note" and args[1] == "first epoch complete"
        assert kwargs["global_step"] == 0

        assert dl._wandb_logger.log.called
        wandb_call = dl._wandb_logger.log.call_args
        assert wandb_call.args[0] == {"note": "first epoch complete"}
        assert wandb_call.kwargs["step"] == 0

        dl.close()
        print("  DataLogger text: OK")


def test_data_logger_record_dict():
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=True, log_wandb=False)
        dl._tb_logger.reset_mock()
        dl.record_dict({"loss": 0.1, "acc": 0.9, "lr": 1e-3}, step=5)

        scalar_calls = dl._tb_logger.add_scalar.call_args_list
        recorded_keys = {c.args[0] for c in scalar_calls}
        assert recorded_keys == {"loss", "acc", "lr"}, "got {}".format(recorded_keys)
        for c in scalar_calls:
            assert c.args[2] == 5, "step mismatch"
        dl.close()
        print("  DataLogger record_dict: OK")


def test_data_logger_epoch_alias_backward_compat():
    """Legacy callers pass `epoch=` instead of `step=`; both must work."""
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=True, log_wandb=False)
        dl._tb_logger.reset_mock()
        dl.record("legacy", 1.0, epoch=42)
        dl.record("modern", 2.0, step=42)

        calls = dl._tb_logger.add_scalar.call_args_list
        steps = {c.args[0]: c.args[2] for c in calls}
        assert steps == {"legacy": 42, "modern": 42}
        dl.close()
        print("  DataLogger epoch backward-compat: OK")


def test_data_logger_no_backends():
    """Both backends disabled — record() should not error and not write."""
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()
        dl = log_utils.DataLogger(log_dir, cfg, log_tb=False, log_wandb=False)
        assert dl._tb_logger is None and dl._wandb_logger is None
        dl.record("x", 1.0, step=0)            # no error
        dl.log_text("note", "hi", step=0)       # no error
        dl.record_dict({"a": 1, "b": 2}, step=0)
        dl.close()
        print("  DataLogger no-backends: OK")


def test_log_warning_and_flush():
    log_utils.WARNINGS_BUFFER.clear()
    old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        log_utils.log_warning("alpha", print_now=False)
        log_utils.log_warning("beta", print_now=False)
        assert len(log_utils.WARNINGS_BUFFER) == 2
        log_utils.flush_warnings()
        flushed = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    assert "alpha" in flushed and "beta" in flushed
    assert log_utils.WARNINGS_BUFFER == []
    print("  log_warning + flush_warnings: OK")


def test_silence_stdout():
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        with log_utils.silence_stdout():
            print("should not appear")
        print("should appear")
    finally:
        sys.stdout = old_stdout
    assert "should not appear" not in captured.getvalue()
    assert "should appear" in captured.getvalue()
    print("  silence_stdout: OK")


def test_custom_tqdm_writes_stdout():
    # Contract 1: `file=` kwarg is rejected by custom_tqdm.
    try:
        log_utils.custom_tqdm(range(1), file=sys.stdout)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "custom_tqdm should reject `file=` kwarg"

    # Contract 2: default construction must pass file=sys.stdout to parent tqdm.
    # tqdm versions vary on attribute names; intercept the parent __init__ call instead.
    captured = {}
    real_init = log_utils.tqdm.__init__
    def spy(self, *a, **kw):
        captured.update(kw)
        kw["disable"] = True  # don't render
        return real_init(self, *a, **kw)
    log_utils.tqdm.__init__ = spy
    try:
        bar = log_utils.custom_tqdm(range(2))
        bar.close()
    finally:
        log_utils.tqdm.__init__ = real_init
    assert captured.get("file") is sys.stdout, "expected file=sys.stdout, got {}".format(captured.get("file"))
    print("  custom_tqdm: OK")


def test_wandb_init_retry_and_offline_fallback():
    """wandb.init raises N-1 times then succeeds in offline mode on the last attempt."""
    with tempfile.TemporaryDirectory() as log_dir:
        cfg = _make_fake_config()

        wandb_mock = sys.modules["wandb"]
        attempts = {"n": 0}
        def init_side_effect(**kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("network down")
            return None
        wandb_mock.init.side_effect = init_side_effect

        log_utils.WARNINGS_BUFFER.clear()
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            dl = log_utils.DataLogger(log_dir, cfg, log_tb=False, log_wandb=True,
                                      wandb_init_retries=3, wandb_init_backoff_seconds=0)
        finally:
            sys.stdout = old_stdout
        assert attempts["n"] == 3, "expected 3 init attempts, got {}".format(attempts["n"])
        # last-attempt mode must be offline
        last_call = wandb_mock.init.call_args_list[-1]
        assert last_call.kwargs["mode"] == "offline", "expected last attempt offline"
        # warning buffer should have 2 wandb-init warnings (from the failures)
        warn_msgs = [w for w in log_utils.WARNINGS_BUFFER if "wandb initialization error" in w]
        assert len(warn_msgs) == 2, "expected 2 warnings, got {}".format(len(warn_msgs))

        # Reset mock for other tests
        wandb_mock.init.side_effect = None
        dl.close()
        print("  wandb retry + offline fallback: OK")


# --- Driver ---

if __name__ == "__main__":
    _install_mocks()

    # Make robomimic importable from source layout if not already pip-installed.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from robomimic.utils import log_utils  # noqa: E402

    tests = [
        test_print_logger,
        test_data_logger_scalar_and_stats,
        test_data_logger_image,
        test_data_logger_text,
        test_data_logger_record_dict,
        test_data_logger_epoch_alias_backward_compat,
        test_data_logger_no_backends,
        test_log_warning_and_flush,
        test_silence_stdout,
        test_custom_tqdm_writes_stdout,
        test_wandb_init_retry_and_offline_fallback,
    ]
    for fn in tests:
        fn()
    print("\nALL {} tests passed.".format(len(tests)))
