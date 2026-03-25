"""Stub for debug training batch dump — no-op implementation."""

import os


def dump_training_batch_debug(batch, log_dir, tag="debug"):
    """Save a minimal text summary of the batch shapes to log_dir."""
    try:
        lines = ["=== {} batch shapes ===".format(tag)]
        for k, v in batch.items():
            if hasattr(v, "shape"):
                lines.append("  {}: {}".format(k, tuple(v.shape)))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(vv, "shape"):
                        lines.append("  {}/{}: {}".format(k, kk, tuple(vv.shape)))
        out_path = os.path.join(log_dir, "{}_batch_debug.txt".format(tag))
        os.makedirs(log_dir, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return out_path
    except Exception:
        return None
