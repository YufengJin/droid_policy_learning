#!/usr/bin/env python3
"""
Recursively list HDF5 datasets (paths, shapes, dtypes) for LIBERO / RoboCasa files.

Use this to verify whether joint positions, joint velocities, or Cartesian *velocity*
datasets exist. Empirical checks on mounted datasets (see dataset module docstrings):

  LIBERO (e.g. libero_10 HDF5): obs often includes joint_states (T,7), ee_pos, ee_ori,
  ee_states (T,6) == concat(ee_pos, ee_ori), gripper_states — no separate eef velocity.
  actions remain (T,7) end-effector style, not joint commands.

  RoboCasa v0.1 demo_im128: obs includes robot0_joint_pos, robot0_joint_vel,
  robot0_eef_pos/quat, robot0_gripper_qvel, etc. No robot0_eef_*linear/angular* velocity
  channel in obs. data/{demo}/action_dict may exist; RoboCasaDataset still uses actions[:, :7].

Loader cross-reference:
  - LIBERODataset: named keys in LIBERO_*_KEYS; any other obs/ dataset is loaded if it
    appears in obs_keys (generic branch) and is float32/64.
  - RoboCasaDataset: same pattern for extra keys under obs/. Add keys to train YAML
    observation.modalities.low_dim (or rgb) so ObsUtils and encoders see them.

Example:
  python -m robomimic.scripts.inspect_libero_robocasa_hdf5 \\
      --dataset /workspace/datasets/libero/libero_10/put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5
  python -m robomimic.scripts.inspect_libero_robocasa_hdf5 \\
      --dataset /workspace/datasets/robocasa/v0.1/multi_stage/.../demo_im128.hdf5
"""
from __future__ import print_function

import argparse
import os
import re
import sys

import h5py
import numpy as np


def inspect_file(path, demo_filter=None, keywords_only=False):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print("ERROR: not a file: {}".format(path), file=sys.stderr)
        return 2

    joint_pat = re.compile(r"joint|qpos|qvel|dof", re.I)
    vel_pat = re.compile(r"vel|velocity|speed", re.I)
    eef_pat = re.compile(r"eef|end.effector|cartesian|tcp", re.I)

    joint_hits = []
    vel_hits = []
    eef_hits = []
    all_rows = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if demo_filter is not None:
            # e.g. only data/demo_0/...
            prefix = "data/{}/".format(demo_filter)
            if not (name == "data" or name.startswith(prefix)):
                return
        try:
            shape = obj.shape
            dtype = obj.dtype
        except Exception as e:
            shape = "?"
            dtype = str(e)
        row = (name, shape, dtype)
        all_rows.append(row)
        n = name
        if joint_pat.search(n):
            joint_hits.append(row)
        if vel_pat.search(n):
            vel_hits.append(row)
        if eef_pat.search(n):
            eef_hits.append(row)

    with h5py.File(path, "r") as f:
        f.visititems(visitor)

    print("==== File: {} ====".format(path))
    if demo_filter:
        print("(filtered to demo: {})".format(demo_filter))
    print("Total datasets (after filter): {}".format(len(all_rows)))

    def _print_hits(title, hits):
        print("")
        print("==== {} ====".format(title))
        if not hits:
            print("(none)")
        else:
            for name, shape, dtype in sorted(hits, key=lambda x: x[0]):
                print("  {}  shape={}  dtype={}".format(name, shape, dtype))

    _print_hits("Keyword hits: joint / qpos / qvel / dof", joint_hits)
    _print_hits("Keyword hits: vel / velocity / speed", vel_hits)
    _print_hits("Keyword hits: eef / cartesian / tcp", eef_hits)

    if not keywords_only:
        print("")
        print("==== All datasets (sorted by path) ====")
        for name, shape, dtype in sorted(all_rows, key=lambda x: x[0]):
            print("  {}  shape={}  dtype={}".format(name, shape, dtype))

    # First demo actions/obs summary
    print("")
    print("==== First demo quick summary ====")
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("No 'data' group (not robomimic-style).")
            return 0
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        if not demos:
            print("Empty data/")
            return 0
        d0 = demos[0]
        g = f["data/{}".format(d0)]
        print("demo: {}".format(d0))
        if "actions" in g:
            a = g["actions"]
            print("  actions shape={} dtype={} sample[0]={}".format(
                a.shape, a.dtype, np.array(a[0]).tolist() if a.shape[0] else "empty"))
        for sub in ("obs", "next_obs", "observations"):
            if sub in g:
                print("  group '{}': keys={}".format(sub, list(g[sub].keys())))
        for k in g.keys():
            if k in ("obs", "next_obs", "observations"):
                continue
            item = g[k]
            if isinstance(item, h5py.Dataset):
                print("  dataset '{}' shape={} dtype={}".format(k, item.shape, item.dtype))
            elif isinstance(item, h5py.Group) and k == "action_dict":
                print("  group 'action_dict' keys={}".format(list(item.keys())))

    return 0


def main():
    p = argparse.ArgumentParser(description="Inspect LIBERO/RoboCasa HDF5 structure and keyword fields.")
    p.add_argument("--dataset", type=str, required=True, help="Path to .hdf5 or .h5 file")
    p.add_argument("--demo", type=str, default=None, help="Only list datasets under data/{demo}/")
    p.add_argument("--keywords-only", action="store_true", help="Only print keyword sections")
    args = p.parse_args()
    return inspect_file(args.dataset, demo_filter=args.demo, keywords_only=args.keywords_only)


if __name__ == "__main__":
    sys.exit(main() or 0)
