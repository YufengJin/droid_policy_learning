"""
LIBERO HDF5 dataset for robomimic training pipeline.

Reads LIBERO HDF5 files (robosuite standard format) and returns data in robomimic
format:  ``{"obs": {...}, "actions": array}``.

Reference: cosmos_policy/datasets/libero_dataset.py

HDF5 layout:
    data/{demo}/obs/agentview_rgb[_jpeg]       (T, H, W, 3) or JPEG bytes
    data/{demo}/obs/eye_in_hand_rgb[_jpeg]
    data/{demo}/obs/robot_states               (T, 9)
    data/{demo}/actions                        (T, 7)
"""

import os
import numpy as np
import h5py
from collections import OrderedDict
from copy import deepcopy

import torch.utils.data

from robomimic.utils.dataset_utils import (
    get_hdf5_files,
    decode_jpeg_bytes_dataset,
    calculate_dataset_statistics,
    load_or_compute_dataset_statistics,
    rescale_array,
    unrescale_array,
    get_action_chunk_with_padding,
)
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


# ---------------------------------------------------------------------------
# Key mapping: robomimic obs key  ->  (raw HDF5 key, JPEG fallback key)
# ---------------------------------------------------------------------------
LIBERO_IMAGE_KEYS = OrderedDict([
    ("agentview_image", ("agentview_rgb", "agentview_rgb_jpeg")),
    ("eye_in_hand_image", ("eye_in_hand_rgb", "eye_in_hand_rgb_jpeg")),
])

LIBERO_LOW_DIM_KEYS = OrderedDict([
    ("robot0_eef_pos", None),
    ("robot0_eef_quat", None),
    ("robot0_gripper_qpos", None),
    ("robot_states", "robot_states"),
])


class LIBERODataset(torch.utils.data.Dataset):
    """
    LIBERO HDF5 dataset that outputs robomimic-compatible dicts.
    """

    ACTION_DIM = 7
    PROPRIO_DIM = 9

    def __init__(
        self,
        data_dir,
        obs_keys,
        action_keys=("actions",),
        seq_length=1,
        frame_stack=1,
        pad_seq_length=True,
        pad_frame_stack=True,
        action_config=None,
        filter_key=None,
        image_size=None,
        normalize_actions=True,
    ):
        """
        Args:
            data_dir (str): Directory containing LIBERO HDF5 file(s).
            obs_keys (list[str]): Observation keys to include.
            action_keys (tuple): Action keys.
            seq_length (int): Sequence length to sample.
            frame_stack (int): Number of history frames to stack.
            pad_seq_length (bool): Pad short sequences at the end.
            pad_frame_stack (bool): Pad beginning for frame stacking.
            action_config (dict | None): Action normalisation configuration.
            filter_key (str | None): "train" or "valid" to select demos.
            image_size (tuple | None): (H, W) target for image resizing.
            normalize_actions (bool): Whether to normalise actions to [-1, 1].
        """
        super().__init__()

        self.data_dir = data_dir
        self.obs_keys = list(obs_keys)
        self.action_keys = list(action_keys)
        self.seq_length = seq_length
        self.n_frame_stack = frame_stack
        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.action_config = action_config or {}
        self.image_size = image_size
        self.normalize_actions = normalize_actions

        # ------------------------------------------------------------------
        # Discover HDF5 files
        # ------------------------------------------------------------------
        hdf5_files = get_hdf5_files(data_dir)
        if len(hdf5_files) == 0:
            if os.path.isfile(data_dir) and data_dir.lower().endswith((".h5", ".hdf5")):
                hdf5_files = [data_dir]
            else:
                raise FileNotFoundError("No HDF5 files found in {}".format(data_dir))

        # ------------------------------------------------------------------
        # Load episodes
        # ------------------------------------------------------------------
        self.demos = []
        self.demo_lengths = []
        self._obs_data = {}
        self._action_data = {}
        self._index_to_demo = {}

        demo_idx = 0
        for fpath in sorted(hdf5_files):
            with h5py.File(fpath, "r") as f:
                if "data" not in f:
                    continue
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)

                # Apply filter key
                if filter_key is not None and "mask" in f:
                    if filter_key in f["mask"]:
                        valid_demos = set(
                            ep.decode("utf-8") if isinstance(ep, bytes) else str(ep)
                            for ep in f["mask/{}".format(filter_key)][:]
                        )
                        demo_keys = [k for k in demo_keys if k in valid_demos]

                for dk in demo_keys:
                    obs_group = f["data/{}/obs".format(dk)]

                    obs_dict = {}
                    for obs_key in self.obs_keys:
                        if obs_key in LIBERO_IMAGE_KEYS:
                            raw_key, jpeg_key = LIBERO_IMAGE_KEYS[obs_key]
                            if jpeg_key in obs_group:
                                imgs = decode_jpeg_bytes_dataset(obs_group[jpeg_key])
                            elif raw_key in obs_group:
                                imgs = obs_group[raw_key][:]
                            else:
                                raise KeyError("Neither '{}' nor '{}' found in {}".format(raw_key, jpeg_key, fpath))
                            obs_dict[obs_key] = imgs.astype(np.uint8)
                        elif obs_key == "robot_states":
                            obs_dict[obs_key] = obs_group["robot_states"][:].astype(np.float32)
                        elif obs_key == "robot0_eef_pos":
                            obs_dict[obs_key] = obs_group["robot_states"][:, :3].astype(np.float32)
                        elif obs_key == "robot0_eef_quat":
                            obs_dict[obs_key] = obs_group["robot_states"][:, 3:7].astype(np.float32)
                        elif obs_key == "robot0_gripper_qpos":
                            obs_dict[obs_key] = obs_group["robot_states"][:, 7:9].astype(np.float32)
                        else:
                            if obs_key in obs_group:
                                obs_dict[obs_key] = obs_group[obs_key][:].astype(np.float32)
                            else:
                                raise KeyError("Obs key '{}' not found in {}".format(obs_key, fpath))

                    actions = f["data/{}/actions".format(dk)][:].astype(np.float32)

                    T = actions.shape[0]
                    self.demos.append((fpath, dk))
                    self.demo_lengths.append(T)
                    self._obs_data[demo_idx] = obs_dict
                    self._action_data[demo_idx] = actions
                    demo_idx += 1

        if len(self.demos) == 0:
            raise RuntimeError("No demonstrations loaded from {}".format(data_dir))

        # ------------------------------------------------------------------
        # Normalisation
        # ------------------------------------------------------------------
        self._action_stats = None
        if self.normalize_actions:
            self._dataset_stats = load_or_compute_dataset_statistics(
                data_dir=data_dir,
                data={i: {"actions": self._action_data[i],
                           "proprio": self._obs_data[i].get("robot_states",
                                       np.zeros((self.demo_lengths[i], self.PROPRIO_DIM), dtype=np.float32))}
                       for i in range(len(self.demos))},
            )
            a_min = self._dataset_stats["actions_min"]
            a_max = self._dataset_stats["actions_max"]
            for i in range(len(self.demos)):
                self._action_data[i] = rescale_array(self._action_data[i], a_min, a_max).astype(np.float32)

        # ------------------------------------------------------------------
        # Index map
        # ------------------------------------------------------------------
        self._build_index_map()

        print("LIBERODataset: loaded {} demos, {} total samples from {}".format(
            len(self.demos), len(self), data_dir))

    def _build_index_map(self):
        self._index_to_demo = {}
        total = 0
        for di, length in enumerate(self.demo_lengths):
            n_samples = length
            if not self.pad_seq_length:
                n_samples = max(1, length - self.seq_length + 1)
            if not self.pad_frame_stack:
                n_samples = max(1, n_samples - self.n_frame_stack + 1)
            for j in range(n_samples):
                self._index_to_demo[total] = (di, j)
                total += 1
        self._total_samples = total

    def __len__(self):
        return self._total_samples

    def __getitem__(self, index):
        demo_idx, index_in_demo = self._index_to_demo[index]
        T = self.demo_lengths[demo_idx]

        seq_begin = max(0, index_in_demo - (self.n_frame_stack - 1))
        seq_end = min(T, index_in_demo + self.seq_length)
        begin_pad = max(0, (self.n_frame_stack - 1) - index_in_demo)
        end_pad = max(0, index_in_demo + self.seq_length - T)

        obs_dict = {}
        for key in self.obs_keys:
            arr = self._obs_data[demo_idx][key][seq_begin:seq_end]
            if begin_pad > 0 or end_pad > 0:
                arr = TensorUtils.pad_sequence(
                    {key: arr}, padding=(begin_pad, end_pad), pad_same=True
                )[key]
            obs_dict[key] = arr

        actions = self._action_data[demo_idx][seq_begin:seq_end]
        if begin_pad > 0 or end_pad > 0:
            actions = TensorUtils.pad_sequence(
                {"a": actions}, padding=(begin_pad, end_pad), pad_same=True
            )["a"]

        return {"obs": obs_dict, "actions": actions}

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_dataset_sampler(self):
        return None

    def get_action_normalization_stats(self):
        from robomimic.utils.dataset import action_stats_to_normalization_stats
        if self._action_stats is not None:
            return self._action_stats

        from robomimic.utils.dataset import _compute_traj_stats, _aggregate_traj_stats
        action_stats = None
        for di in range(len(self.demos)):
            traj = {"actions": self._action_data[di]}
            stats = _compute_traj_stats(traj)
            action_stats = stats if action_stats is None else _aggregate_traj_stats(action_stats, stats)

        self._action_stats = action_stats_to_normalization_stats(action_stats, {"actions": {"normalization": "min_max"}})
        return self._action_stats

    def get_obs_normalization_stats(self):
        return None
