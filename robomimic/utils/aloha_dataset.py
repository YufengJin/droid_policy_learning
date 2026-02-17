"""
ALOHA HDF5 dataset for robomimic training pipeline.

Reads ALOHA HDF5 files (non-standard format, similar to DROID) and returns data
in robomimic format:  ``{"obs": {...}, "actions": array}``.

Reference: cosmos_policy/datasets/aloha_dataset.py

HDF5 layout (per-file, one episode per file):
    observations/images/cam_high           (T, H, W, 3) uint8
    observations/images/cam_left_wrist     (T, H, W, 3) uint8
    observations/images/cam_right_wrist    (T, H, W, 3) uint8
    observations/qpos                      (T, 14)
    action                                 (T, 14)

For preprocessed datasets with video paths:
    observations/video_paths/cam_high      -> path to MP4 file
    observations/video_paths/cam_left_wrist
    observations/video_paths/cam_right_wrist
"""

import os
import numpy as np
import h5py
from collections import OrderedDict
from copy import deepcopy

import torch.utils.data

from robomimic.utils.dataset_utils import (
    get_hdf5_files,
    calculate_dataset_statistics,
    load_or_compute_dataset_statistics,
    rescale_array,
    unrescale_array,
    get_action_chunk_with_padding,
    resize_images,
)
import robomimic.utils.tensor_utils as TensorUtils

# Optional: OpenCV for MP4 loading
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def load_video_as_images(video_path, resize_size=None):
    """Load MP4 video -> (T, H, W, 3) uint8 RGB."""
    if not _HAS_CV2:
        raise ImportError("cv2 (OpenCV) is required for ALOHA MP4 video loading")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video: {}".format(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames in video: {}".format(video_path))
    images = np.array(frames, dtype=np.uint8)
    if resize_size is not None:
        images = resize_images(images, resize_size)
    return images


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------
ALOHA_IMAGE_KEYS = OrderedDict([
    ("cam_high_image", "cam_high"),
    ("cam_left_wrist_image", "cam_left_wrist"),
    ("cam_right_wrist_image", "cam_right_wrist"),
])


class ALOHADataset(torch.utils.data.Dataset):
    """
    ALOHA HDF5 dataset that outputs robomimic-compatible dicts.

    Each HDF5 file represents a single episode.  The dataset scans
    ``data_dir`` recursively for HDF5 files (optionally filtering
    by train/val subdirectories).
    """

    ACTION_DIM = 14
    PROPRIO_DIM = 14

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
        is_train=None,
    ):
        """
        Args:
            data_dir (str): Root directory containing ALOHA HDF5 file(s).
            obs_keys (list[str]): Observation keys to include
                (e.g. ``["cam_high_image", "cam_left_wrist_image", "cam_right_wrist_image", "qpos"]``).
            action_keys (tuple): Action key(s).
            seq_length (int): Sequence length to sample per item.
            frame_stack (int): Number of history frames to stack.
            pad_seq_length (bool): Pad sequences shorter than seq_length at the end.
            pad_frame_stack (bool): Pad sequences at the beginning for frame stacking.
            action_config (dict | None): Action normalisation config.
            filter_key (str | None): Unused for ALOHA (kept for interface compatibility).
            image_size (int | None): Target square image size for resizing.
            normalize_actions (bool): Whether to normalise actions to [-1, 1].
            is_train (bool | None): If not None, only load files from 'train' or 'val' subdirs.
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
        hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        if len(hdf5_files) == 0:
            if os.path.isfile(data_dir) and data_dir.lower().endswith((".h5", ".hdf5")):
                hdf5_files = [data_dir]
            else:
                raise FileNotFoundError("No HDF5 files found in {}".format(data_dir))

        # ------------------------------------------------------------------
        # Load episodes (one per file)
        # ------------------------------------------------------------------
        self.demos = []
        self.demo_lengths = []
        self._obs_data = {}
        self._action_data = {}
        self._index_to_demo = {}

        demo_idx = 0
        for fpath in sorted(hdf5_files):
            with h5py.File(fpath, "r") as f:
                obs_group = f.get("observations", None)
                if obs_group is None:
                    continue

                # Detect raw images vs MP4 video paths
                has_raw = "images" in obs_group
                has_video = "video_paths" in obs_group

                obs_dict = {}
                episode_length = None

                for obs_key in self.obs_keys:
                    if obs_key in ALOHA_IMAGE_KEYS:
                        cam_name = ALOHA_IMAGE_KEYS[obs_key]
                        if has_raw and cam_name in obs_group["images"]:
                            imgs = obs_group["images"][cam_name][:]
                        elif has_video and cam_name in obs_group["video_paths"]:
                            vp = obs_group["video_paths"][cam_name][()]
                            if isinstance(vp, bytes):
                                vp = vp.decode("utf-8")
                            video_abs = os.path.join(os.path.dirname(fpath), str(vp))
                            imgs = load_video_as_images(video_abs, resize_size=self.image_size)
                        else:
                            raise KeyError("Camera '{}' not found in {}".format(cam_name, fpath))
                        if self.image_size is not None and imgs.shape[1] != self.image_size:
                            imgs = resize_images(imgs, self.image_size)
                        obs_dict[obs_key] = imgs.astype(np.uint8)
                        if episode_length is None:
                            episode_length = imgs.shape[0]
                    elif obs_key == "qpos":
                        qpos = obs_group["qpos"][:].astype(np.float32)
                        obs_dict[obs_key] = qpos
                        if episode_length is None:
                            episode_length = qpos.shape[0]
                    else:
                        if obs_key in obs_group:
                            obs_dict[obs_key] = obs_group[obs_key][:].astype(np.float32)
                        else:
                            raise KeyError("Obs key '{}' not found in {}".format(obs_key, fpath))

                # Actions
                actions = f["action"][:].astype(np.float32)
                T = actions.shape[0]
                if episode_length is not None and T != episode_length:
                    T = min(T, episode_length)
                    actions = actions[:T]
                    for k in obs_dict:
                        obs_dict[k] = obs_dict[k][:T]

                self.demos.append(fpath)
                self.demo_lengths.append(T)
                self._obs_data[demo_idx] = obs_dict
                self._action_data[demo_idx] = actions
                demo_idx += 1

        if len(self.demos) == 0:
            raise RuntimeError("No episodes loaded from {}".format(data_dir))

        # ------------------------------------------------------------------
        # Normalisation
        # ------------------------------------------------------------------
        self._action_stats = None
        if self.normalize_actions:
            self._dataset_stats = load_or_compute_dataset_statistics(
                data_dir=data_dir,
                data={i: {"actions": self._action_data[i],
                           "proprio": self._obs_data[i].get("qpos",
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

        print("ALOHADataset: loaded {} episodes, {} total samples from {}".format(
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
