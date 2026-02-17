"""
Shared utility functions for ALOHA, LIBERO, and RoboCasa dataset classes.

Ported and adapted from cosmos-policy's dataset_utils.py and dataset_common.py
for use with the robomimic training pipeline.
"""

import io
import json
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HDF5 file discovery
# ---------------------------------------------------------------------------

def get_hdf5_files(data_dir, is_train=None):
    """
    Recursively find all HDF5 files in *data_dir* (follows symlinks).

    Args:
        data_dir (str): Root directory to search.
        is_train (bool | None):
            None  -> return all HDF5 files
            True  -> only files under a 'train' subdirectory
            False -> only files under a 'val' subdirectory

    Returns:
        list[str]: Sorted list of absolute paths.
    """
    assert os.path.exists(data_dir), "Directory '{}' does not exist.".format(data_dir)
    hdf5_files = []
    for root, _dirs, files in os.walk(data_dir, followlinks=True):
        for fname in files:
            if fname.lower().endswith((".h5", ".hdf5", ".he5")):
                filepath = os.path.join(root, fname)
                if is_train is None:
                    hdf5_files.append(filepath)
                else:
                    parts = os.path.normpath(os.path.relpath(filepath, data_dir)).split(os.sep)
                    if is_train and "train" in parts:
                        hdf5_files.append(filepath)
                    elif (not is_train) and "val" in parts:
                        hdf5_files.append(filepath)
    return sorted(hdf5_files)


# ---------------------------------------------------------------------------
# JPEG helpers
# ---------------------------------------------------------------------------

def decode_single_jpeg_frame(jpeg_bytes):
    """Decode a single JPEG byte-string into a uint8 numpy array (H, W, 3)."""
    img = Image.open(io.BytesIO(jpeg_bytes.tobytes()))
    return np.array(img, dtype=np.uint8)


def decode_jpeg_bytes_dataset(jpeg_ds):
    """Decode a variable-length JPEG dataset (T,) -> (T, H, W, 3) uint8."""
    frames = [np.array(Image.open(io.BytesIO(b.tobytes()))) for b in jpeg_ds]
    return np.stack(frames, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def calculate_dataset_statistics(data):
    """
    Compute per-dimension min/max/mean/std/median over all episodes for
    both ``actions`` and ``proprio`` keys.

    Args:
        data (dict): {episode_idx: {"actions": (T, D_a), "proprio": (T, D_p), ...}}

    Returns:
        dict: Keys like ``actions_min``, ``proprio_max``, etc.  Values are 1-D numpy arrays.
    """
    all_actions, all_proprio = [], []
    for _idx, ep in data.items():
        all_actions.append(ep["actions"])
        all_proprio.append(ep["proprio"])

    all_actions = np.concatenate(all_actions, axis=0)
    all_proprio = np.concatenate(all_proprio, axis=0)

    stats = {}
    for name, arr in [("actions", all_actions), ("proprio", all_proprio)]:
        stats["{}_min".format(name)] = np.min(arr, axis=0)
        stats["{}_max".format(name)] = np.max(arr, axis=0)
        stats["{}_mean".format(name)] = np.mean(arr, axis=0)
        stats["{}_std".format(name)] = np.std(arr, axis=0)
        stats["{}_median".format(name)] = np.median(arr, axis=0)
    return stats


def load_or_compute_dataset_statistics(data_dir, data, calc_fn=None, stats_path=None):
    """
    Load statistics from ``dataset_statistics.json``, or compute and save if missing.

    Args:
        data_dir: Used when stats_path is None: stats_path = data_dir/dataset_statistics.json
        data: Episode data dict for computing statistics.
        calc_fn: Function to compute raw stats; default: calculate_dataset_statistics.
        stats_path: Optional override. If provided, load/save from this exact path.

    Returns:
        dict: Statistics with numpy-array values.
    """
    if calc_fn is None:
        calc_fn = calculate_dataset_statistics
    if stats_path is None:
        stats_path = os.path.join(data_dir, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            json_stats = json.load(f)
        print("Loaded dataset statistics from: {}".format(stats_path))
    else:
        raw = calc_fn(data)
        json_stats = {k: v.tolist() for k, v in raw.items()}
        stats_parent = os.path.dirname(stats_path)
        if stats_parent:
            os.makedirs(stats_parent, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(json_stats, f, indent=4)
        print("Dataset statistics saved to: {}".format(stats_path))
    return {k: np.array(v) for k, v in json_stats.items()}


# ---------------------------------------------------------------------------
# Normalisation / rescaling
# ---------------------------------------------------------------------------

def rescale_array(arr, data_min, data_max):
    """Rescale *arr* from [data_min, data_max] to [-1, +1]."""
    return 2.0 * (arr - data_min) / (data_max - data_min + 1e-8) - 1.0


def unrescale_array(arr, data_min, data_max):
    """Inverse of :func:`rescale_array` – map [-1, +1] back to [data_min, data_max]."""
    return (arr + 1.0) / 2.0 * (data_max - data_min + 1e-8) + data_min


def rescale_data(data, dataset_stats, data_key):
    """
    Rescale ``data[ep][data_key]`` to [-1, +1] for every episode using
    pre-computed *dataset_stats*.

    Returns a **new** dict (episodes are shallow-copied).
    """
    d_min = dataset_stats["{}_min".format(data_key)]
    d_max = dataset_stats["{}_max".format(data_key)]
    out = {}
    for ep_idx, ep in data.items():
        ep_copy = ep.copy()
        ep_copy[data_key] = rescale_array(ep[data_key], d_min, d_max)
        out[ep_idx] = ep_copy
    return out


def rescale_episode_data(episode_data, dataset_stats, data_key):
    """Rescale a single episode's *data_key* array to [-1, +1]."""
    arr = episode_data[data_key]
    d_min = dataset_stats["{}_min".format(data_key)]
    d_max = dataset_stats["{}_max".format(data_key)]
    return rescale_array(arr, d_min, d_max)


# ---------------------------------------------------------------------------
# Action chunking
# ---------------------------------------------------------------------------

def get_action_chunk_with_padding(actions, start_idx, chunk_size, num_steps):
    """
    Extract an action chunk of length *chunk_size* starting at *start_idx*.
    If there are fewer remaining actions than *chunk_size*, the last action
    is repeated to fill the chunk.

    Args:
        actions (np.ndarray): (T, D)
        start_idx (int): Start index in the episode.
        chunk_size (int): Desired chunk length.
        num_steps (int): Total episode length.

    Returns:
        np.ndarray: (chunk_size, D)
    """
    remaining = num_steps - start_idx
    if remaining >= chunk_size:
        return actions[start_idx: start_idx + chunk_size]
    avail = actions[start_idx:]
    pad = np.tile(actions[-1], (chunk_size - remaining, 1))
    return np.concatenate([avail, pad], axis=0)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_images(images, target_size):
    """
    Resize a batch of images to (target_size, target_size).

    Args:
        images (np.ndarray): (T, H, W, C) uint8
        target_size (int): Square target size.

    Returns:
        np.ndarray: (T, target_size, target_size, C) uint8
    """
    if images.shape[1] == target_size and images.shape[2] == target_size:
        return images.copy()
    out = np.empty((images.shape[0], target_size, target_size, images.shape[3]), dtype=images.dtype)
    for i in range(images.shape[0]):
        out[i] = np.array(Image.fromarray(images[i]).resize((target_size, target_size)))
    return out
