"""
RoboCasa HDF5 dataset for robomimic training pipeline.

Reads RoboCasa HDF5 files and returns data in robomimic format:  ``{"obs": {...}, "actions": array}``.

Reference: cosmos_policy/datasets/robocasa_dataset.py

Supported HDF5 layouts:

1) RoboCasa v0.1 (e.g. v0.1/multi_stage/.../demo_im128.hdf5):
    data/{demo}/obs/robot0_agentview_left_image    (T, H, W, 3) uint8
    data/{demo}/obs/robot0_agentview_right_image
    data/{demo}/obs/robot0_eye_in_hand_image
    data/{demo}/obs/robot0_eef_pos                (T, 3)
    data/{demo}/obs/robot0_eef_quat               (T, 4)
    data/{demo}/obs/robot0_gripper_qpos           (T, 2)
    data/{demo}/actions                           (T, 12) -> take first 7D
    data/{demo}/action_dict/                      (optional)
    data/{demo}.attrs["ep_meta"]                  (task_description in ep_meta)

2) Robosuite / Cosmos (legacy):
    data/{demo}/obs/robot0_agentview_left_rgb[_jpeg]
    data/{demo}/obs/robot0_agentview_right_rgb[_jpeg]
    data/{demo}/obs/robot0_eye_in_hand_rgb[_jpeg]
    data/{demo}/robot_states   or  data/{demo}/obs/robot_states  (T, 9)
    data/{demo}/actions                            (T, 12)
    data/{demo}.attrs["task_description"]
"""

import hashlib
import json
import os
import pickle
import numpy as np
import h5py
from collections import OrderedDict, defaultdict
from copy import deepcopy

import torch
import torch.utils.data

# ---------------------------------------------------------------------------
# Cache root: /workspace/droid_policy_learning/.robocasa/
#   data_statistics/   — dataset normalisation stats (JSON)
#   lang_token/        — precomputed DistilBERT embeddings (pickle)
# ---------------------------------------------------------------------------
_ROBOCASA_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".robocasa",
)
_ROBOCASA_STATS_CACHE_DIR = os.path.join(_ROBOCASA_CACHE_ROOT, "data_statistics")
_ROBOCASA_LANG_CACHE_DIR = os.path.join(_ROBOCASA_CACHE_ROOT, "lang_token")

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
# Key mapping: robomimic obs key  ->  HDF5 obs keys (try in order)
# RoboCasa v0.1 uses *_image directly; robosuite uses *_rgb / *_rgb_jpeg
# ---------------------------------------------------------------------------
ROBOCASA_IMAGE_KEYS = OrderedDict([
    # agentview_image: alias for primary (left) agent view
    ("agentview_image", ("robot0_agentview_left_image", "robot0_agentview_left_rgb", "robot0_agentview_left_rgb_jpeg")),
    ("robot0_agentview_left_image", ("robot0_agentview_left_image", "robot0_agentview_left_rgb", "robot0_agentview_left_rgb_jpeg")),
    ("robot0_agentview_right_image", ("robot0_agentview_right_image", "robot0_agentview_right_rgb", "robot0_agentview_right_rgb_jpeg")),
    ("robot0_eye_in_hand_image", ("robot0_eye_in_hand_image", "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_rgb_jpeg")),
])

# Single global cache file: no duplicate embeddings across datasets
_LANG_CACHE_FILENAME = "distilbert_lang.pkl"


def _compute_lang_embeddings(unique_commands, cache_path):
    """
    Precompute DistilBERT embeddings for unique language commands.
    Uses a single global cache file: only missing commands are computed and
    merged into the cache. Returns dict: {command_str: np.ndarray (768,)}.
    """
    cached = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:
            print("RoboCasaDataset: failed to load lang cache {}: {}. Recomputing.".format(cache_path, e))

    missing = [c for c in unique_commands if c not in cached]
    if not missing:
        n_total = len(cached)
        print("Loaded lang embeddings from {} ({} total in cache, {} needed)".format(
            cache_path, n_total, len(unique_commands)))
        return {c: cached[c] for c in unique_commands}

    n_new = len(missing)
    print("Computing DistilBERT embeddings for {} new commands ({} already in cache)...".format(
        n_new, len(cached)))
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for cmd in missing:
            encoded = tokenizer(cmd, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            emb = outputs.last_hidden_state.sum(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            cached[cmd] = emb

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cached, f)
        print("Saved lang embeddings ({} total) to {}".format(len(cached), cache_path))
    except Exception as e:
        print("RoboCasaDataset: failed to save lang cache: {}".format(e))

    return {c: cached[c] for c in unique_commands}


ROBOCASA_LOW_DIM_KEYS = OrderedDict([
    ("robot0_eef_pos", None),       # from obs/robot0_eef_pos (v0.1) or robot_states[:, :3]
    ("robot0_eef_quat", None),      # from obs/robot0_eef_quat (v0.1) or robot_states[:, 3:7]
    ("robot0_gripper_qpos", None),  # from obs/robot0_gripper_qpos (v0.1) or robot_states[:, 7:9]
    ("robot_states", "robot_states"),
])


class RoboCasaDataset(torch.utils.data.Dataset):
    """
    RoboCasa HDF5 dataset that outputs robomimic-compatible dicts.

    The dataset loads all demonstration data into RAM at construction time
    and supports sequence sampling with padding.
    """

    ACTION_DIM = 7  # first 7 of 12D (ignore mobile base)
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
        verbose=False,
    ):
        """
        Args:
            data_dir (str): Directory containing RoboCasa HDF5 file(s).
            obs_keys (list[str]): Observation keys to include.
            action_keys (tuple): Action keys (always ["actions"]).
            seq_length (int): Sequence length to sample.
            frame_stack (int): Number of history frames to stack.
            pad_seq_length (bool): Pad short sequences at the end.
            pad_frame_stack (bool): Pad beginning for frame stacking.
            action_config (dict | None): Action normalisation configuration.
            filter_key (str | None): "train" or "valid" to select demos via HDF5 filter keys.
            image_size (tuple | None): If provided, (H, W) to resize images.
            normalize_actions (bool): Whether to normalise actions to [-1, 1].
            verbose (bool): If True, print debug info (HDF5 files, first demo structure).
        """
        super().__init__()

        self.data_dir = data_dir
        self.verbose = verbose
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
        # 同一目录下若有 demo.hdf5（原始）和 demo_im128.hdf5（含 obs），仅用 demo_im128
        by_dir = defaultdict(list)
        for f in hdf5_files:
            by_dir[os.path.dirname(f)].append(f)
        filtered = []
        for d, flist in by_dir.items():
            im128 = [p for p in flist if "_im128" in os.path.basename(p)]
            filtered.extend(im128 if im128 else flist)
        hdf5_files = sorted(filtered)
        if self.verbose:
            print("[RoboCasaDataset] data_dir={}, hdf5_files={}".format(data_dir, hdf5_files))
        if len(hdf5_files) == 0:
            # Maybe data_dir IS the hdf5 file
            if os.path.isfile(data_dir) and data_dir.lower().endswith((".h5", ".hdf5")):
                hdf5_files = [data_dir]
            else:
                raise FileNotFoundError("No HDF5 files found in {}".format(data_dir))

        # ------------------------------------------------------------------
        # Load episodes
        # ------------------------------------------------------------------
        self.demos = []          # list of (file_path, demo_key)
        self.demo_lengths = []   # length of each demo
        self._obs_data = {}      # demo_idx -> {"obs_key": array}
        self._action_data = {}   # demo_idx -> (T, ACTION_DIM)
        self._lang_data = {}     # demo_idx -> str (task_description)
        self._lang_embeddings = {}  # populated after loading if lang is requested
        self._index_to_demo = {} # global sample index -> (demo_idx, index_in_demo)
        self._use_lang = any(k in ("raw_language", "lang_fixed/language_distilbert") for k in self.obs_keys)

        demo_idx = 0
        for fpath in sorted(hdf5_files):
            with h5py.File(fpath, "r") as f:
                if "data" not in f:
                    continue
                demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)

                # Apply filter key if provided
                if filter_key is not None and "mask" in f:
                    if filter_key in f["mask"]:
                        valid_demos = set(
                            ep.decode("utf-8") if isinstance(ep, bytes) else str(ep)
                            for ep in f["mask/{}".format(filter_key)][:]
                        )
                        demo_keys = [k for k in demo_keys if k in valid_demos]

                for dk in demo_keys:
                    demo_group = f["data/{}".format(dk)]
                    # obs / observations: RoboCasa v0.1 用 obs；部分格式用 observations；原始 collect 无此 group
                    if "obs" in demo_group:
                        obs_group = demo_group["obs"]
                    elif "observations" in demo_group:
                        obs_group = demo_group["observations"]
                    else:
                        if self.verbose:
                            print("[RoboCasaDataset] data/{} keys: {}".format(dk, list(demo_group.keys())))
                        avail = list(demo_group.keys())
                        raise KeyError(
                            "data/{} 中缺少 'obs' 或 'observations'。当前 keys: {}. "
                            "若为 RoboCasa 原始 collect（无图像/obs），需先运行 dataset_states_to_obs.py 提取观测："
                            " OMP_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs.py --dataset <path>".format(dk, avail)
                        )

                    if self.verbose and demo_idx == 0:
                        print("[RoboCasaDataset] data/{} obs_group keys: {}".format(dk, list(obs_group.keys())))
                    # robot_states: optional. RoboCasa v0.1 has eef_pos, eef_quat, gripper_qpos directly in obs.
                    robot_states = None
                    if "robot_states" in demo_group:
                        robot_states = demo_group["robot_states"][:].astype(np.float32)
                    elif "robot_states" in obs_group:
                        robot_states = obs_group["robot_states"][:].astype(np.float32)

                    # --- Images + low-dim ---
                    obs_dict = {}
                    for obs_key in self.obs_keys:
                        if obs_key in ROBOCASA_IMAGE_KEYS:
                            hdf5_key_candidates = ROBOCASA_IMAGE_KEYS[obs_key]
                            imgs = None
                            for hk in hdf5_key_candidates:
                                if hk in obs_group:
                                    if "jpeg" in hk.lower():
                                        imgs = decode_jpeg_bytes_dataset(obs_group[hk])
                                    else:
                                        imgs = obs_group[hk][:]
                                    break
                            if imgs is None:
                                raise KeyError("None of {} found in data/{}/obs in {}".format(hdf5_key_candidates, dk, fpath))
                            obs_dict[obs_key] = imgs.astype(np.uint8)
                        elif obs_key == "robot_states":
                            if robot_states is not None:
                                obs_dict[obs_key] = robot_states
                            else:
                                raise KeyError("'robot_states' requested but not found under data/{} in {}".format(dk, fpath))
                        elif obs_key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                            # RoboCasa v0.1: direct keys in obs; else derive from robot_states
                            if obs_key in obs_group:
                                obs_dict[obs_key] = obs_group[obs_key][:].astype(np.float32)
                            elif robot_states is not None:
                                slices = {"robot0_eef_pos": (0, 3), "robot0_eef_quat": (3, 7), "robot0_gripper_qpos": (7, 9)}
                                lo, hi = slices[obs_key]
                                obs_dict[obs_key] = robot_states[:, lo:hi].astype(np.float32)
                            else:
                                raise KeyError("'{}' not in obs and robot_states not found in {}".format(obs_key, fpath))
                        elif obs_key in ("raw_language", "lang_fixed/language_distilbert"):
                            # Language: load from attrs
                            # Robosuite/Cosmos: task_description (str)
                            # RoboCasa v0.1: ep_meta (may be JSON str or dict) with task_description/lang
                            task_desc = demo_group.attrs.get("task_description", "")
                            if not task_desc and "ep_meta" in demo_group.attrs:
                                em = demo_group.attrs["ep_meta"]
                                if isinstance(em, dict):
                                    task_desc = em.get("task_description", em.get("lang", ""))
                                elif isinstance(em, (str, bytes)):
                                    s = em.decode("utf-8") if isinstance(em, bytes) else em
                                    try:
                                        em = json.loads(s)
                                        task_desc = em.get("task_description", em.get("lang", ""))
                                    except Exception:
                                        task_desc = ""
                                else:
                                    task_desc = ""
                            if isinstance(task_desc, bytes):
                                task_desc = task_desc.decode("utf-8")
                            task_desc = str(task_desc) if task_desc else ""
                            obs_dict["_lang_str"] = task_desc
                        else:
                            # generic fallback
                            if obs_key in obs_group:
                                obs_dict[obs_key] = obs_group[obs_key][:].astype(np.float32)
                            else:
                                raise KeyError("Obs key '{}' not found in {}".format(obs_key, fpath))

                    # Store lang string per demo (will be resolved to embedding below)
                    if "_lang_str" in obs_dict:
                        self._lang_data[demo_idx] = obs_dict["_lang_str"]
                        del obs_dict["_lang_str"]

                    # --- Actions (first 7 of 12D) ---
                    actions = f["data/{}/actions".format(dk)][:, :self.ACTION_DIM].astype(np.float32)

                    T = actions.shape[0]
                    self.demos.append((fpath, dk))
                    self.demo_lengths.append(T)
                    self._obs_data[demo_idx] = obs_dict
                    self._action_data[demo_idx] = actions
                    demo_idx += 1

        if len(self.demos) == 0:
            raise RuntimeError("No demonstrations loaded from {}".format(data_dir))
        if self.verbose:
            print("[RoboCasaDataset] loaded {} demos, total_samples={}".format(
                len(self.demos), sum(self.demo_lengths)))

        # ------------------------------------------------------------------
        # Compute / load normalisation statistics
        # ------------------------------------------------------------------
        self._action_stats = None
        if self.normalize_actions:
            def _get_proprio(obs_dict, length):
                if "robot_states" in obs_dict:
                    return obs_dict["robot_states"]
                # Build from eef_pos + eef_quat + gripper_qpos (RoboCasa v0.1)
                parts = []
                for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                    if k in obs_dict:
                        parts.append(obs_dict[k])
                if parts:
                    return np.concatenate(parts, axis=-1).astype(np.float32)
                return np.zeros((length, self.PROPRIO_DIM), dtype=np.float32)

            # Cache key from loaded demos: same demo set -> same stats file
            demo_sig = ",".join(sorted("{}::{}".format(fp, dk) for fp, dk in self.demos))
            cache_key = hashlib.md5(demo_sig.encode()).hexdigest()[:16]
            stats_path = os.path.join(_ROBOCASA_STATS_CACHE_DIR, "dataset_statistics_{}.json".format(cache_key))

            self._dataset_stats = load_or_compute_dataset_statistics(
                data_dir=_ROBOCASA_STATS_CACHE_DIR,
                data={i: {
                    "actions": self._action_data[i],
                    "proprio": _get_proprio(self._obs_data[i], self.demo_lengths[i]),
                }
                       for i in range(len(self.demos))},
                stats_path=stats_path,
            )
            # Normalise actions in-place
            a_min = self._dataset_stats["actions_min"]
            a_max = self._dataset_stats["actions_max"]
            for i in range(len(self.demos)):
                self._action_data[i] = rescale_array(self._action_data[i], a_min, a_max).astype(np.float32)

        # ------------------------------------------------------------------
        # Precompute / load lang embeddings (DistilBERT -> 768-d per command)
        # ------------------------------------------------------------------
        if self._use_lang and self._lang_data:
            unique_cmds = set(self._lang_data.values())
            unique_cmds.discard("")  # skip empty
            n_empty = sum(1 for v in self._lang_data.values() if not v)
            if n_empty == len(self._lang_data):
                print("RoboCasaDataset WARNING: all {} demos have empty task_description. "
                      "Lang embeddings will be zeros. Check HDF5 attrs: task_description or ep_meta.".format(len(self._lang_data)))
            elif n_empty > 0:
                print("RoboCasaDataset: {} demos have empty task_description (will use zero embedding).".format(n_empty))
            if unique_cmds:
                print("RoboCasaDataset: {} unique lang commands -> precomputing embeddings.".format(len(unique_cmds)))
                lang_cache_path = os.path.join(_ROBOCASA_LANG_CACHE_DIR, _LANG_CACHE_FILENAME)
                emb_dict = _compute_lang_embeddings(unique_cmds, lang_cache_path)
                # Map each demo_idx to its precomputed embedding
                _zero_emb = np.zeros(768, dtype=np.float32)
                for di, cmd in self._lang_data.items():
                    self._lang_embeddings[di] = emb_dict.get(cmd, _zero_emb)
            else:
                _zero_emb = np.zeros(768, dtype=np.float32)
                for di in self._lang_data:
                    self._lang_embeddings[di] = _zero_emb
        elif self._use_lang:
            _zero_emb = np.zeros(768, dtype=np.float32)
            for di in range(len(self.demos)):
                self._lang_embeddings[di] = _zero_emb

        # ------------------------------------------------------------------
        # Build global index map
        # ------------------------------------------------------------------
        self._build_index_map()

        print("RoboCasaDataset: loaded {} demos, {} total samples from {}".format(
            len(self.demos), len(self), data_dir))

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        demo_idx, index_in_demo = self._index_to_demo[index]
        T = self.demo_lengths[demo_idx]

        # Sequence range with padding
        seq_begin = max(0, index_in_demo - (self.n_frame_stack - 1))
        seq_end = min(T, index_in_demo + self.seq_length)
        begin_pad = max(0, (self.n_frame_stack - 1) - index_in_demo)
        end_pad = max(0, index_in_demo + self.seq_length - T)

        # Observations
        obs_dict = {}
        for key in self.obs_keys:
            if key in ("raw_language", "lang_fixed/language_distilbert"):
                # Return precomputed DistilBERT embedding (768,) as float32 ndarray
                emb = self._lang_embeddings.get(demo_idx)
                if emb is None:
                    emb = np.zeros(768, dtype=np.float32)
                obs_dict["lang_fixed/language_distilbert"] = emb
                continue
            arr = self._obs_data[demo_idx][key][seq_begin:seq_end]
            if begin_pad > 0 or end_pad > 0:
                arr = TensorUtils.pad_sequence(
                    {key: arr}, padding=(begin_pad, end_pad), pad_same=True
                )[key]
            obs_dict[key] = arr

        # Actions
        actions = self._action_data[demo_idx][seq_begin:seq_end]
        if begin_pad > 0 or end_pad > 0:
            actions = TensorUtils.pad_sequence(
                {"a": actions}, padding=(begin_pad, end_pad), pad_same=True
            )["a"]

        meta = {"obs": obs_dict, "actions": actions}
        return meta

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_dataset_sampler(self):
        return None

    def get_action_normalization_stats(self):
        """Return normalization stats compatible with robomimic's action_stats_to_normalization_stats."""
        from robomimic.utils.dataset import action_stats_to_normalization_stats
        if self._action_stats is not None:
            return self._action_stats

        # Compute running stats over all demos
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
