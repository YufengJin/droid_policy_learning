"""
RoboCasa rollout utilities aligned with benchmarks/robocasa/scripts/run_eval.py.

Provides run_eval-style environment creation, episode loop (NUM_WAIT_STEPS,
task-specific horizon, 7D->action_dim padding), and three-view video saving.
"""

import ast
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np

# RoboCasa-specific rollout config defaults. Suppress robosuite controller warning:
# "The config has defined for the controller X but the robot does not have this component"
# (PandaMobile has no left/head/legs; robosuite correctly removes them)
def _suppress_robosuite_controller_warnings():
    """Temporarily suppress robosuite controller mismatch warnings (benign for PandaMobile)."""
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
    _prev = ROBOSUITE_DEFAULT_LOGGER.level
    ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
    return _prev


def _restore_robosuite_log_level(prev_level):
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
    ROBOSUITE_DEFAULT_LOGGER.setLevel(prev_level)


# RoboCasa-specific rollout config defaults. Injected into config.experiment.rollout
# by train_robocasa.get_config_from_args() so YAML can override without polluting base_config.
ROBOCASA_ROLLOUT_CONFIG_DEFAULTS = {
    "layout_and_style_ids": None,
    "obj_instance_split": "B",
    "img_res": 224,
    "num_wait_steps": 10,
    "robots": "PandaMobile",
    "parallel_workers": 1,
}

# ---------------------------------------------------------------------------
# Task-specific max steps (from run_eval.py)
# ---------------------------------------------------------------------------
TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


def get_task_max_steps(task_name, default_horizon=500):
    """Return horizon for task, with optional lookup from dataset_registry."""
    if task_name in TASK_MAX_STEPS:
        return TASK_MAX_STEPS[task_name]
    try:
        from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
        all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
        if task_name in all_tasks and "horizon" in all_tasks[task_name]:
            return all_tasks[task_name]["horizon"]
    except ImportError:
        pass
    return default_horizon


def _get_lang_embedding_for_rollout(task_description):
    """Compute or load DistilBERT embedding for a single task description."""
    from robomimic.utils.robocasa_dataset import _compute_lang_embeddings, _ROBOCASA_LANG_CACHE_DIR, _LANG_CACHE_FILENAME
    cache_path = os.path.join(_ROBOCASA_LANG_CACHE_DIR, _LANG_CACHE_FILENAME)
    emb_dict = _compute_lang_embeddings([task_description] if task_description else [], cache_path)
    if task_description and task_description in emb_dict:
        return emb_dict[task_description]
    return np.zeros(768, dtype=np.float32)


def _get_task_description_from_env(env, task_name_fallback=None):
    """Get task description (lang) from environment. Handles both raw and wrapped envs.
    Traverses env.env chain to reach robosuite env (EnvRobosuite wraps robosuite.make output)."""
    inner = env
    while inner is not None:
        if hasattr(inner, "get_ep_meta"):
            meta = inner.get_ep_meta()
            if isinstance(meta, dict):
                lang = meta.get("lang") or meta.get("task_description") or ""
                if lang:
                    return str(lang)
        inner = getattr(inner, "env", None)
    return task_name_fallback or ""


def create_robocasa_env_for_rollout(
    env_name,
    robots="PandaMobile",
    img_res=224,
    obj_instance_split="B",
    layout_and_style_ids="((1,1),(2,2),(4,4),(6,9),(7,10))",
    seed=None,
    episode_idx=None,
    use_image_obs=True,
    postprocess_visual_obs=True,
):
    """
    Create a RoboCasa environment for rollout, matching run_eval's create_robocasa_env.

    Uses load_composite_controller_config, explicit camera_names, obj_instance_split,
    layout_and_style_ids. Returns EnvRobosuite compatible with robomimic RolloutPolicy.

    Args:
        env_name (str): RoboCasa task name (e.g. PnPCounterToCab)
        robots (str): Robot type
        img_res (int): Camera resolution (square)
        obj_instance_split (str): Object instance split (B = held-out test)
        layout_and_style_ids (str): String of ((1,1),(2,2),...) for scene selection
        seed (int | None): Random seed
        episode_idx (int | None): Used for layout selection: (episode_idx // 10) % len(ids)
        use_image_obs (bool): Include camera observations
        postprocess_visual_obs (bool): Postprocess images for policy input

    Returns:
        EnvRobosuite instance
    """
    try:
        import robocasa  # noqa: F401 - register RoboCasa envs with robosuite
    except ImportError:
        pass
    from robosuite.controllers import load_composite_controller_config
    from robomimic.envs.env_robosuite import EnvRobosuite

    layout_and_style_ids_parsed = None
    if layout_and_style_ids:
        all_ids = ast.literal_eval(layout_and_style_ids)
        if episode_idx is not None:
            scene_index = (episode_idx // 10) % len(all_ids)
            layout_and_style_ids_parsed = (all_ids[scene_index],)
        else:
            layout_and_style_ids_parsed = all_ids

    robot_str = robots if isinstance(robots, str) else robots[0]
    controller_configs = load_composite_controller_config(controller=None, robot=robot_str)

    kwargs = dict(
        robots=robots,
        controller_configs=controller_configs,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=img_res,
        camera_heights=img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=use_image_obs,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=None,
        randomize_cameras=False,
        translucent_robot=False,
    )
    if layout_and_style_ids_parsed is not None:
        kwargs["layout_and_style_ids"] = layout_and_style_ids_parsed

    prev_level = _suppress_robosuite_controller_warnings()
    try:
        env = EnvRobosuite(
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=use_image_obs,
            postprocess_visual_obs=postprocess_visual_obs,
            **kwargs,
        )
    finally:
        _restore_robosuite_log_level(prev_level)
    return env


def _detect_frame_stack(ob_dict):
    """Infer frame_stack from obs returned by a (possibly wrapped) env.
    FrameStackWrapper gives every key a leading (T, ...) dimension where T > 1."""
    for v in ob_dict.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return v.shape[0]
    return 1


def _match_frame_stack(arr, frame_stack):
    """Tile a 1-D array to (frame_stack, D) if frame_stack > 1."""
    if frame_stack <= 1 or arr.ndim >= 2:
        return arr
    return np.tile(arr, (frame_stack, 1))


def _obs_image_for_video(obs_val, frame_stack):
    """Extract last frame from frame-stacked, postprocessed obs for video.
    Handles (T, C, H, W) float [0,1] -> (H, W, C) uint8."""
    img = np.asarray(obs_val)
    if img.ndim >= 3 and frame_stack > 1 and img.shape[0] == frame_stack:
        img = img[-1]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def _pad_action_7d_to_env(action, env, policy_ac_dim=7):
    """Extend 7D policy output to env.action_dim (e.g. PandaMobile 11/12)."""
    env_dim = env.action_dimension
    if action.shape[-1] == policy_ac_dim and env_dim > policy_ac_dim:
        pad_dim = env_dim - policy_ac_dim
        mobile_base = np.zeros(pad_dim, dtype=np.float64)
        mobile_base[-1] = -1.0
        action = np.concatenate([action, mobile_base])
    return np.array(action, dtype=np.float64, copy=True)


def run_rollout_robocasa(
    policy,
    env,
    task_name,
    horizon=None,
    num_wait_steps=10,
    use_goals=False,
    render=False,
    video_writer=None,
    video_skip=5,
    terminate_on_success=True,
    default_horizon=500,
    lang_keys=None,
    verbose=False,
    worker_id=0,
):
    """
    Run a single RoboCasa episode with run_eval-style logic.

    - NUM_WAIT_STEPS dummy steps for object settling
    - Task-specific horizon from TASK_MAX_STEPS
    - 7D action padded to env.action_dim
    - Appends three-view frames to video_writer if provided

    Args:
        policy: RolloutPolicy instance
        env: EnvRobosuite or compatible
        task_name (str): For horizon lookup
        horizon (int | None): Override; if None uses TASK_MAX_STEPS
        num_wait_steps (int): Dummy steps after reset
        use_goals (bool): Goal-conditioned policy
        render (bool): Render to screen
        video_writer: imageio Writer or None; append primary|secondary|wrist frames
        video_skip (int): Frame sampling for video
        terminate_on_success (bool): Early stop on success
        default_horizon (int): Fallback horizon
        lang_keys (list | None): obs keys that need lang (e.g. ["lang_fixed/language_distilbert"])

    Returns:
        results (dict): Return, Horizon, Success_Rate, etc.
    """
    from robomimic.algo import RolloutPolicy
    from robomimic.envs.env_base import EnvBase
    from robomimic.envs.wrappers import EnvWrapper

    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    # RoboCasa uses obs from env (images + lang); avoid DiffusionPolicy's DROID eval_params path
    _prev_eval = getattr(policy, "eval_mode", True)
    policy.eval_mode = False

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        goal_dict = env.get_goal()

    max_steps = horizon if horizon is not None else get_task_max_steps(task_name, default_horizon)

    # Detect frame_stack from env output (FrameStackWrapper adds leading T dim)
    _fs = _detect_frame_stack(ob_dict)

    # Inject lang embedding if policy uses it (tiled to match frame_stack)
    lang_emb_stacked = None
    if lang_keys:
        task_desc = _get_task_description_from_env(env, task_name_fallback=task_name)
        lang_emb = _get_lang_embedding_for_rollout(task_desc)
        lang_emb_stacked = _match_frame_stack(lang_emb, _fs)
        for k in lang_keys:
            if k == "lang_fixed/language_distilbert":
                ob_dict[k] = lang_emb_stacked
                break

    # Wait for objects to settle (run_eval NUM_WAIT_STEPS)
    action_dim = getattr(env, "action_dimension", None)
    if action_dim is not None:
        dummy_action = np.zeros(action_dim, dtype=np.float64)
        for _ in range(num_wait_steps):
            ob_dict, _, _, _ = env.step(dummy_action)
            if lang_emb_stacked is not None:
                for k in lang_keys:
                    if k == "lang_fixed/language_distilbert":
                        ob_dict[k] = lang_emb_stacked
                        break

    results = {}
    total_reward = 0.0
    success = {"task": False}

    _log_interval = max(1, max_steps // 20)  # ~20 progress logs per episode
    try:
        step_i = 0
        for step_i in range(max_steps):
            if verbose and (step_i % _log_interval == 0 or step_i == max_steps - 1):
                print("  [worker_id={}] step: {}/{}".format(worker_id, step_i + 1, max_steps))
            # Get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # Pad 7D to env.action_dim (PandaMobile)
            ac = _pad_action_7d_to_env(ac, env)

            # Append three-view frame to video
            if video_writer is not None and step_i % video_skip == 0:
                p = ob_dict.get("robot0_agentview_left_image")
                s = ob_dict.get("robot0_agentview_right_image")
                w = ob_dict.get("robot0_eye_in_hand_image")
                if p is not None and s is not None and w is not None:
                    frame = np.concatenate(
                        [_obs_image_for_video(p, _fs),
                         _obs_image_for_video(s, _fs),
                         _obs_image_for_video(w, _fs)],
                        axis=1,
                    )
                    video_writer.append_data(frame)

            ob_dict, r, done, _ = env.step(ac)
            if lang_emb_stacked is not None:
                for k in lang_keys:
                    if k == "lang_fixed/language_distilbert":
                        ob_dict[k] = lang_emb_stacked
                        break

            if render:
                env.render(mode="human")

            total_reward += r
            cur_success = env.is_success()
            if isinstance(cur_success, dict):
                success["task"] = success["task"] or cur_success.get("task", False)
            else:
                success["task"] = success["task"] or bool(cur_success)

            if terminate_on_success and success["task"]:
                break

    except Exception as e:
        if hasattr(env, "rollout_exceptions") and isinstance(e, env.rollout_exceptions):
            print("WARNING: got rollout exception {}".format(e))
        else:
            raise
    finally:
        policy.eval_mode = _prev_eval

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    return results


def save_rollout_video_three_view(
    primary_images,
    secondary_images,
    wrist_images,
    output_path,
    fps=30,
):
    """Save concatenated MP4 of primary | secondary | wrist camera views (run_eval style)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, format="FFMPEG", codec="libx264")
    for p, s, w in zip(primary_images, secondary_images, wrist_images):
        frame = np.concatenate([p, s, w], axis=1)
        writer.append_data(frame)
    writer.close()
    return output_path


def _run_single_episode(args):
    """Worker helper for parallel rollout. Runs one episode and returns (ep_i, info)."""
    (ep_i, policy, env, task_name, horizon, num_wait_steps, use_goals, render,
     video_writer, video_skip, terminate_on_success, default_horizon, lang_keys,
     verbose, worker_id) = args
    t0 = time.time()
    info = run_rollout_robocasa(
        policy=policy,
        env=env,
        task_name=task_name,
        horizon=horizon,
        num_wait_steps=num_wait_steps,
        use_goals=use_goals,
        render=render,
        video_writer=video_writer,
        video_skip=video_skip,
        terminate_on_success=terminate_on_success,
        default_horizon=default_horizon,
        lang_keys=lang_keys,
        verbose=verbose,
        worker_id=worker_id,
    )
    info["time"] = time.time() - t0
    return ep_i, info


def rollout_with_stats_robocasa(
    policy,
    envs,
    horizon_map=None,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=True,
    verbose=False,
    num_wait_steps=10,
    default_horizon=500,
    lang_keys=None,
    parallel_workers=1,
    env_factory=None,
    env_pools=None,
):
    """
    RoboCasa-style rollout aggregation. Same return shape as TrainUtils.rollout_with_stats.

    Args:
        policy: RolloutPolicy instance
        envs (dict): env_name -> env (main env per task)
        horizon_map (dict | None): env_name -> horizon override
        num_episodes (int): Episodes per env
        parallel_workers (int): Number of parallel workers; 1=serial. When >1, requires env_factory or env_pools.
        env_factory (callable | None): (env_name) -> env. Used to create extra envs when env_pools not provided.
        env_pools (dict | None): env_name -> [env1, env2, ...]. Pre-created pool; if provided, no create/close per rollout.
        ... (others match rollout_with_stats)

    Returns:
        all_rollout_logs (dict): env_name -> {Return, Horizon, Success_Rate, Time_Episode, ...}
        video_paths (dict): env_name -> path
    """
    import robomimic.utils.log_utils as LogUtils

    all_rollout_logs = OrderedDict()
    video_paths = OrderedDict()
    write_video = video_dir is not None

    for env_name, env in envs.items():
        rollout_logs = []
        horizon = (horizon_map or {}).get(env_name) if horizon_map else None
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4"
        vid_path = os.path.join(video_dir, "{}{}".format(env_name, video_str)) if write_video else None
        main_video_writer = imageio.get_writer(vid_path, fps=20) if vid_path else None

        if write_video:
            video_paths[env_name] = vid_path
            print("video writes to " + vid_path)

        # Use pre-created pool if available; else create via env_factory
        has_pool = env_pools is not None and env_name in env_pools and len(env_pools[env_name]) > 1
        has_factory = parallel_workers > 1 and env_factory is not None
        use_parallel = (has_pool or has_factory) and parallel_workers > 1
        n_workers = min(parallel_workers, num_episodes) if use_parallel else 1
        if use_parallel and n_workers < parallel_workers:
            print(">> parallel_workers={} capped to {} (num_episodes={})".format(
                parallel_workers, n_workers, num_episodes))

        if has_pool:
            env_pool = env_pools[env_name][:n_workers]  # Use pre-created pool (no create/close)
            n_workers = len(env_pool)
            use_parallel = n_workers > 1
            if use_parallel:
                print(">> Using pre-created env pool ({} envs)".format(n_workers))
        else:
            env_pool = [env]
            if use_parallel and env_factory is not None:
                print(">> Creating {} extra envs for parallel rollout (target {} workers)...".format(
                    n_workers - 1, n_workers))
                t_create = time.time()
                for w_i in range(n_workers - 1):
                    try:
                        extra_env = env_factory(env_name)
                        env_pool.append(extra_env)
                    except Exception as e:
                        print(">> WARNING: env_factory failed (worker {}): {}. Falling back to {} workers.".format(
                            w_i + 1, e, len(env_pool)))
                        break
                n_workers = len(env_pool)
                use_parallel = n_workers > 1
                if use_parallel:
                    print(">> {} envs ready in {:.1f}s".format(n_workers, time.time() - t_create))

        n_batches = (num_episodes + n_workers - 1) // n_workers if use_parallel else num_episodes
        _horizon_str = horizon or get_task_max_steps(env_name, default_horizon)
        print("rollout: env={}, horizon={}, episodes={}, workers={}{})".format(
            env_name, _horizon_str, num_episodes, n_workers,
            " ({} batches x {} parallel".format(n_batches, n_workers) if use_parallel else " (serial",
        ))

        if use_parallel:
            rollout_results = [None] * num_episodes
            t_parallel = time.time()
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for batch_start in range(0, num_episodes, n_workers):
                    batch_end = min(batch_start + n_workers, num_episodes)
                    futures = []
                    for i, ep_i in enumerate(range(batch_start, batch_end)):
                        worker_env = env_pool[i]
                        vw = main_video_writer if ep_i == 0 else None
                        args = (ep_i, policy, worker_env, env_name, horizon, num_wait_steps, use_goals, render,
                                vw, video_skip, terminate_on_success, default_horizon, lang_keys,
                                verbose, i)
                        futures.append(executor.submit(_run_single_episode, args))
                    for fut in futures:
                        ep_i, info = fut.result()
                        rollout_results[ep_i] = info
                    done_count = min(batch_end, num_episodes)
                    successes = sum(1 for r in rollout_results[:done_count] if r and r.get("Success_Rate", 0) > 0)
                    print("  batch {}/{}: {}/{} episodes done, {} success so far".format(
                        batch_start // n_workers + 1, n_batches, done_count, num_episodes, successes))
            rollout_logs = rollout_results
            print(">> Parallel rollout finished in {:.1f}s (vs ~{:.0f}s serial estimate)".format(
                time.time() - t_parallel,
                sum(r["time"] for r in rollout_results if r)))
            # Only close envs that were created this rollout (via env_factory), not pre-created pool
            if not has_pool:
                for i in range(1, len(env_pool)):
                    if hasattr(env_pool[i], "close"):
                        try:
                            env_pool[i].close()
                        except Exception:
                            pass
        else:
            # Serial
            iterator = range(num_episodes)
            if not verbose:
                iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)
            for ep_i in iterator:
                t0 = time.time()
                info = run_rollout_robocasa(
                    policy=policy,
                    env=env,
                    task_name=env_name,
                    horizon=horizon,
                    num_wait_steps=num_wait_steps,
                    use_goals=use_goals,
                    render=render,
                    video_writer=main_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                    default_horizon=default_horizon,
                    lang_keys=lang_keys,
                    verbose=verbose,
                    worker_id=ep_i,
                )
                info["time"] = time.time() - t0
                rollout_logs.append(info)

        if main_video_writer is not None:
            main_video_writer.close()

        # Aggregate (match TrainUtils.rollout_with_stats format)
        keys = list(rollout_logs[0].keys())
        agg = {}
        for k in keys:
            vals = [r[k] for r in rollout_logs]
            if k == "time":
                agg["time"] = float(np.mean(vals))
                agg["Time_Episode"] = np.sum(vals) / 60.0
            else:
                agg[k] = float(np.mean(vals))
        all_rollout_logs[env_name] = agg

    return all_rollout_logs, video_paths
