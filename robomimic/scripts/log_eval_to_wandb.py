#!/usr/bin/env python3
"""
log_eval_to_wandb.py — 把一次评测的结果回写到训练时的同一个 wandb run。

设计动机（解耦评测）：
  评测是 policy server(droid.sif) + sim client(libero/robocasa.sif) 双容器。成功率
  在 *client* 侧算出并写成 results.json；而 wandb_run_id / project / entity 都存在 ckpt
  里（且只有 droid.sif 装了 wandb + macros + API key）。所以最干净的做法是：评测跑完后，
  在 droid.sif 里跑这个脚本——直接读 ckpt 拿到 run 标识，用 resume="allow" 重连训练那个
  run，把 eval/* 指标 append 上去。无需改 websocket 协议，client 容器也无需装 wandb。

用法（见 slurm/eval_app.slurm 末尾自动调用）：
  python -m robomimic.scripts.log_eval_to_wandb \
      --ckpt /ckpts/model_epoch_N.pth --results /eval_logs/<run>/results.json

results.json 约定（由两个 repo 的 scripts/run_eval.py 写出）：
  {"benchmark","task","num_episodes","num_success","success_rate",
   ["avg_task_success_rate","per_task":[{"task_id","name","success_rate"}]]}

行为约定：
  * ckpt 无 wandb_run_id（训练时没开 wandb）→ 打印告警并 exit 0，绝不让评测作业失败。
  * wandb.init 失败（无网/无 key）→ 同样告警 exit 0。
"""

import argparse
import json
import os
import re
import sys


def _warn_skip(msg: str) -> None:
    print(f"[log_eval_to_wandb] 跳过 wandb 回写：{msg}", flush=True)
    sys.exit(0)


def _parse_eval_log(path: str) -> dict:
    """兜底：sif 内若仍是旧版 run_eval（不写 results.json），从 eval.log 文本里抠数字。
    libero / robocasa 两种格式都覆盖（per-task 不解析）。"""
    txt = open(path).read()
    def _find(pat):
        m = re.search(pat, txt, re.IGNORECASE)
        return m.group(1) if m else None
    eps = _find(r"Total episodes:\s*([0-9]+)")
    suc = _find(r"Total successes:\s*([0-9]+)")
    sr = _find(r"(?:Overall success rate|Success rate):\s*([0-9.]+)")
    if eps is None or suc is None:
        raise ValueError(f"无法从 eval.log 解析 episodes/successes: {path}")
    eps_i, suc_i = int(eps), int(suc)
    sr_f = float(sr) if sr is not None else (suc_i / eps_i if eps_i else 0.0)
    return {"benchmark": "?", "task": "?", "num_episodes": eps_i,
            "num_success": suc_i, "success_rate": sr_f}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="评测所用 ckpt（含 wandb_run_id / config）")
    parser.add_argument("--run-dir", required=True,
                        help="run_eval 的本次输出目录（含 results.json 或 eval.log）")
    parser.add_argument("--prefix", default="eval", help="wandb 指标前缀（默认 eval）")
    args = parser.parse_args()

    results_path = os.path.join(args.run_dir, "results.json")
    eval_log_path = os.path.join(args.run_dir, "eval.log")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
    elif os.path.isfile(eval_log_path):
        print(f"[log_eval_to_wandb] 无 results.json，回退解析 {eval_log_path}", flush=True)
        results = _parse_eval_log(eval_log_path)
    else:
        _warn_skip(f"run-dir 内无 results.json / eval.log: {args.run_dir}")

    import robomimic.utils.file_utils as FileUtils

    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.ckpt)
    run_id = ckpt_dict.get("wandb_run_id", None)
    if not run_id:
        _warn_skip(f"ckpt 无 wandb_run_id（训练未开 wandb？）: {args.ckpt}")

    # 从 ckpt 取 project / run 名
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=False)
    project = config.experiment.logging.wandb_proj_name
    run_name = config.experiment.name
    epoch = ckpt_dict.get("epoch", None)

    import robomimic.macros as Macros
    if getattr(Macros, "WANDB_API_KEY", None):
        os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY
    entity = getattr(Macros, "WANDB_ENTITY", None)

    try:
        import wandb
    except Exception as e:  # pragma: no cover
        _warn_skip(f"import wandb 失败: {e}")

    init_kwargs = dict(project=project, name=run_name, id=run_id, resume="allow")
    if entity is not None:
        init_kwargs["entity"] = entity
    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        _warn_skip(f"wandb.init 失败（无网/无 key/项目不匹配？）: {e}")

    p = args.prefix
    # eval/* 用 eval/epoch 作为 x 轴，便于和训练曲线按 epoch 对齐
    try:
        wandb.define_metric(f"{p}/epoch")
        wandb.define_metric(f"{p}/*", step_metric=f"{p}/epoch")
    except Exception:
        pass

    log_dict = {
        f"{p}/success_rate": float(results.get("success_rate", 0.0)),
        f"{p}/num_success": int(results.get("num_success", 0)),
        f"{p}/num_episodes": int(results.get("num_episodes", 0)),
    }
    if epoch is not None:
        log_dict[f"{p}/epoch"] = int(epoch)
    if "avg_task_success_rate" in results:
        log_dict[f"{p}/avg_task_success_rate"] = float(results["avg_task_success_rate"])
    # 单值 summary 也写一份，便于在 run overview / table 里直接看到
    task = results.get("task", "")
    for tnt in results.get("per_task", []) or []:
        name = str(tnt.get("name", tnt.get("task_id", "?")))
        safe = name.strip().replace("/", "_").replace(" ", "_")[:64] or str(tnt.get("task_id", "?"))
        log_dict[f"{p}/per_task/{safe}"] = float(tnt.get("success_rate", 0.0))

    wandb.log(log_dict)
    # summary：最近一次评测的关键数字（覆盖式，便于排序/对比）
    run.summary[f"{p}/last_success_rate"] = log_dict[f"{p}/success_rate"]
    run.summary[f"{p}/last_task"] = task
    if epoch is not None:
        run.summary[f"{p}/last_epoch"] = int(epoch)
    wandb.finish()
    print(
        f"[log_eval_to_wandb] 已写入 run_id={run_id} project={project} "
        f"task={task} epoch={epoch} success_rate={log_dict[f'{p}/success_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
