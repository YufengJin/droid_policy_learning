# `slurm/` — 在 FAU NHR **Alex** 上训练 / 评测指南

本目录的 SLURM 脚本用 **apptainer** 容器在 Alex（a100 分区）上跑训练与跨容器评测,集群无外网也能运行(源码/素材已烤入镜像,依赖走代理)。

```
slurm/
├── train_libero_app.slurm          # 训练：LIBERO（4×A100，multirun drift+diffusion）
├── train_robocasa_app.slurm        # 训练：RoboCasa
├── train_droid_app.slurm           # 训练：DROID（RLDS）
├── eval_app.slurm                  # 评测：跨容器 client/server（libero 或 robocasa）
├── train_drift_vs_diffusion.slurm  # 实验：array job，2 数据×2 算法×3 seed = 12（见 drift_vs_diffusion_exp.md）
├── train_action_space_comparison.slurm # 实验：array job，2 数据×3 动作空间×3 seed = 18（见 action_space_exp.md）
├── patches/                        # 评测用的 websocket server/client 补丁（关 keepalive ping）
└── .env.wandb                      # wandb key：内容须为 `export WANDB_API_KEY=...`（mode 600）
```

> 训练 / resume / 评测的**冒烟测试逐步流程**见仓库根的 `cluster_test_plan.md`(交互式 srun 版,3 步即停)。本文是**正式作业(sbatch)**的参考手册。

---

## 0. 架构速览

- **训练**:单容器 `droid.sif`,`torchrun --nproc_per_node=4` 4 卡 DDP。三种数据各一个入口(`train_libero` / `train_robocasa` / `train_droid`),都是 Hydra 配置,可命令行覆盖任意键。
- **评测**:**解耦双容器**,同一节点:
  - `droid.sif` 起 **policy server**(GPU,`robomimic.scripts.policy_server`)
  - `<benchmark>.sif`(`libero.sif`/`robocasa.sif`)起 **sim client**(CPU osmesa 渲染,`scripts/run_eval.py`)
  - 两者经 **websocket + msgpack** 在 `localhost:$PORT` 通信(包 `policy_websocket`,三个镜像须同一 commit,见 `cluster_test_plan.md` §2B)。

---

## 1. 前置条件

| 项 | 路径 / 值 |
|---|---|
| `$WORK`(大盘,放 sif/数据/缓存) | `$HOME/atuin` = `/home/hpc/g108ea/g108ea10/atuin` |
| 镜像 | `$WORK/sif/{droid,libero,robocasa}.sif` |
| 数据集 | `$WORK/datasets/{libero/libero_10, robocasa/v0.1/..., droid}` |
| 代码仓库(`$REPO`) | `/home/hpc/g108ea/g108ea10/vault/vla-wam/droid_policy_learning` |
| wandb key | `$REPO/slurm/.env.wandb`,内容 `export WANDB_API_KEY=xxxx`(脚本会 `source`) |

```bash
export WORK=$HOME/atuin
export REPO=/home/hpc/g108ea/g108ea10/vault/vla-wam/droid_policy_learning
```

**`.env.wandb` 格式很重要**:必须是 `export WANDB_API_KEY=<key>`(裸 key `source` 不生效)。建立:
```bash
umask 077 && echo 'export WANDB_API_KEY=你的key' > $REPO/slurm/.env.wandb
```
不想在线 log wandb:训练命令尾部加 `experiment.logging.log_wandb=false` 即可。

---

## 2. 代理 / wandb(集群无外网必读)

所有脚本已内置:宿主导出 `http_proxy`/`https_proxy`(`http://proxy.nhr.fau.de:80`),并透传进容器(`APPTAINERENV_*`)。**评测**额外设了 `no_proxy=localhost,127.0.0.1`(否则 server↔client 的 localhost websocket 会被代理 403 拒)。细节见 `cluster_test_plan.md` §0″。

---

## 3. 训练(`train_{libero,robocasa,droid}_app.slurm`)

所有训练脚本:`#SBATCH --partition=a100 --gres=gpu:4 --constraint=a100_80 --time=24:00:00`,日志落 `logs/%j.{out,err}`,checkpoint 落 `$PROJECT_ROOT/outputs/<exp>/<ts>/models/`(每 ~1h 存一次,见 yaml `experiment.save.every_n_seconds=3600`)。

> ℹ️ batch size 已按 **A100-80GB** 调优:libero `512`/GPU、robocasa `320`/GPU(脚本变量 `LIBERO_BATCH`/`ROBOCASA_BATCH` 可覆盖)。若被调度到 **a100_40** 节点需手动降到 libero≤320 / robocasa≤192(见 §3.1)。

### 提交
```bash
cd $REPO
sbatch slurm/train_libero_app.slurm        # LIBERO   （默认 multirun: drift_policy + diffusion_policy）
sbatch slurm/train_robocasa_app.slurm      # RoboCasa （默认 multirun）
sbatch slurm/train_droid_app.slurm         # DROID（RLDS）
squeue --me
```

### 可覆盖的环境变量(`--export=ALL,KEY=VAL,...`)

| 脚本 | 变量(默认) | 作用 |
|---|---|---|
| 通用 | `DROID_PROJECT_DIR`($REPO) | 代码/输出根;输出落 `$DROID_PROJECT_DIR/outputs` |
| 通用 | `APPTAINER_IMAGE`($WORK/sif/droid.sif) | 换镜像 |
| 通用 | `HF_HOME`($HOME/atuin/huggingface) | HF 缓存(可写) |
| libero | `LIBERO_DATASET_DIR`,`LIBERO_SUBDIR`(libero_10) | 数据根 / 子集 |
| libero | `LIBERO_MULTIRUN`(1),`LIBERO_ALGOS`("drift_policy diffusion_policy"),`LIBERO_ALGO`(diffusion_policy) | 多算法顺序跑 / 单算法 |
| robocasa | `ROBOCASA_DATASET_DIR`,`ROBOCASA_SUBDIR`(v0.1/single_stage) | 数据根 / 子集 ⚠️ 默认 single_stage 全量(809GB)会 RAM-OOM,长训请指到单 task,见 §3.2 |
| robocasa | `ROBOCASA_MULTIRUN`/`ROBOCASA_ALGOS`/`ROBOCASA_ALGO` | 同上 |
| droid | `DROID_DATASET_DIR`,`DROID_DATASET_NAMES`(insert_pin) | RLDS 根 / 子集名(对应 `train_rlds.yaml` 的 `dataset_names`) |

**例**:只跑 RoboCasa 单算法、换子集:
```bash
sbatch --export=ALL,ROBOCASA_MULTIRUN=0,ROBOCASA_ALGO=diffusion_policy,ROBOCASA_SUBDIR=v0.1/single_stage/kitchen_pnp \
  slurm/train_robocasa_app.slurm
```

> ⚠️ **输出盘**:默认 `outputs/` 在仓库(vault,配额小)。长训建议把 ckpt 引到 `$WORK`(大盘):
> `sbatch --export=ALL,DROID_PROJECT_DIR=$REPO slurm/train_robocasa_app.slurm` 后把 `outputs` 软链到 `$WORK`,或直接改脚本里 `train.output_dir`。

### 任意 Hydra 覆盖
训练入口是 Hydra,脚本末尾 `torchrun ... -m robomimic.scripts.train_<x> <overrides>`。要改超参,直接在脚本的那行追加 `key=val`(如 `train.batch_size=256 train.num_epochs=3000`)。

### 3.1 batch size 与 GPU 显存(A100-40GB 实测,2026-06-29)

**`train.batch_size` 是 _per-GPU_ 的**(代码打印 `effective batch_size = batch_size × world_size`):DDP 下每张卡各加载 `batch_size` 个样本,**加更多卡不会降低单卡显存**。所以单卡能否放下,只看 per-GPU `batch_size`。

> ⚠️ **重要:训练脚本会「吞掉」OOM** —— OOM 时打印 `run failed with error: CUDA out of memory` 但**进程仍 exit 0**。所以一个「跑完」的作业可能其实 OOM 了却看着像成功。判断真假只看日志里有没有 **`finished run successfully!`**。

单卡 A100-40GB 实测(每个 batch 跑 5 步,看 `finished` + nvidia-smi 峰值;数据:libero_10 `KITCHEN_SCENE3`、robocasa `multi_stage/PrepareCoffee/demo_im128`):

| 数据 | per-GPU batch | 结果 | 显存峰值 |
|---|---|---|---|
| **libero** | 256 | ✅ 跑完 | 32.1 GB (78%) |
| **libero** | **320** | ✅ 跑完(**实测上限**) | 39.9 GB (98%) |
| **libero** | 384 / **512** / 768 / 1024 | ❌ **OOM** | — |
| **robocasa** | **128** | ✅ 跑完 | 25.2 GB (62%) |
| **robocasa** | **192** | ✅ 跑完(**推荐**) | 34.4 GB (84%) |
| **robocasa** | 256 / 384 / 512 | ❌ **OOM** | — |

**结论 / 推荐(A100-40GB)**:
- ❗ **`train_libero_app.slurm` 当前默认 `train.batch_size=512` 会 OOM**(单卡 512 放不下)。在 a100_40 上请改 **256**(78%,有余量)或最多 **320**(98%,贴边、风险高)。
- `train_robocasa_app.slurm` 默认 `128` **偏保守**(仅 62%);可提到 **192**(84%)换约 1.5× 吞吐;**256 会 OOM**。
- 提交时直接覆盖即可,例如:
  ```bash
  # 在脚本 torchrun 那行把 train.batch_size 改掉，或用 Hydra override 重跑单卡 smoke 验证：
  #   ... -m robomimic.scripts.train_libero  ... train.batch_size=256
  #   ... -m robomimic.scripts.train_robocasa ... train.batch_size=192
  ```
- 显存峰值还受**数据**影响(图像分辨率、`frame_stack`、action horizon);换数据子集(如 robocasa `single_stage/kitchen_pnp`、array job 用的子集)上限可能略移,建议换数据后用 §3.1 的 5 步 smoke 复测一次。a100_80 节点大致可翻倍。
- 复测脚本:`$WORK/resume_test/bs_sweep.sh`(改 batch 列表即可)。

### 3.2 Modality 消融实验:robot_state vs 无 proprio(4 个对比任务)

对比「带本体 proprio」与「纯视觉(+语言)」两种 modality 下的表现。两个数据集各 2 个 run,共 **4 个实验**:

| # | 数据 | proprio | low_dim 内容 | 推荐 EXP_NAME |
|---|---|---|---|---|
| 1 | libero_10 | ✅ 有 | `[robot_states]`(yaml 默认) | `libero10_with_proprio` |
| 2 | libero_10 | ❌ 无 | `[]` | `libero10_no_proprio` |
| 3 | robocasa | ✅ 有 | `[robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos]`(9D) | `robocasa_with_proprio` |
| 4 | robocasa | ❌ 无 | `[]`(yaml 默认) | `robocasa_no_proprio` |

控制 modality 的开关(脚本已内置,见 `train_*_app.slurm` 的 `EXTRA_OVERRIDES`):

| 变量 | 语义 |
|---|---|
| `LIBERO_LOW_DIM` **未设置** | 用 yaml 默认 `low_dim=[robot_states]`(= 有 proprio) |
| `LIBERO_LOW_DIM=""`(空串) | `low_dim=[]`(纯视觉) |
| `ROBOCASA_LOW_DIM` **未设置** | 用 yaml 默认 `low_dim=[]`(纯视觉+语言) |
| `ROBOCASA_LOW_DIM="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos"` | 加入 9D proprio |
| `EXP_NAME` | 覆盖 wandb / 实验名,用于区分 4 个 run |

> ⚠️ **robocasa 数据范围**:full `v0.1/single_stage`(809GB im128)**会被 OOM-Killed**(每个 DDP rank 各自 eager-load 全量到 RAM,见 §3.1 末/[memory])。必须用单 category 或单 task,例如 `v0.1/single_stage/kitchen_pnp/PnPSinkToCounter`(~35GB,可放下)。两个 robocasa run 务必用**同一个** `ROBOCASA_SUBDIR`,否则对比不干净。
>
> ⚠️ **部分 hdf5 是损坏(截断)的**(下载不全):`PnPCounterToCab`/`PnPCounterToMicrowave`/`PnPCounterToSink` 的某个文件 truncated,一开训就 `OSError: ... truncated file`。**已验证完好**的 kitchen_pnp 单 task:`PnPSinkToCounter`(~35G,最小)、`PnPStoveToCounter`、`PnPCabToCounter`、`PnPCounterToStove`、`PnPMicrowaveToCounter`。换子集前先 header-open 校验:容器内 `python3 -c "import h5py,glob;[h5py.File(f,'r').close() for f in glob.glob('<dir>/**/*im128*.hdf5',recursive=True)]"`。
>
> ⚠️ **`ROBOCASA_LOW_DIM` 含逗号**,会和 `--export=ALL,K=V,...` 的逗号分隔符冲突。**必须先 `export` 再 `sbatch --export=ALL`**(见下),不要写成 `--export=ALL,ROBOCASA_LOW_DIM=a,b,c`。
>
> ✅ optimizer 保持不变(Adam, lr=1e-4):batch 仅 ~1.6× 增幅,为保证 modality 消融干净不动 lr。

#### A) Slurm 提交(正式作业,4×A100,24h)

```bash
cd $REPO
RC_SUB=v0.1/single_stage/kitchen_pnp/PnPSinkToCounter   # robocasa 共用子集(~35GB)

# 1) libero_10 + proprio（LIBERO_LOW_DIM 不设 → yaml 默认 [robot_states]）
sbatch --export=ALL,EXP_NAME=libero10_with_proprio slurm/train_libero_app.slurm

# 2) libero_10 无 proprio（空串 → low_dim=[]）
sbatch --export=ALL,LIBERO_LOW_DIM=,EXP_NAME=libero10_no_proprio slurm/train_libero_app.slurm

# 3) robocasa + proprio（含逗号，先 export 再 --export=ALL）
export ROBOCASA_LOW_DIM="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos"
export ROBOCASA_SUBDIR="$RC_SUB"; export EXP_NAME=robocasa_with_proprio
sbatch --export=ALL slurm/train_robocasa_app.slurm
unset ROBOCASA_LOW_DIM EXP_NAME                          # 防止泄漏到下一条

# 4) robocasa 无 proprio（ROBOCASA_LOW_DIM 不设 → yaml 默认 []）
sbatch --export=ALL,ROBOCASA_SUBDIR=$RC_SUB,EXP_NAME=robocasa_no_proprio slurm/train_robocasa_app.slurm

squeue --me
```

提交后确认 4 个作业各自**真正开始训练**(不是秒退/OOM),逐个 tail 主日志:
```bash
tail -f $REPO/logs/<jobid>.out
# 健康标志:看到 EXTRA_OVERRIDES=... 正确 → 数据加载完 → 进入 epoch 循环打印 loss。
# OOM 假成功:日志只到 "run failed with error: CUDA out of memory" 却 exit 0(见 §3.1)。
```

#### B) Interactive 提交(salloc,适合先冒烟一遍再放 24h 作业)

先要一个交互节点,再**直接复用同一套 `train_*_app.slurm` 脚本**(SBATCH 行只是注释,salloc 下照样跑;env 变量手动 `export`):

```bash
cd $REPO
salloc --partition=a100 --gres=gpu:4 --constraint=a100_80 --cpus-per-task=8 --time=02:00:00
# ↑ 进入计算节点后:

# 1) libero_10 + proprio
EXP_NAME=libero10_with_proprio bash slurm/train_libero_app.slurm

# 2) libero_10 无 proprio
LIBERO_LOW_DIM="" EXP_NAME=libero10_no_proprio bash slurm/train_libero_app.slurm

# 3) robocasa + proprio
ROBOCASA_LOW_DIM="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos" \
  ROBOCASA_SUBDIR=v0.1/single_stage/kitchen_pnp/PnPSinkToCounter \
  EXP_NAME=robocasa_with_proprio bash slurm/train_robocasa_app.slurm

# 4) robocasa 无 proprio
ROBOCASA_SUBDIR=v0.1/single_stage/kitchen_pnp/PnPSinkToCounter \
  EXP_NAME=robocasa_no_proprio bash slurm/train_robocasa_app.slurm
```
- interactive 下变量直接写在命令前(`KEY=VAL bash ...`),**逗号不再冲突**(没有 `--export` 解析),所以 `ROBOCASA_LOW_DIM` 可直接内联。
- 冒烟可加 Hydra 覆盖快速跑几步:在脚本 `torchrun` 行末尾不便改时,改用直接调 `python -m robomimic.scripts.train_libero ... train.num_epochs=1 experiment.epoch_every_n_steps=5 experiment.logging.log_wandb=false`;或直接信任 §3.1 已测的 batch,salloc 仅用于「确认 4 条命令能进 epoch 循环」即 Ctrl-C。
- 占着 4 卡跑满 24h 不划算 → 交互验证通过后,用上面 **A) Slurm** 正式提交。

---

## 4. resume 续训(同一个 wandb run id 接着练)

`train_libero` / `train_robocasa` / `train_droid` 均支持从 ckpt 完整 resume,**并复用同一个 wandb run**(机制:`save_model` 把 `wandb_run_id` 写进 ckpt;resume 时 `wandb.init(id=…, resume="allow")` 重接)。aloha 暂不支持。

用 Hydra 覆盖两个键即可:
```bash
... experiment.ckpt_path=/workspace/.../models/model_epoch_N.pth experiment.resume=true
```
- `resume=true` → 恢复 model+optimizer+scheduler+epoch,从 `epoch N+1` 开始;wandb 续接同一 run。
- `resume=false`(默认)+ `ckpt_path` → **仅加载权重做微调**(epoch 从 1、wandb 新开 run)。
- **前提**:被 resume 的 ckpt 必须是「开着 wandb」训出来的(否则没存 `wandb_run_id`)。

**实测通过(2026-06-29)**:libero(run `a34mg8aa`)、robocasa(run `5n9xejmb`)续训日志均出现 `RESUME: starting from epoch 2`,wandb 网页同一 run 显示 `resumed`。逐步脚本见 `$WORK/resume_test/`,完整记录见 `cluster_test_plan.md` §3B。

---

## 5. 评测(`eval_app.slurm`,跨容器)

`#SBATCH --partition=a100 --gres=gpu:1 --time=04:00:00`。同节点起 server(droid.sif)+ client(bench.sif),跑完 trap 清理(杀 server、删 overlay)。结果落 `$PROJECT_ROOT/eval_logs/<TASK>--<ts>/eval.log`。

### 提交(⚠️ 两个必带项见下)
```bash
cd $REPO
# robocasa（最快，单任务）
RC_CKPT=$(ls -t $WORK/test_outputs/diffusion_policy_demo_im128_*/*/models/model_epoch_3.pth | head -1)
sbatch --export=ALL,DROID_PROJECT_DIR=$REPO,WORK=$WORK,BENCHMARK=robocasa,CKPT=$RC_CKPT,NUM_TRIALS=1 slurm/eval_app.slurm

# libero（跑完 libero_10 全 10 任务；NUM_TRIALS=每任务次数）
LB_CKPT=$(ls -t $WORK/test_outputs/diffusion_policy_*libero*/*/models/model_epoch_3.pth | head -1)
sbatch --export=ALL,DROID_PROJECT_DIR=$REPO,WORK=$WORK,BENCHMARK=libero,CKPT=$LB_CKPT,NUM_TRIALS=1,PORT=8770 slurm/eval_app.slurm
```

### 必传 / 可覆盖变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `BENCHMARK` | (必填) | `libero` 或 `robocasa` |
| `CKPT` | (必填) | 宿主机 `.pth` 绝对路径 |
| **`DROID_PROJECT_DIR`** | `$SCRIPT_DIR/..` | **必传 `=$REPO`**(原因见下 ④c) |
| **`WORK`** | (必须可见) | sif 目录 `$WORK/sif`;建议显式 `WORK=$WORK` |
| `PORT` | 8765 | server/client 端口。**并发跑两个 eval 时务必错开**(见 ④e) |
| `NUM_TRIALS` | libero 20 / robocasa 5 | libero=每任务次数(`--num_trials_per_task`);robocasa=总次数 |
| `TASK_SUITE` | libero_10 | (libero)任务套件 |
| `TASK` | PnPCounterToCab | (robocasa)单任务名 |
| `EVAL_LOGS_DIR` | `$PROJECT_ROOT/eval_logs` | 结果落盘 |
| `DROID_SIF`/`BENCH_SIF`/`SIF_DIR` | `$WORK/sif/*` | 换镜像 |

### 查结果 & 通过标准
```bash
tail -f $REPO/logs/<jobid>.out          # 主日志
cat $REPO/logs/<jobid>_server.log       # server 端日志
```
日志依次出现即为**链路通**:
`[INFO] server ready` → `Connecting to policy server at ws://localhost:<PORT>` → `Evaluating task: …` → `Episode N: …(length=…)` → `Total episodes` → `评测完成`。
> 成功率 0% 属预期(只验链路,不看效果)。**实测通过(2026-06-29)**:robocasa job 3794844(`eval_logs/PnPCounterToCab--20260629_102937`)、libero job 3794910(`eval_logs/libero_10--20260629_103731`,10 任务)。

---

## 6. 批量实验(array job)

两个 array 脚本把 (数据×算法/动作空间×seed) 展开成多个 array task,各自 4×A100、5000 epoch。详见同目录:
- `drift_vs_diffusion_exp.md` —— `train_drift_vs_diffusion.slurm`(12 runs,batch 256)
- `action_space_exp.md` —— `train_action_space_comparison.slurm`(18 runs,batch 64)

```bash
sbatch slurm/train_drift_vs_diffusion.slurm                 # 全部 12（同时最多 4 个：--array=0-11%4）
sbatch --array=0-5%3 slurm/train_drift_vs_diffusion.slurm   # 仅 LIBERO 6 个
sbatch --array=4     slurm/train_drift_vs_diffusion.slurm   # 单次：libero drift seed=42
# 用环境变量覆盖 array 映射（注意 --export=NONE 时 NONE 要写最前）：
sbatch --array=0 --export=NONE,DATASET=libero,ALGO=drift,SEED=42 slurm/train_drift_vs_diffusion.slurm
```
Array ID 映射(drift_vs_diffusion):`ID = dataset_idx*6 + algo_idx*3 + seed_idx`,dataset=[libero,robocasa]、algo=[diffusion,drift]、seed=[1,42,123]。
action_space:`ID = dataset_idx*9 + space_idx*3 + seed_idx`,space=[pos_euler,pos_rot6d,pos_axisangle]。

---

## 7. 产物落点

| 内容 | 路径 |
|---|---|
| 训练 ckpt(周期) | `$PROJECT_ROOT/outputs/<exp>/<ts>/models/model_epoch_N.pth` —— **每 ~1h 存一个**(按 `every_n_seconds=3600`),累积;含 `wandb_run_id`/`optimizer`/`epoch`,可 resume |
| 训练 ckpt(最优) | `.../models/model_best.pth` —— **单个、覆盖式**(验证 loss / rollout 成功率创新高时重写;无验证的 run 不产生)。2026-06-30 起 3 个 train 脚本统一此行为(不再累积 `model_epoch_N_best_validation_*.pth`) |
| 训练日志(作业) | `$REPO/logs/<jobid>.out` / `.err`;array 为 `<jobid>_<arrayid>` |
| 评测结果 | `$PROJECT_ROOT/eval_logs/<TASK>--<ts>/eval.log` |
| 评测 server 日志 | `$REPO/logs/<jobid>_server.log` |
| wandb(在线) | `https://wandb.ai/yufeng-jin/<wandb_proj_name>/runs/<id>` |

---

## 8. 故障排查(集群高发)

| 现象 | 处理 |
|---|---|
| `GLIBC_2.38 not found`(import cv2/mujoco) | `--nv` 注入宿主 GL 与容器 glibc 冲突。训练/server 用 `LD_LIBRARY_PATH=容器库优先`,client 用 `MUJOCO_GL=osmesa` 不加 `--nv`(脚本已处理)。 |
| `Read-only file system` | sif 只读;训练/server 已带 `--writable-tmpfs`,client 用 overlay(脚本已处理)。 |
| `Permission denied: …/huggingface` | 用 `HF_HOME=/hf` + bind 可写目录(脚本已处理),确认 `$WORK/huggingface` 可写。 |
| wandb / HF 下载卡住 | 代理没透传进容器。见 §2;临时关 wandb:`experiment.logging.log_wandb=false`。 |
| **④c** 评测秒退 `mkdir …/var/tmp/slurmd_spool/eval_logs: Permission denied` | slurm 在 spool 跑脚本副本,`BASH_SOURCE`→spool 使 `PROJECT_ROOT` 算错。**提交时加 `DROID_PROJECT_DIR=$REPO,WORK=$WORK`**。 |
| **④d** libero client `TypeError: create_connection() got an unexpected keyword argument 'ping_interval'` | `patches/websocket_client.py` 的 `ping_interval` 只在 websockets≥16(robocasa)是 connect() 形参;libero 是 13.1。**已修**:按 `inspect.signature` 条件传入。 |
| **④e** 评测 `端口 8765 被占用`,server 进程退出 | 两个 eval 作业被调到同一节点、server 都绑 8765。**并发跑给不同 `PORT`**(如 8765/8770)或串行。 |
| wandb resume 没接上同一 run | 确认被 resume 的 ckpt 是开 wandb 训的(含 `wandb_run_id`),且传了 `experiment.resume=true`。aloha 暂不支持。 |

---

## 9. 一行速查

```bash
# 训练
cd $REPO && sbatch slurm/train_{libero,robocasa,droid}_app.slurm
# modality 消融 4 连发（slurm；robocasa 含逗号变量先 export，见 §3.2）
sbatch --export=ALL,EXP_NAME=libero10_with_proprio                  slurm/train_libero_app.slurm
sbatch --export=ALL,LIBERO_LOW_DIM=,EXP_NAME=libero10_no_proprio    slurm/train_libero_app.slurm
export ROBOCASA_LOW_DIM="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos" ROBOCASA_SUBDIR=v0.1/single_stage/kitchen_pnp/PnPSinkToCounter EXP_NAME=robocasa_with_proprio
sbatch --export=ALL                                                 slurm/train_robocasa_app.slurm; unset ROBOCASA_LOW_DIM EXP_NAME
sbatch --export=ALL,ROBOCASA_SUBDIR=v0.1/single_stage/kitchen_pnp/PnPSinkToCounter,EXP_NAME=robocasa_no_proprio slurm/train_robocasa_app.slurm
# interactive（salloc 后内联 KEY=VAL；逗号不冲突）
salloc --partition=a100 --gres=gpu:4 --constraint=a100_80 --cpus-per-task=8 --time=02:00:00
EXP_NAME=libero10_with_proprio bash slurm/train_libero_app.slurm
# resume（Hydra 覆盖）
... experiment.ckpt_path=<.pth> experiment.resume=true
# 评测（必带 DROID_PROJECT_DIR / WORK；并发错开 PORT）
sbatch --export=ALL,DROID_PROJECT_DIR=$REPO,WORK=$WORK,BENCHMARK=<libero|robocasa>,CKPT=<abs.pth>,NUM_TRIALS=1[,PORT=8770] slurm/eval_app.slurm
```
