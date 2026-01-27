# Robomimic Algorithms Overview

本目录包含多种 **模仿学习 (Imitation Learning)** 与 **离线强化学习 (Offline RL)** 算法实现，用于从专家示教或离线数据中学习机器人策略。下面按算法分别介绍 **方法思想**、**网络结构** 与 **参考文献**。

---

## 1. BC — Behavioral Cloning

**Paper:** [Behavioral Cloning: Supervised Imitation Learning](https://www.emergentmind.com/topics/behavioral-cloning)  
**思想：** 将模仿学习视为监督学习：给定专家状态-动作对，学习策略 π(a|s)，最小化预测动作与专家动作的差异（如 L2 / L1 / 余弦损失）。简单直接，但存在 **covariate shift** 与 **误差累积**，对分布外状态敏感。

**网络结构：**
- **Policy:** `ActorNetwork` — 观测编码器（支持 low-dim / rgb / depth 等）+ MLP 输出动作。
- **变体：**
  - **BC_Gaussian:** `GaussianActorNetwork`，输出动作分布，负对数似然训练。
  - **BC_GMM:** `GMMActorNetwork`，高斯混合模型，多峰动作分布。
  - **BC_VAE:** `VAEActor`，VAE 先采样 latent 再解码为动作，重建损失 + KL。
  - **BC_RNN / BC_RNN_GMM:** `RNNActorNetwork` / `RNNGMMActorNetwork`，时序编码，支持 open-loop 等。
  - **BC_Transformer / BC_Transformer_GMM:** `TransformerActorNetwork` / `TransformerGMMActorNetwork`，基于 Transformer 的序列策略。  
  - 若使用 Transformer 变体，须在 `algo/__init__.py` 中 import 对应类（当前未导出）。

**配置:** `actor_layer_dims`，`loss.l2_weight` / `l1_weight` / `cos_weight`，以及各变体对应 `gaussian` / `gmm` / `vae` / `rnn` / `transformer` 配置。

---

## 2. BCQ — Batch-Constrained Q-Learning

**Paper:** [Off-Policy Deep RL without Exploration](https://arxiv.org/abs/1812.02900) (ICML 2019), [Fujimoto et al.](https://github.com/sfujim/BCQ)  
**思想：** 针对 **离线 RL** 的 **extrapolation error**：若策略产生数据分布外的动作，Q 容易过估。BCQ 通过 **batch constraint** 限制策略只选数据支撑的动作。实现上：用 **生成模型**（VAE/GMM）拟合数据中的动作分布，再叠加 **扰动网络** 在保持“接近数据”的前提下微调动作，最后用 **Q 网络** 选动作。

**网络结构：**
- **Critic:** `ActionValueNetwork` 的 **ensemble**（多 Q 网 + target），Q(s,a)。
- **Action Sampler:**  
  - **BCQ (VAE):** `VAEActor` — 条件 VAE，从策略 π(a|s) 采样，再经 perturbation 得到最终动作。  
  - **BCQ_GMM:** GMM 作为 action sampler。  
  - **BCQ_Distributional:** 分布型 Q。
- **Actor (可选):** `PerturbationActorNetwork`，在 VAE/GMM 采样基础上做有界扰动。

**配置:** `action_sampler.vae` / `gmm`，`critic.ensemble.n`，`actor.perturbation_scale` 等。

---

## 3. CQL — Conservative Q-Learning

**Paper:** [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779) (NeurIPS 2020)  
**思想：** 学习 **保守的 Q 函数**，使策略在该 Q 下的期望回报是真实值的一个 **下界**，从而缓解 **value overestimation**。在标准 TD 目标上增加 **CQL 正则**：压低 **OOD 动作** 的 Q、提高 **in-dataset 动作** 的 Q。本实现基于 **SAC 风格** actor-critic，并加 CQL 项。

**网络结构：**
- **Actor:** `GaussianActorNetwork`，随机策略。
- **Critic:** `ActionValueNetwork` **ensemble** + target。
- **可选:** `log_entropy_weight`（自动熵系数）、`log_cql_weight`（自动 CQL 权重）。

**配置:** `critic.min_q_weight` / `cql_weight` / `target_q_gap`，`actor.target_entropy`，`n_step` 等。

---

## 4. IQL — Implicit Q-Learning

**Paper:** [Offline RL with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) (ICLR 2022)  
**思想：** **不显式对 OOD 动作做 Q 查询**，避免分布偏移。通过 **expectile regression** 拟合 state value V(s)，再 implicit 地 backup 成 Q，最后用 **advantage-weighted BC** 提取策略，始终只使用数据里的动作。

**网络结构：**
- **Actor:** `GaussianActorNetwork` 或 `GMMActorNetwork`。
- **Critic:** `ActionValueNetwork` ensemble + target，用于 TD 学 Q。
- **Value:** `ValueNetwork`，仅 V(s)，用 expectile loss 训练。

**配置:** `critic.ensemble.n`，expectile τ，advantage 温度等。

---

## 5. TD3-BC — TD3 + Behavior Cloning

**Paper:** [A Minimalist Approach to Offline RL](https://arxiv.org/abs/2106.06860) (NeurIPS 2021), [Fujimoto & Gu](https://github.com/sfujim/TD3_BC)  
**思想：** 在 **TD3** 上做**极小改动**：  
(1) 在 **policy 更新** 中加 **BC 项**，拉近策略输出与数据集动作；  
(2) **数据归一化**。  
无需生成模型、实现简单、调参少，常作为 **offline RL 基线**。

**网络结构：**
- **Actor:** `ActorNetwork`（确定性 MLP），+ `actor_target`。
- **Critic:** 与 BCQ 相同的 `ActionValueNetwork` **双 Q + target**。

**配置:** `actor.layer_dims`，`critic.layer_dims`，BC 权重 α 等。

---

## 6. Diffusion Policy (DiffusionPolicyUNet)

**Paper:** [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) (RSS 2023, Columbia)  
**思想：** 把 **策略** 建模为 **条件扩散模型**：给定观测，对动作 **加噪→去噪**，通过 **DDPM / DDIM** 迭代去噪得到动作序列。适合 **多峰、高维动作**，训练稳定，支持 **chunk 式 receding horizon** 控制。

**网络结构：**
- **Obs Encoder:** `ObservationGroupEncoder`（与 BC 等共用 encoder 配置，如 VisualCore + ResNet），输出 `obs_dim`；BatchNorm 换为 GroupNorm 以兼容 EMA。
- **Noise Predictor:** `ConditionalUnet1D` — 1D 时序 Unet：  
  - 输入：噪声动作 `(B, T, action_dim)`，扩散步 `t`，全局条件 `global_cond = [diffusion_step_embed; flatten(obs)]`。  
  - `SinusoidalPosEmb` + MLP 编码 `t`；观测拼入条件，经 FiLM 调制 `ConditionalResidualBlock1D`。  
  - `down_dims`（如 [256,512,1024]）控制通道；下采样→bottleneck→上采样 + skip，`kernel_size`、`n_groups`（GroupNorm）。
- **Scheduler:** `DDPMScheduler` 或 `DDIMScheduler`；可选 **EMA**。

**配置:** `horizon.observation_horizon` / `action_horizon` / `prediction_horizon`，`unet.down_dims`，`ddpm` / `ddim`，`ema.power` 等。

---

## 7. GL — Goal / Subgoal Learning (Planner)

**用途：** 为 **HBC / IRIS** 提供 **子目标 (subgoal)** 预测，不直接用于 rollout。  
**思想：** 给定当前 obs（及可选 goal），预测未来某步的 **subgoal 观测**（如 s_{t+k}），用于分层策略的 high-level 规划。

**网络结构：**
- **GL:** `MIMO_MLP` — 多输入多输出 MLP，`obs` + `goal` → `subgoal_shapes`，L2 损失。
- **GL_VAE:** `VAE` — 以 obs/goal 为条件，对 subgoal 做 VAE；可输出 **latent subgoal** 供 HBC actor 使用。

**配置:** `subgoal_horizon`，`ae.planner_layer_dims`，`vae.*`（若用 GL_VAE）。

---

## 8. HBC — Hierarchical Behavioral Cloning

**Paper:** [GTI: Generalization Through Imitation](https://arxiv.org/abs/2003.06085)  
**思想：** **分层模仿**：  
(1) **High-level planner (GL/GL_VAE):** 由当前状态（+ goal）预测 **subgoal**；  
(2) **Low-level policy (BC 类):** **goal-conditioned**，以 subgoal 为条件输出动作。  
每 `subgoal_update_interval` 步更新一次 subgoal，其间低层策略固定指向当前 subgoal，利于长时程、多任务泛化。

**网络结构：**
- **Planner:** `GL` 或 `GL_VAE`（见上）。
- **Actor:** BC / BC_RNN 等，**goal 输入** 改为 planner 的 subgoal（或 GL_VAE 的 latent subgoal）。

**配置:** `planner`（GL 配置），`actor`（BC 配置），`subgoal_update_interval`，`latent_subgoal` 等。

---

## 9. IRIS — Implicit Reinforcement without Interaction at Scale

**Paper:** [IRIS: Implicit Reinforcement without Interaction at Scale](https://arxiv.org/abs/1911.05321) (ICRA 2020, Stanford 等)  
**思想：** 从 **大规模离线机器人数据** 学控制，**无需在线交互**。  
**分层结构：**  
(1) **High-level:** **ValuePlanner** = **GL_VAE**（采样多个候选 subgoal）+ **BCQ-style value**（Q 网络）**挑选** 最优 subgoal；  
(2) **Low-level:** **goal-conditioned BC** 执行子目标。  
即用 “subgoal 生成 + value 选择” 替代纯模仿的 subgoal，从而 **结合多段 suboptimal 轨迹** 完成更优任务。

**网络结构：**
- **Planner:** `ValuePlanner` = **GL_VAE** + **BCQ value**（`ActionValueNetwork` ensemble，用于给候选 subgoal 打分）。
- **Actor:** BC 类，以 **选中 subgoal** 为条件。

**配置:** `value_planner.planner`（GL_VAE），`value_planner.value`（BCQ 配置），`actor`（BC）。

---

## 10. 算法选择速查

| 算法 | 类型 | 需 reward | 特点 |
|------|------|-----------|------|
| **BC** | 模仿 | 否 | 简单、多种 policy 形式 |
| **BCQ** | 离线 RL | 是 | Batch constraint，VAE/GMM + perturbation |
| **CQL** | 离线 RL | 是 | 保守 Q，SAC-style |
| **IQL** | 离线 RL | 是 | 无 OOD 动作查询，expectile + AWBC |
| **TD3-BC** | 离线 RL | 是 | 极简，TD3+BC 正则 |
| **Diffusion Policy** | 模仿 | 否 | 扩散策略，多峰、高维、chunk |
| **GL** | 规划 | 否 | 仅 subgoal 预测，用于 HBC/IRIS |
| **HBC** | 分层模仿 | 否 | Planner + goal-conditioned BC |
| **IRIS** | 分层离线 | 是 | ValuePlanner + BC，大数据离线 |

---

## 参考文献与链接

- BC: [Emergent Mind – Behavioral Cloning](https://www.emergentmind.com/topics/behavioral-cloning)
- BCQ: [Off-Policy Deep RL without Exploration](https://arxiv.org/abs/1812.02900), [BCQ GitHub](https://github.com/sfujim/BCQ)
- CQL: [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779)
- IQL: [Offline RL with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- TD3-BC: [A Minimalist Approach to Offline RL](https://arxiv.org/abs/2106.06860), [TD3_BC GitHub](https://github.com/sfujim/TD3_BC)
- Diffusion Policy: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- HBC / GTI: [Generalization Through Imitation](https://arxiv.org/abs/2003.06085)
- IRIS: [IRIS](https://sites.google.com/stanford.edu/iris/), [IRIS Paper](https://arxiv.org/abs/1911.05321)
