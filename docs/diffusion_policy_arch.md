# Diffusion Policy UNet 结构简图

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  DiffusionPolicyUNet                      │
                    └─────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                                                               ▼
┌─────────────────────────────┐                               ┌─────────────────────────────┐
│   obs_encoder (DataParallel) │                               │ noise_pred_net (DataParallel)│
│   ObservationGroupEncoder    │                               │   ConditionalUnet1D         │
└─────────────────────────────┘                               └─────────────────────────────┘
              │                                                               ▲
              │  obs 组输入                                                    │
              ▼                                                               │ global_cond
┌─────────────────────────────┐                                               │ (obs_cond:
│ ObservationEncoder          │                                               │  512×To=1280)
│  ├ robot_state/cartesian    │ ──(6d)──┐                                     │
│  ├ robot_state/gripper      │ ──(1d)───┤                                     │
│  ├ cam1 (varied_camera_1)   │          ├── concat ──► combine ──► 512 维 ───┘
│  │   ColorRand→CropRand     │          │   (1031→1024→512→512)
│  │   → VisualCore(ResNet50) │ ──512d──┘
│  └ cam2 (varied_camera_2)   │
│      sharing_from cam1      │ ──512d──┘
└─────────────────────────────┘

noise_pred_net 内部:
   noisy_actions(10d) + timestep ──► diffusion_step_encoder(256d)
   obs_cond(1280d) ──► cond_encoder(2048d)
   down: 10→256→512→1024  (3层) + cond
   mid:  1024 (2块) + cond
   up:   1024→512→256  (2层) + cond
   final_conv: 256 → 10d (action)
```

## 要点

| 模块 | 说明 |
|------|------|
| **obs_encoder** | 低维 7 维 + 两路 RGB 各 512 维 (ResNet50)，concat 后 MLP 得到 512 维 |
| **noise_pred_net** | 1D U-Net，条件为 obs_cond (512×observation_horizon)，输出 10 维 action |
| **VisualCore** | 输入 3×116×116 (crop 后)，ResNet50Conv，输出 512 维，cam2 与 cam1 共享权重 |
