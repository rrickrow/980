# MPC + LeWorldModel + Planning-Residual RL（论文实验级工程）

本工程实现三个可区分的研究组件：

1. **Ensemble World Model + CEM-MPC**：用于 MuJoCo 状态空间控制的可训练基线。
2. **Planning-level Residual SAC**：RL 不从零生成动作，而对 MPC 的短时 action chunk 做残差修正；可选世界模型不确定性自适应门控与 MPC 一致性正则。
3. **LeWorldModel 复现路径**：提供 LeWM 的两项核心训练目标（next-embedding prediction + SIGReg）及 latent CEM-MPC 适配代码；另附官方仓库对齐说明。

> 说明：ZPRL 原论文建立在预训练生成式机器人策略（flow-matching policy）上。本工程的 `zprl_style` 是用于 MuJoCo 公平对比的**瓶颈潜空间扰动代理基线**，不是对 ZPRL 机器人实验的等价复现。这样可避免把“世界模型潜空间”和“冻结基础策略瓶颈潜空间”混为一谈。

## 目录

```text
MPC_RL_ZPRL_PaperGrade/
├─ configs/                   # 环境/训练/消融配置
├─ mpcrl/
│  ├─ world_model.py          # Ensemble dynamics + reward model
│  ├─ cem.py                  # 真正的 CEM-MPC
│  ├─ sac.py                  # Residual SAC
│  ├─ gate.py                 # 不确定性门控
│  ├─ zprl_style.py           # 瓶颈潜空间扰动代理基线
│  ├─ lewm.py                 # LeWM core + SIGReg + reward probe
│  ├─ lewm_planner.py         # latent CEM planner
│  ├─ replay.py               # replay buffers
│  ├─ metrics.py              # 论文指标
│  └─ plotting.py             # 论文绘图
├─ experiments/
│  ├─ train.py                # MPC / action residual / planning residual
│  ├─ train_zprl_style.py     # ZPRL-style proxy
│  ├─ train_lewm.py           # LeWM objective-level reproduction
│  ├─ run_suite.py            # 多任务多 seed 批量实验
│  ├─ ablation.py             # 消融实验矩阵
│  └─ aggregate.py            # CSV 聚合
├─ scripts/
│  ├─ smoke_test.sh
│  ├─ run_main_suite.sh
│  └─ run_ablation.sh
├─ tests/
└─ docs/
```

## RTX 5090 环境配置

本分支提供 `configs/rtx5090/` 专用配置。RTX 5090 属于 NVIDIA Blackwell（SM 12.0），建议使用 **PyTorch CUDA 12.8+** 官方构建。专用配置默认启用 BF16 自动混合精度、TF32、cuDNN benchmark，并针对 32 GB 显存增大 world model、SAC batch 和 CEM 并行候选数。

### 1. 创建 Conda 环境

```bash
conda create -n mpc_rl_zprl_5090 python=3.11 -y
conda activate mpc_rl_zprl_5090
```

也可以使用工程中的 `environment.yml` 创建基础环境：

```bash
conda env create -f environment.yml
conda activate mpc_rl_zprl_5090
```

### 2. 安装 RTX 5090 可用的 PyTorch

优先安装 CUDA 12.8 官方 wheel：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

随后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

> 不建议给 RTX 5090 安装旧的 `cu121/cu124/cu126` PyTorch wheel。工程要求 `torch>=2.7`，并以 CUDA 12.8+ 构建为目标。

### 3. 检查 RTX 5090 是否正确工作

```bash
nvidia-smi
python scripts/check_rtx5090.py
```

正常情况下应看到类似：

```text
GPU: NVIDIA GeForce RTX 5090
Compute capability: 12.0
VRAM GiB: 31.x
BF16 supported: True
RTX 5090 environment check: PASS
```

### 4. 一键安装（Ubuntu / bash）

若本机已经安装 Conda，可直接执行：

```bash
bash scripts/install_rtx5090.sh
conda activate mpc_rl_zprl_5090
```

该脚本依次创建 Conda 环境、安装 CUDA 12.8 PyTorch、安装工程依赖并执行 GPU 自检。

## 最小训练

```bash
python -m experiments.train --config configs/halfcheetah.yaml --method mpc_only --steps 30000
python -m experiments.train --config configs/halfcheetah.yaml --method action_residual --steps 30000
python -m experiments.train --config configs/halfcheetah.yaml --method planning_residual --steps 30000
# 对保存的 run 做确定性评估：
python -m experiments.evaluate --run-dir runs/<run_name> --episodes 10
```

## RTX 5090 推荐运行顺序

先进行最小 GPU 验证：

```bash
conda activate mpc_rl_zprl_5090
python scripts/check_rtx5090.py
python -m unittest discover -s tests -v
python -m experiments.train \
  --config configs/rtx5090/halfcheetah.yaml \
  --method planning_residual \
  --steps 2000 \
  --run-name rtx5090_smoke
```

确认无报错后，运行单个正式实验：

```bash
python -m experiments.train \
  --config configs/rtx5090/halfcheetah.yaml \
  --method planning_residual \
  --steps 100000 \
  --seed 0
```

四种方法分别为：

```text
mpc_only
action_residual
zprl_style
planning_residual
```

RTX 5090 主配置的关键参数：

- `precision: bf16`
- World Model：7-member ensemble，hidden=512
- SAC：hidden=512，batch=1024
- Hopper/Walker2d：CEM candidates=2048
- HalfCheetah：CEM candidates=4096
- TF32 与 cuDNN benchmark 默认开启

这些配置用于提高单卡吞吐。若显存被其他进程占用，可优先将 `mpc.candidates` 降至 2048，再将 `train.batch_size` 降至 512。

## 论文主实验

RTX 5090 推荐：

```bash
python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml
# 如需同时运行计算量更大的 LeWorldModel-MPC：
python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml --include-lewm
python -m experiments.aggregate --root runs --out results/summary.csv
python -m experiments.plot_results --summary results/summary.csv --outdir results/figures
```

原始保守配置仍保留在 `configs/*.yaml`，用于硬件无关的对照和消融复现。

主对比：

- `mpc_only`
- `action_residual`
- `zprl_style`
- `planning_residual`（本文方法）
- `lewm_mpc`（单独通过 `train_lewm.py` 运行）

## 消融实验

```bash
python -m experiments.ablation --config configs/halfcheetah.yaml --steps 30000
```

消融默认覆盖：

- 无自适应门控
- 无不确定性惩罚
- 无一致性正则
- action chunk 长度 1 / 3 / 5
- 固定 residual scale 与自适应 residual scale

## 记录指标

训练过程写出：

- episode return
- success rate（环境提供 `success`/`is_success` 时直接读取；否则仅在配置了 return threshold 时计算）
- sample efficiency（首次达到配置阈值的环境交互步）
- MPC 单步规划时间
- 动作一阶平滑度 `||a_t-a_{t-1}||`
- 动作二阶平滑度 `||a_t-2a_{t-1}+a_{t-2}||`
- 世界模型一步预测 MSE
- ensemble uncertainty
- residual norm
- adaptive gate

## LeWorldModel 对齐边界

`mpcrl/lewm.py` 复现的是 LeWorldModel 的核心训练机制：

```text
z_t = encoder(o_t)
zhat_{t+1} = predictor(z_t, a_t)
L = MSE(zhat_{t+1}, z_{t+1}) + lambda * SIGReg(Z)
```

SIGReg 使用随机一维投影并比较经验特征函数与标准高斯特征函数。`train_lewm.py` 为使 MuJoCo locomotion 可规划，在 LeWM 训练完成并冻结后，**单独**拟合 reward probe；因此 reward probe 不进入 LeWM 两项核心目标。

论文官方实现以 pixels + action-conditioned latent predictor + CEM latent planning 为主。本工程面向可直接训练和方法消融，编码器采用轻量 CNN，便于单卡实验；若要做严格数值复现，应使用官方代码、官方数据与 checkpoint，详见 `docs/LEWM_REPRODUCTION.md`。

## 从 GitHub 下载与更新

仓库上传到 GitHub 后，首次下载使用：

```bash
git clone https://github.com/rrickrow/MPC_RL_ZPRL_RTX5090.git
cd MPC_RL_ZPRL_RTX5090
```

之后获取最新代码：

```bash
cd MPC_RL_ZPRL_RTX5090
git pull
```

如果服务器没有 GitHub SSH Key，直接使用上面的 HTTPS `git clone` 即可。公开仓库下载不需要登录；私有仓库则需要 GitHub 凭据或 SSH Key。

## 验证

```bash
python -m compileall mpcrl experiments tests
python -m unittest discover -s tests -v
```
#   M P C _ R L _ Z P R L _ R T X 5 0 9 0  
 