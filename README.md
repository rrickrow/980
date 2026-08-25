# 2D DOA 估计实验 —— TS-MLP 扩展版

> 基于论文：*A Two-Stage Multi-Layer Perceptron for High-Resolution DOA Estimation*（Zhang et al., IEEE TVT 2024）  
> 本项目将原论文的一维方位角估计扩展至**二维（方位角 + 俯仰角）**，并引入单频脉冲信号模型与三元 L 型阵列。

---

## 目录

1. [项目概述](#项目概述)
2. [环境要求](#环境要求)
3. [快速开始（一键运行）](#快速开始一键运行)
4. [分步运行](#分步运行)
   - [Step 1 — 生成数据集](#step-1--生成数据集)
   - [Step 2 — 训练模型](#step-2--训练模型)
   - [Step 3 — 评估模型](#step-3--评估模型)
5. [文件说明](#文件说明)
6. [信号模型与阵列配置](#信号模型与阵列配置)
7. [网络结构](#网络结构)
8. [数据集 HDF5 结构](#数据集-hdf5-结构)
9. [常见问题](#常见问题)

---

## 项目概述

本项目实现了一个**三阶段 MLP（TS-MLP）**网络，用于同时估计 K = 2、3 或 4 个目标的**二维来波方向（DOA）**：

| 阶段 | 输出 | 说明 |
|------|------|------|
| Stage 1 | `z1_az`（61 维）, `z1_el`（61 维）| 方位角 / 俯仰角粗网格（整数度）|
| Stage 2 | `z2_az`（K × 100 维）, `z2_el`（K × 100 维）| 每目标小数精细修正（± 0.50° @ 0.01° 步长）|
| Stage 3 | `z3`（K! 维）| 方位–俯仰跨维度配对 |

**输入特征**：3 通道 × 2 快拍 × 2（I/Q）→ 向量化 3×3 协方差矩阵 = **9 维实数向量**。

---

## 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.10 |
| NumPy | ≥ 1.23 |
| h5py | ≥ 3.7 |
| CUDA（可选）| 11.x / 12.x（GPU 加速训练）|

### 安装依赖

```bash
pip install tensorflow numpy h5py
```

> **GPU 用户**：将上方 `tensorflow` 替换为与 CUDA 版本匹配的 `tensorflow[and-cuda]` 或参照  
> [TensorFlow 官方安装文档](https://www.tensorflow.org/install/pip)。

---

## 快速开始（一键运行）

### Windows

在项目根目录中双击 **`run_experiment.bat`**，或在命令提示符中执行：

```bat
run_experiment.bat
```

该脚本将依次完成：

1. 安装 Python 依赖
2. 生成 K = 2、3、4 的完整数据集（数据量：约 2 × 633M + 318M 样本，**需数小时至数天，视硬件而定**）
3. 训练三个模型
4. 对每个模型进行真实协方差与采样协方差两种模式的评估

> **快速测试**（小数据量验证流程）：编辑 `run_experiment.bat`，在数据集生成步骤中将 `MAX_SAMPLES` 改为 `100000`；如需减少训练轮数，请修改 `config.py` 中的 `N_EPOCHS`（默认 300）。

### Linux / macOS

```bash
chmod +x run_experiment.sh   # 如果提供了 shell 版本
./run_experiment.sh
# 或者逐步执行（见下文）
```

---

## 分步运行

### Step 1 — 生成数据集

```bash
# 生成全部三种目标数量的数据集（约数 TB，耗时较长）
python generate_dataset.py --targets 2 3 4

# 只生成 K=2 的数据集
python generate_dataset.py --targets 2

# 快速测试（每个数据集仅生成 10 万样本）
python generate_dataset.py --targets 2 3 4 --max_samples 100000

# 自定义随机种子
python generate_dataset.py --targets 2 --seed 123
```

生成的文件保存至：

```
datasets/
  dataset_2targets.h5
  dataset_3targets.h5
  dataset_4targets.h5
```

---

### Step 2 — 训练模型

训练前请确认对应 HDF5 数据集文件已存在。

```bash
# 训练 K=2 模型
python train.py --K 2 --dataset datasets/dataset_2targets.h5

# 训练 K=3 模型
python train.py --K 3 --dataset datasets/dataset_3targets.h5

# 训练 K=4 模型
python train.py --K 4 --dataset datasets/dataset_4targets.h5

# 指定 GPU（例如使用第 1 块 GPU）
python train.py --K 2 --dataset datasets/dataset_2targets.h5 --gpu 1

# 仅使用 CPU
python train.py --K 2 --dataset datasets/dataset_2targets.h5 --gpu -1
```

训练输出保存至：

```
models_saved/
  ts_mlp_K2_best.h5          ← 最优模型检查点
  training_history_K2.csv    ← 每轮损失 / 精度记录
  history_K2.npy             ← NumPy 格式训练历史
```

**训练超参数**（均可在 `config.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 1000 | 小批量大小 |
| `N_EPOCHS` | 300 | 最大训练轮数 |
| `LEARNING_RATE` | 1e-4 | Adam 初始学习率 |
| `LR_PATIENCE` | 10 | 验证损失停滞多少轮后降低 LR |
| `LR_DECAY_FACTOR` | 0.7 | LR 衰减系数 |
| `DROPOUT_RATE` | 0.5 | Dropout 比例 |

---

### Step 3 — 评估模型

```bash
# 真实协方差模式（与训练条件相同）
python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5

# 采样协方差模式（模拟真实推理场景）
python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5 --use_sample_cov

# 指定 SNR 列表
python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5 --snr_list 0 10 20 30

# 调整每个 SNR 下的测试样本数（默认 1000）
python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5 --n_test 500
```

**输出格式示例**：

```
Evaluating TS-MLP for K = 2 targets
Model : models_saved/ts_mlp_K2_best.h5
Mode  : true covariance

  SNR (dB)   RMSE_az (°)   RMSE_el (°)   Pairing acc (%)
----------------------------------------------------------
         0        1.2345        1.3456               91.2
         5        0.5678        0.6789               97.8
        10        0.1234        0.1345               99.5
       ...
```

---

## 文件说明

```
ts_mlp_2d_doa/              # 项目根目录（即本仓库）
├── config.py              # 全局配置（信号参数、阵列、超参数、文件路径）
├── signal_model.py        # 阵列导向矢量、真实协方差与采样协方差
├── label_generator.py     # 二维 DOA 标签编码 / 解码（三阶段）
├── dataset_generator.py   # K=2/3/4 数据集生成器
├── generate_dataset.py    # 入口：生成并保存 HDF5 数据集
├── ts_mlp.py              # TS-MLP 网络定义（Keras/TensorFlow）
├── train.py               # 训练脚本
├── evaluate.py            # 评估 / 推理脚本
├── run_experiment.bat     # 一键运行完整实验（Windows）
└── README.md              # 本文档
```

---

## 信号模型与阵列配置

### 脉冲信号参数

| 参数 | 值 |
|------|----|
| 载波频率 `f_c` | 5 GHz |
| 脉冲重复频率 `PRF` | 1 kHz |
| 脉冲宽度 `τ` | 50 ns |
| 采样频率 `F_s` | 250 MHz |
| 每脉冲采样数 | 12 |

### L 型阵列

```
y
|
3---+
    |
    1---2---x
```

- 元素 1：(0, 0)——参考元素  
- 元素 2：(d, 0)——方位角敏感  
- 元素 3：(0, d)——俯仰角敏感  
- 元素间距 `d = λ/2`（半波长）

### 导向矢量

$$
a_n(\varphi, \theta) = \exp\!\left(j\frac{2\pi}{\lambda}(x_n \sin\varphi\cos\theta + y_n \sin\theta)\right)
$$

### 协方差矩阵

$$
\mathbf{R} = \mathbf{A}\,\mathbf{R}_s\,\mathbf{A}^H + \sigma^2\mathbf{I}
$$

---

## 网络结构

```
输入 (9 维)
    │
    ├─[全连接块 × 4, 256 单元, ReLU, Dropout(0.5)]
    │
    ├─→ z1_az  Softmax(61)        ← 方位角粗网格
    ├─→ z1_el  Softmax(61)        ← 俯仰角粗网格
    ├─→ z2_az  Softmax(K, 100)    ← 每目标方位角细调
    ├─→ z2_el  Softmax(K, 100)    ← 每目标俯仰角细调
    └─→ z3     Softmax(K!)        ← 方位–俯仰配对
```

**损失函数**：KL 散度 × 权重（整数级 1.0，小数级 0.2，配对级 10.0）

---

## 数据集 HDF5 结构

每个 `.h5` 文件包含以下数据集：

| 键名 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `X` | `(N, 9)` | float32 | 协方差矩阵特征向量 |
| `z1_az` | `(N, 61)` | float32 | Stage-1 方位角标签 |
| `z1_el` | `(N, 61)` | float32 | Stage-1 俯仰角标签 |
| `z2_az` | `(N, K, 100)` | float32 | Stage-2 每目标方位角小数标签 |
| `z2_el` | `(N, K, 100)` | float32 | Stage-2 每目标俯仰角小数标签 |
| `z3` | `(N, K!)` | float32 | Stage-3 配对标签（one-hot）|
| `az_sorted` | `(N, K)` | float32 | 真实方位角（升序排列）|
| `el_sorted` | `(N, K)` | float32 | 真实俯仰角（与方位角对应）|

---

## 常见问题

**Q: 磁盘空间不足怎么办？**  
A: 使用 `--max_samples` 参数限制数据集大小，例如 `--max_samples 500000`。完整数据集约需 500 GB–2 TB 空间。

**Q: 内存不足（OOM）？**  
A: 减小 `config.py` 中的 `BATCH_SIZE`（默认 1000），或降低 `N_SNAPSHOTS`（默认 2）以减小协方差矩阵计算量。

**Q: 如何只复现 K=2 的结果？**  
A: 依次执行：
```bash
python generate_dataset.py --targets 2
python train.py --K 2 --dataset datasets/dataset_2targets.h5
python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5
```

**Q: 训练时 `val_loss` 不下降？**  
A: 检查数据集是否正确生成（样本数 > 0），并尝试调小 `LEARNING_RATE`（如 `5e-5`）。

**Q: Windows 下如何使用 GPU？**  
A: 确认已安装 CUDA 工具包与与 TensorFlow 版本匹配的 cuDNN，详见  
[TensorFlow GPU 支持文档](https://www.tensorflow.org/install/pip#hardware_requirements)。
