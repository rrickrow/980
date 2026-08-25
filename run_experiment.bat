@echo off
:: ============================================================
::  一键运行完整 2D DOA 估计实验
::  Run complete 2D DOA estimation experiment (K = 2, 3, 4)
::
::  使用方法 / Usage:
::    双击此文件  —— 或 ——  在命令提示符中执行:
::    Double-click this file  —— or ——  run in CMD:
::        run_experiment.bat
::
::  可选：快速测试模式（数据量小，速度快）
::  Optional: quick-test mode (small dataset, fast)
::    将下方 MAX_SAMPLES 改为 100000
::    Set MAX_SAMPLES=100000 below
:: ============================================================

setlocal enabledelayedexpansion

:: ── 配置区（根据需要修改）Configuration section ──────────────
:: 要运行的目标数量（空格分隔），可改为 "2" 只跑 K=2
set TARGETS=2 3 4

:: 每个数据集的最大样本数（0 = 生成全量数据，约数百 GB 至数 TB）
:: Maximum samples per dataset (0 = full dataset, may take days)
set MAX_SAMPLES=0

:: GPU 编号（-1 = 仅 CPU）  GPU index (-1 = CPU only)
set GPU=0

:: Python 可执行文件路径（通常无需修改）
set PYTHON=python
:: ─────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo  2D DOA 估计实验 ^| 2D DOA Estimation Experiment
echo ============================================================
echo  目标数量 Targets : %TARGETS%
echo  最大样本 MaxSamples : %MAX_SAMPLES% ^(0=full^)
echo  GPU : %GPU%
echo ============================================================
echo.

:: ── Step 0: 安装 Python 依赖 Install Python dependencies ──────
echo [Step 0/3] 安装依赖 / Installing Python dependencies ...
%PYTHON% -m pip install --quiet --upgrade tensorflow numpy h5py
if errorlevel 1 (
    echo [ERROR] 依赖安装失败 / Dependency installation failed.
    echo         请手动运行: pip install tensorflow numpy h5py
    pause
    exit /b 1
)
echo [Step 0/3] 依赖安装完成 / Dependencies installed.
echo.

:: ── Step 1: 生成数据集 Generate datasets ─────────────────────
echo [Step 1/3] 生成数据集 / Generating datasets ...
echo            目标: K = %TARGETS%

if "%MAX_SAMPLES%"=="0" (
    %PYTHON% generate_dataset.py --targets %TARGETS%
) else (
    %PYTHON% generate_dataset.py --targets %TARGETS% --max_samples %MAX_SAMPLES%
)

if errorlevel 1 (
    echo [ERROR] 数据集生成失败 / Dataset generation failed.
    pause
    exit /b 1
)
echo [Step 1/3] 数据集生成完成 / Datasets generated.
echo.

:: ── Step 2: 训练模型 Train models ────────────────────────────
echo [Step 2/3] 训练模型 / Training models ...

for %%K in (%TARGETS%) do (
    set K_NUM=%%K
    echo.
    echo   ── 训练 K=!K_NUM! 模型 / Training K=!K_NUM! model ──
    %PYTHON% train.py --K !K_NUM! --dataset datasets/dataset_!K_NUM!targets.h5 --gpu %GPU%
    if errorlevel 1 (
        echo [ERROR] K=!K_NUM! 模型训练失败 / Training K=!K_NUM! failed.
        pause
        exit /b 1
    )
    echo   K=!K_NUM! 训练完成 / K=!K_NUM! training done.
)

echo.
echo [Step 2/3] 所有模型训练完成 / All models trained.
echo.

:: ── Step 3: 评估模型 Evaluate models ─────────────────────────
echo [Step 3/3] 评估模型 / Evaluating models ...

for %%K in (%TARGETS%) do (
    set K_NUM=%%K
    echo.
    echo   ── 评估 K=!K_NUM!（真实协方差 / True covariance）──
    %PYTHON% evaluate.py --K !K_NUM! --model models_saved/ts_mlp_K!K_NUM!_best.h5
    if errorlevel 1 (
        echo [WARN] K=!K_NUM! 真实协方差评估出错，继续 / True-cov eval error, continuing.
    )

    echo.
    echo   ── 评估 K=!K_NUM!（采样协方差 / Sample covariance）──
    %PYTHON% evaluate.py --K !K_NUM! --model models_saved/ts_mlp_K!K_NUM!_best.h5 --use_sample_cov
    if errorlevel 1 (
        echo [WARN] K=!K_NUM! 采样协方差评估出错，继续 / Sample-cov eval error, continuing.
    )
)

echo.
echo ============================================================
echo  实验完成！/ Experiment complete!
echo  模型保存于 / Models saved in: models_saved\
echo ============================================================
echo.
pause
