"""
Configuration parameters for the 2D DOA Estimation project.

Extended from: A Two-Stage Multi-Layer Perceptron for High-Resolution DOA Estimation
(Zhang et al., IEEE TVT 2024, https://github.com/Whisperzyj/TS-MLP)

Modifications:
  - Signal model: s(t) changed from zero-mean complex Gaussian to a single-frequency pulse
  - Array: 3-element L-shaped array for 2D DOA (azimuth + elevation)
  - Number of targets: extended to K = 2, 3, or 4
"""

import numpy as np

# =====================================================================
# Physical Constants
# =====================================================================
C_LIGHT = 3e8  # speed of light [m/s]

# =====================================================================
# Pulse Signal Parameters
# =====================================================================
PRF = 1000           # Pulse Repetition Frequency [Hz]
F_CARRIER = 5e9      # Carrier frequency [Hz]  (5 GHz)
PULSE_WIDTH = 50e-9  # Pulse width [s]          (50 ns)
F_SAMPLE = 250e6     # Sampling frequency [Hz]  (250 MHz)
BEAM_POINTING = (0.0, 0.0)  # Beam pointing (azimuth, elevation) [degrees]

# Derived pulse parameters
WAVELENGTH = C_LIGHT / F_CARRIER          # Wavelength [m] (~0.06 m @ 5 GHz)
T_PULSE_REPEAT = 1.0 / PRF               # Pulse repetition interval [s]
N_SAMPLES_PER_PULSE = int(PULSE_WIDTH * F_SAMPLE)  # Samples within one pulse (~12)

# =====================================================================
# Receiving Array Configuration
# 3-element L-shaped planar array for 2D (azimuth + elevation) DOA.
#
#   Element layout (looking from above, x=azimuth axis, y=elevation axis):
#       y
#       |
#   3---+
#       |
#       1---2---x
#
#   Element 1: (0, 0)  - reference
#   Element 2: (d, 0)  - azimuth-sensitive
#   Element 3: (0, d)  - elevation-sensitive
#
# One "measurement" = 3 channels × 2 snapshots × 2 (I/Q) = 12 real values.
# =====================================================================
N_ELEMENTS = 3                                   # Number of receive elements
D_ELEMENT = WAVELENGTH / 2                       # Half-wavelength element spacing [m]
N_SNAPSHOTS = 2                                  # Snapshots per measurement (L)

# Element positions in the horizontal plane [m], shape [N_ELEMENTS × 2]
ARRAY_POS = np.array([[0, 0],
                      [1, 0],
                      [0, 1]], dtype=float) * D_ELEMENT

# Input dimension to the network = N² (vectorized lower-triangular covariance)
INPUT_DIM = N_ELEMENTS ** 2  # = 9 real values

# =====================================================================
# DOA Angle Search Ranges
# =====================================================================
AZ_MIN = -30.0   # Azimuth minimum  [degrees]
AZ_MAX = 30.0    # Azimuth maximum  [degrees]
EL_MIN = -30.0   # Elevation minimum [degrees]
EL_MAX = 30.0    # Elevation maximum [degrees]

# Stage-1 coarse grid (integer degrees, 1° resolution)
AZ_GRID_INT = np.arange(AZ_MIN, AZ_MAX + 0.5, 1.0)   # -30, -29, ..., 30 → 61 points
EL_GRID_INT = np.arange(EL_MIN, EL_MAX + 0.5, 1.0)   # 61 points
N_AZ_INT = len(AZ_GRID_INT)   # 61
N_EL_INT = len(EL_GRID_INT)   # 61

# Stage-2 fine grid (decimal degree, 0.01° resolution)
# Covers ±0.50° around each integer grid point → 100 grid points
DECIMAL_STEP = 0.01                                       # [degrees]
DECIMAL_GRID = np.arange(-0.50, 0.50, DECIMAL_STEP)      # 100 points
N_DECIMAL = len(DECIMAL_GRID)   # 100

# =====================================================================
# Target Scenario
# =====================================================================
TARGET_RANGE = 3000.0  # Slant range to all targets [m]

# =====================================================================
# Dataset Generation Parameters
# =====================================================================
SNR_LIST = [0, 5, 10, 15, 20, 25, 30]   # SNR values [dB]
N_SNR = len(SNR_LIST)                    # 7 SNR levels

# Amplitude ratio  a = A_k / A_1  for secondary targets (k ≥ 2)
# A_1 = 1.0 (reference), a ∈ {0.01, 0.02, …, 1.00}
A_MIN = 0.01
A_MAX = 1.00
A_STEP = 0.01
A_VALUES = np.round(np.arange(A_MIN, A_MAX + A_STEP / 2, A_STEP), 4)  # 100 values
N_A_VALUES = len(A_VALUES)   # 100

# --------------- 2-target dataset parameters (follow paper strategy) ---------------
# Angle separation Δθ between the two targets ∈ [1°, 10°], step 0.05°
DELTA_ANGLE_MIN_2T = 1.0
DELTA_ANGLE_MAX_2T = 10.0
DELTA_ANGLE_STEP_2T = 0.05
DELTA_ANGLE_VALUES_2T = np.round(
    np.arange(DELTA_ANGLE_MIN_2T,
              DELTA_ANGLE_MAX_2T + DELTA_ANGLE_STEP_2T / 2,
              DELTA_ANGLE_STEP_2T), 4)   # 181 values
N_DELTA_2T = len(DELTA_ANGLE_VALUES_2T)  # 181
SAMPLES_PER_COMB_2T = 5000               # random angle pairs per (a, Δθ)

# --------------- 3-target dataset parameters ---------------
# Allocation strategy:
#   - Single amplitude ratio a (same for targets 2 & 3): 100 values (same as 2T)
#   - Minimum pairwise angular separation Δθ_min ∈ [1°, 10°], step 0.05° → 181 values
#   - 5000 random angle triplets per (a, Δθ_min) satisfying all pairwise sep ≥ Δθ_min
#   - Total samples = 100 × 181 × 5000 × 7 (same order as 2T dataset)
DELTA_ANGLE_MIN_3T = 1.0
DELTA_ANGLE_MAX_3T = 10.0
DELTA_ANGLE_STEP_3T = 0.05
DELTA_ANGLE_VALUES_3T = np.round(
    np.arange(DELTA_ANGLE_MIN_3T,
              DELTA_ANGLE_MAX_3T + DELTA_ANGLE_STEP_3T / 2,
              DELTA_ANGLE_STEP_3T), 4)   # 181 values
N_DELTA_3T = len(DELTA_ANGLE_VALUES_3T)  # 181
SAMPLES_PER_COMB_3T = 5000

# --------------- 4-target dataset parameters ---------------
# Allocation strategy:
#   - Single amplitude ratio a (same for targets 2, 3 & 4): 100 values
#   - Minimum pairwise angular separation Δθ_min ∈ [0.5°, 5°], step 0.05° → 91 values
#     (reduced max separation: fitting 4 targets in [-30°, 30°] with large separations
#      is geometrically harder, so the range is halved)
#   - 5000 random angle quadruplets per (a, Δθ_min)
#   - Total samples = 100 × 91 × 5000 × 7 ≈ 318.5 M
DELTA_ANGLE_MIN_4T = 0.5
DELTA_ANGLE_MAX_4T = 5.0
DELTA_ANGLE_STEP_4T = 0.05
DELTA_ANGLE_VALUES_4T = np.round(
    np.arange(DELTA_ANGLE_MIN_4T,
              DELTA_ANGLE_MAX_4T + DELTA_ANGLE_STEP_4T / 2,
              DELTA_ANGLE_STEP_4T), 4)   # 91 values
N_DELTA_4T = len(DELTA_ANGLE_VALUES_4T)  # 91
SAMPLES_PER_COMB_4T = 5000

# =====================================================================
# File Paths
# =====================================================================
DATASET_DIR = "datasets"
DATASET_2T_PATH = f"{DATASET_DIR}/dataset_2targets.h5"
DATASET_3T_PATH = f"{DATASET_DIR}/dataset_3targets.h5"
DATASET_4T_PATH = f"{DATASET_DIR}/dataset_4targets.h5"

# Model checkpoint directory
MODEL_DIR = "models_saved"

# =====================================================================
# Network Hyper-parameters
# =====================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 1000
N_EPOCHS = 300
LR_PATIENCE = 10       # epochs without improvement before LR decay
LR_DECAY_FACTOR = 0.7  # LR multiplied by this when validation loss plateaus
DROPOUT_RATE = 0.5

# Loss weights for the three network outputs (integer, decimal, pairing)
LOSS_WEIGHT_INT = 1.0
LOSS_WEIGHT_DEC = 0.2
LOSS_WEIGHT_PAIR = 10.0

# Peak label value (following the paper: labels use 100 rather than 1)
LABEL_PEAK_VALUE = 100.0
