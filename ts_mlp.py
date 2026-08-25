"""
TS-MLP network architecture for 2D DOA estimation (azimuth + elevation).

Extended from:
    Zhang et al., "A Two-Stage Multi-Layer Perceptron for High-Resolution
    DOA Estimation," IEEE TVT, 2024.

Key changes vs original 1D TS-MLP
-----------------------------------
  - Input: N²=9 (3-element L-shaped array) instead of N²=144 (N=12).
  - Stage-1 is split into azimuth (z1_az, 61-d) and elevation (z1_el, 61-d).
  - Stage-2 outputs PER-TARGET decimal vectors:
      z2_az : (K, 100) — row k = decimal-az for k-th azimuth-sorted target.
      z2_el : (K, 100) — row k = decimal-el for k-th elevation-sorted target.
    This removes all within-dimension int-dec pairing ambiguity.
  - Stage-3 (z3) encodes only the cross-dimension az→el pairing (K! outputs).
  - Hidden layer sizes scaled down proportionally to the smaller input.

Network layout
--------------
  Part 1 — integer-degree estimation
      Input  : x  (INPUT_DIM = 9)
      FC(64,ReLU) → Dropout → FC(128,ReLU) → Dropout → FC(64,ReLU) → FC(32,ReLU)
      Outputs: z1_az (61, linear)  |  z1_el (61, linear)

  Part 2 — per-target decimal estimation
      Input  : concat(x, z1_az, z1_el)   shape=(9+61+61=131,)
      FC(128,ReLU) → FC(256,ReLU) → FC(128,ReLU) → FC(64,ReLU) → FC(128,ReLU)
      Outputs: z2_az (K×100, linear → reshaped to (K,100))
               z2_el (K×100, linear → reshaped to (K,100))

  Part 3 — az→el pairing
      Input  : concat(Part1_features(32), Part2_features(128))
      FC(64,ReLU) → FC(32,ReLU) → FC(16,ReLU)
      Output : z3 (K!, sigmoid)
"""

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    INPUT_DIM,
    N_AZ_INT, N_EL_INT, N_DECIMAL,
    DROPOUT_RATE,
    LEARNING_RATE,
    LOSS_WEIGHT_INT, LOSS_WEIGHT_DEC, LOSS_WEIGHT_PAIR,
)


def build_ts_mlp(K: int) -> keras.Model:
    """Build and compile the TS-MLP model for K targets.

    Args:
        K : Number of targets (2, 3, or 4).

    Returns:
        Compiled Keras Model.
        Inputs  : [x]  shape (INPUT_DIM,)
        Outputs : [z1_az, z1_el, z2_az, z2_el, z3]
                  shapes (61,), (61,), (K,100), (K,100), (K!,)
    """
    n_perms = math.factorial(K)

    # ------------------------------------------------------------------ #
    # Input                                                                #
    # ------------------------------------------------------------------ #
    x_in = keras.Input(shape=(INPUT_DIM,), name='input_cov')

    # ------------------------------------------------------------------ #
    # Part 1 — integer-degree estimation                                   #
    # ------------------------------------------------------------------ #
    p1 = layers.Dense(64,  activation='relu', name='p1_fc1')(x_in)
    p1 = layers.Dropout(DROPOUT_RATE,         name='p1_drop1')(p1)
    p1 = layers.Dense(128, activation='relu', name='p1_fc2')(p1)
    p1 = layers.Dropout(DROPOUT_RATE,         name='p1_drop2')(p1)
    p1 = layers.Dense(64,  activation='relu', name='p1_fc3')(p1)
    p1_feat = layers.Dense(32, activation='relu', name='p1_feat')(p1)

    z1_az = layers.Dense(N_AZ_INT, activation=None, name='z1_az')(p1_feat)
    z1_el = layers.Dense(N_EL_INT, activation=None, name='z1_el')(p1_feat)

    # ------------------------------------------------------------------ #
    # Part 2 — per-target decimal estimation                               #
    # ------------------------------------------------------------------ #
    p2_in = layers.Concatenate(name='p2_input')([x_in, z1_az, z1_el])

    p2 = layers.Dense(128, activation='relu', name='p2_fc1')(p2_in)
    p2 = layers.Dense(256, activation='relu', name='p2_fc2')(p2)
    p2 = layers.Dense(128, activation='relu', name='p2_fc3')(p2)
    p2 = layers.Dense(64,  activation='relu', name='p2_fc4')(p2)
    p2_feat = layers.Dense(128, activation='relu', name='p2_feat')(p2)

    # Separate heads for az and el decimals.
    # Each head outputs K*N_DECIMAL values → reshape to (K, N_DECIMAL).
    z2_az_flat = layers.Dense(K * N_DECIMAL, activation=None,
                               name='z2_az_flat')(p2_feat)
    z2_az = layers.Reshape((K, N_DECIMAL), name='z2_az')(z2_az_flat)

    z2_el_flat = layers.Dense(K * N_DECIMAL, activation=None,
                               name='z2_el_flat')(p2_feat)
    z2_el = layers.Reshape((K, N_DECIMAL), name='z2_el')(z2_el_flat)

    # ------------------------------------------------------------------ #
    # Part 3 — az→el pairing                                               #
    # ------------------------------------------------------------------ #
    p3_in = layers.Concatenate(name='p3_input')([p1_feat, p2_feat])

    p3 = layers.Dense(64, activation='relu', name='p3_fc1')(p3_in)
    p3 = layers.Dense(32, activation='relu', name='p3_fc2')(p3)
    p3 = layers.Dense(16, activation='relu', name='p3_fc3')(p3)
    z3 = layers.Dense(n_perms, activation='sigmoid', name='z3')(p3)

    # ------------------------------------------------------------------ #
    # Build model                                                          #
    # ------------------------------------------------------------------ #
    model = keras.Model(
        inputs=x_in,
        outputs=[z1_az, z1_el, z2_az, z2_el, z3],
        name=f'TS_MLP_2D_{K}T'
    )

    # ------------------------------------------------------------------ #
    # Compile                                                              #
    # Stage 1 (integer): MSE — following paper Eq. 9                      #
    # Stage 2 (decimal): MSE per element over (K, 100) tensor             #
    # Stage 3 (pairing): Binary cross-entropy — paper Eq. 10              #
    # ------------------------------------------------------------------ #
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'z1_az': keras.losses.MeanSquaredError(),
            'z1_el': keras.losses.MeanSquaredError(),
            'z2_az': keras.losses.MeanSquaredError(),
            'z2_el': keras.losses.MeanSquaredError(),
            'z3':    keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            'z1_az': LOSS_WEIGHT_INT,
            'z1_el': LOSS_WEIGHT_INT,
            'z2_az': LOSS_WEIGHT_DEC,
            'z2_el': LOSS_WEIGHT_DEC,
            'z3':    LOSS_WEIGHT_PAIR,
        },
        metrics={
            'z1_az': keras.metrics.MeanSquaredError(name='mse'),
            'z1_el': keras.metrics.MeanSquaredError(name='mse'),
            'z2_az': keras.metrics.MeanSquaredError(name='mse'),
            'z2_el': keras.metrics.MeanSquaredError(name='mse'),
            'z3':    keras.metrics.BinaryAccuracy(name='acc'),
        },
    )
    return model


if __name__ == '__main__':
    for K in (2, 3, 4):
        print(f"\n{'='*55}")
        print(f" TS-MLP for K = {K} targets")
        print(f"{'='*55}")
        m = build_ts_mlp(K)
        m.summary()
        print(f"  Outputs: z1_az{m.output[0].shape}, z1_el{m.output[1].shape}, "
              f"z2_az{m.output[2].shape}, z2_el{m.output[3].shape}, "
              f"z3{m.output[4].shape}")
