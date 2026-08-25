"""
Evaluation / inference script for the trained TS-MLP 2D DOA model.

Usage:
    python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5
    python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5 --snr_list 0 5 10 15 20
    python evaluate.py --K 2 --model models_saved/ts_mlp_K2_best.h5 --use_sample_cov

Two evaluation modes
--------------------
True covariance (default):
    Mirrors training conditions.  Generates the analytical covariance matrix
    R = A Rs A^H + σ²I and feeds its vectorised form to the network.

Sample covariance (--use_sample_cov):
    Simulates L = N_SNAPSHOTS = 2 pulses of the single-frequency pulse signal,
    computes the sample covariance R̂, and feeds it to the network.
    This reflects realistic inference conditions.

Reported metrics
----------------
  RMSE (azimuth)   = sqrt( E[(φ̂ - φ)²] )   [degrees]
  RMSE (elevation) = sqrt( E[(θ̂ - θ)²] )   [degrees]
  Pairing accuracy = fraction of correctly paired (az, el) estimates [%]
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    AZ_MIN, AZ_MAX, EL_MIN, EL_MAX, SNR_LIST, N_SNAPSHOTS,
    A_VALUES,
)
from signal_model import (
    true_covariance, sample_covariance, vectorize_covariance,
)
from label_generator import decode_prediction, generate_labels

N_TEST_PER_SNR = 1000   # random test samples per SNR level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse(true_vals, pred_vals):
    return float(np.sqrt(np.mean((np.asarray(true_vals) -
                                  np.asarray(pred_vals)) ** 2)))


def _generate_test_batch(K: int, snr_db: float, n: int,
                          use_sample_cov: bool, rng: np.random.Generator):
    """Generate n test samples for K targets at a given SNR."""
    Xs, true_az, true_el = [], [], []
    while len(Xs) < n:
        az = rng.uniform(AZ_MIN, AZ_MAX, K)
        el = rng.uniform(EL_MIN, EL_MAX, K)
        a = rng.choice(A_VALUES)
        amps = [1.0] + [float(a)] * (K - 1)

        if use_sample_cov:
            R = sample_covariance(list(az), list(el), amps, snr_db,
                                  n_snapshots=N_SNAPSHOTS)
        else:
            R = true_covariance(list(az), list(el), amps, snr_db)

        x = vectorize_covariance(R)
        Xs.append(x)

        order = np.argsort(az)
        true_az.append(az[order])
        true_el.append(el[order])

    return np.stack(Xs), np.stack(true_az), np.stack(true_el)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(K: int, model_path: str, snr_list, use_sample_cov: bool):
    print(f"\nEvaluating TS-MLP for K = {K} targets")
    print(f"Model : {model_path}")
    print(f"Mode  : {'sample covariance' if use_sample_cov else 'true covariance'}")

    model = keras.models.load_model(model_path, compile=False)
    rng = np.random.default_rng(0)

    header = f"{'SNR (dB)':>10}  {'RMSE_az (°)':>12}  {'RMSE_el (°)':>12}  {'Pairing acc (%)':>16}"
    print("\n" + header)
    print("-" * len(header))

    for snr in snr_list:
        X, true_az, true_el = _generate_test_batch(
            K, snr, N_TEST_PER_SNR, use_sample_cov, rng)

        preds = model.predict(X, batch_size=256, verbose=0)
        z1_az_p, z1_el_p, z2_az_p, z2_el_p, z3_p = preds

        est_az, est_el, pair_correct = [], [], []
        for i in range(N_TEST_PER_SNR):
            result = decode_prediction(
                z1_az_p[i], z1_el_p[i],
                z2_az_p[i], z2_el_p[i],
                z3_p[i], K)
            e_az = np.array([r[0] for r in result])
            e_el = np.array([r[1] for r in result])
            # Match to closest true target
            order = np.argsort(e_az)
            est_az.append(e_az[order])
            est_el.append(e_el[order])

            # Pairing check: re-compute true label and compare
            true_lbl = generate_labels(
                list(true_az[i]), list(true_el[i]))
            pred_pair = int(np.argmax(z3_p[i]))
            true_pair = int(np.argmax(true_lbl['z3']))
            pair_correct.append(pred_pair == true_pair)

        est_az = np.stack(est_az)
        est_el = np.stack(est_el)

        rmse_az = _rmse(true_az, est_az)
        rmse_el = _rmse(true_el, est_el)
        acc = 100.0 * np.mean(pair_correct)

        print(f"{snr:>10}  {rmse_az:>12.4f}  {rmse_el:>12.4f}  {acc:>16.1f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global N_TEST_PER_SNR  # must be declared before any use within this function

    parser = argparse.ArgumentParser(
        description='Evaluate the trained TS-MLP 2D DOA model.')
    parser.add_argument('--K', type=int, required=True, choices=[2, 3, 4])
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5)')
    parser.add_argument('--snr_list', type=int, nargs='+',
                        default=SNR_LIST,
                        help='SNR values to evaluate [dB]')
    parser.add_argument('--use_sample_cov', action='store_true',
                        help='Use sample covariance (simulates real inference)')
    parser.add_argument('--n_test', type=int, default=N_TEST_PER_SNR,
                        help='Number of test samples per SNR level')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model file not found: {args.model}")
        sys.exit(1)

    N_TEST_PER_SNR = args.n_test

    evaluate(args.K, args.model, args.snr_list, args.use_sample_cov)


if __name__ == '__main__':
    main()
