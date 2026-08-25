"""
Main script: generate and save datasets for K = 2, 3, and 4 targets.

Usage:
    python generate_dataset.py [--targets 2 3 4] [--max_samples N] [--seed S]

Arguments:
    --targets     Which target-count datasets to generate (default: 2 3 4).
    --max_samples Maximum total samples per dataset (0 = generate all).
                  Useful for quick testing; e.g. --max_samples 100000.
    --seed        Base random seed (default: 42).

Output:
    HDF5 files written to the `datasets/` directory:
        datasets/dataset_2targets.h5
        datasets/dataset_3targets.h5
        datasets/dataset_4targets.h5

    Each file has the following HDF5 datasets:
        /X          float32, shape (N_samples, INPUT_DIM)      — covariance features
        /z1_az      float32, shape (N_samples, N_AZ_INT)       — stage-1 az label
        /z1_el      float32, shape (N_samples, N_EL_INT)       — stage-1 el label
        /z2_az      float32, shape (N_samples, K, N_DECIMAL)   — stage-2 az label per target
        /z2_el      float32, shape (N_samples, K, N_DECIMAL)   — stage-2 el label per target
        /z3         float32, shape (N_samples, K!)             — pairing label
        /az_sorted  float32, shape (N_samples, K)              — true sorted azimuths
        /el_sorted  float32, shape (N_samples, K)              — true elevations (matched)

Full dataset sizes (approximate):
    2 targets : 100 × 181 × 5000 × 7 ≈ 633.5 M samples
    3 targets : 100 × 181 × 5000 × 7 ≈ 633.5 M samples
    4 targets : 100 ×  91 × 5000 × 7 ≈ 318.5 M samples
"""

import argparse
import math
import os
import time

import h5py
import numpy as np

from config import (
    DATASET_DIR, DATASET_2T_PATH, DATASET_3T_PATH, DATASET_4T_PATH,
    INPUT_DIM, N_AZ_INT, N_EL_INT, N_DECIMAL,
    N_A_VALUES, N_DELTA_2T, N_DELTA_3T, N_DELTA_4T,
    SAMPLES_PER_COMB_2T, SAMPLES_PER_COMB_3T, SAMPLES_PER_COMB_4T,
    N_SNR,
)
from dataset_generator import (
    gen_samples_2targets,
    gen_samples_3targets,
    gen_samples_4targets,
)


BATCH_SIZE = 50_000   # samples per write-batch


def _total_samples(n_a, n_delta, n_per_comb, n_snr):
    return n_a * n_delta * n_per_comb * n_snr


def _create_hdf5(path: str, K: int, estimated_total: int):
    """Create a new HDF5 file with resizable datasets."""
    n_perms = math.factorial(K)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, 'w')
    f.create_dataset('X',         shape=(0, INPUT_DIM),     dtype='float32',
                     maxshape=(None, INPUT_DIM),     chunks=(BATCH_SIZE, INPUT_DIM),
                     compression='gzip', compression_opts=4)
    f.create_dataset('z1_az',     shape=(0, N_AZ_INT),      dtype='float32',
                     maxshape=(None, N_AZ_INT),      chunks=(BATCH_SIZE, N_AZ_INT),
                     compression='gzip', compression_opts=4)
    f.create_dataset('z1_el',     shape=(0, N_EL_INT),      dtype='float32',
                     maxshape=(None, N_EL_INT),      chunks=(BATCH_SIZE, N_EL_INT),
                     compression='gzip', compression_opts=4)
    # z2_az and z2_el: per-target decimal vectors, shape (N, K, N_DECIMAL)
    f.create_dataset('z2_az',     shape=(0, K, N_DECIMAL),  dtype='float32',
                     maxshape=(None, K, N_DECIMAL),  chunks=(BATCH_SIZE, K, N_DECIMAL),
                     compression='gzip', compression_opts=4)
    f.create_dataset('z2_el',     shape=(0, K, N_DECIMAL),  dtype='float32',
                     maxshape=(None, K, N_DECIMAL),  chunks=(BATCH_SIZE, K, N_DECIMAL),
                     compression='gzip', compression_opts=4)
    f.create_dataset('z3',        shape=(0, n_perms),       dtype='float32',
                     maxshape=(None, n_perms),       chunks=(BATCH_SIZE, n_perms),
                     compression='gzip', compression_opts=4)
    f.create_dataset('az_sorted', shape=(0, K),             dtype='float32',
                     maxshape=(None, K),             chunks=(BATCH_SIZE, K),
                     compression='gzip', compression_opts=4)
    f.create_dataset('el_sorted', shape=(0, K),             dtype='float32',
                     maxshape=(None, K),             chunks=(BATCH_SIZE, K),
                     compression='gzip', compression_opts=4)
    return f


def _append_batch(f: h5py.File, X_batch, labels_batch):
    """Append one batch to all datasets in the HDF5 file."""
    n = X_batch.shape[0]
    for name, data in [('X', X_batch),
                        ('z1_az',     labels_batch['z1_az']),
                        ('z1_el',     labels_batch['z1_el']),
                        ('z2_az',     labels_batch['z2_az']),
                        ('z2_el',     labels_batch['z2_el']),
                        ('z3',        labels_batch['z3']),
                        ('az_sorted', labels_batch['az_sorted']),
                        ('el_sorted', labels_batch['el_sorted'])]:
        ds = f[name]
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        ds[old:old + n] = data


def generate_and_save(K: int, path: str, generator_fn,
                      max_samples: int = 0, seed: int = 42):
    """Run a generator and save samples to HDF5.

    Args:
        K           : Number of targets.
        path        : Output HDF5 file path.
        generator_fn: Dataset generator function (yields (X, labels)).
        max_samples : Stop after this many samples (0 = unlimited).
        seed        : Random seed.
    """
    n_a = N_A_VALUES
    n_delta = {2: N_DELTA_2T, 3: N_DELTA_3T, 4: N_DELTA_4T}[K]
    n_per = {2: SAMPLES_PER_COMB_2T,
             3: SAMPLES_PER_COMB_3T,
             4: SAMPLES_PER_COMB_4T}[K]
    est = _total_samples(n_a, n_delta, n_per, N_SNR)
    limit = min(est, max_samples) if max_samples > 0 else est

    print(f"\n{'='*60}")
    print(f" Generating {K}-target dataset → {path}")
    print(f" Estimated full size : {est:,} samples")
    print(f" Generating up to    : {limit:,} samples")
    print(f"{'='*60}")

    if os.path.exists(path):
        os.remove(path)

    f = _create_hdf5(path, K, limit)
    total = 0
    t0 = time.time()

    try:
        gen = generator_fn(batch_size=BATCH_SIZE, seed=seed)
        for X_batch, labels_batch in gen:
            n = X_batch.shape[0]
            if max_samples > 0 and total + n > max_samples:
                n = max_samples - total
                X_batch = X_batch[:n]
                for k in labels_batch:
                    labels_batch[k] = labels_batch[k][:n]

            _append_batch(f, X_batch, labels_batch)
            total += n
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            eta = (limit - total) / rate if rate > 0 else float('inf')
            print(f"\r  {total:>12,} / {limit:,} samples  "
                  f"({100*total/limit:.1f}%)  "
                  f"{rate:.0f} samples/s  ETA {eta/60:.1f} min",
                  end='', flush=True)

            if max_samples > 0 and total >= max_samples:
                break
    finally:
        f.attrs['n_targets'] = K
        f.attrs['n_samples'] = total
        f.close()

    print(f"\n  Done. {total:,} samples written in {(time.time()-t0)/60:.1f} min.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate DOA estimation datasets (2, 3, or 4 targets).')
    parser.add_argument('--targets', nargs='+', type=int, default=[2, 3, 4],
                        choices=[2, 3, 4],
                        help='Which target-count datasets to generate (default: 2 3 4)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples per dataset (0 = full dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    args = parser.parse_args()

    generators = {
        2: (gen_samples_2targets, DATASET_2T_PATH),
        3: (gen_samples_3targets, DATASET_3T_PATH),
        4: (gen_samples_4targets, DATASET_4T_PATH),
    }

    for K in sorted(set(args.targets)):
        fn, path = generators[K]
        generate_and_save(K, path, fn, args.max_samples, args.seed + K)


if __name__ == '__main__':
    main()
