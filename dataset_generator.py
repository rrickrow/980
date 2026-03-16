"""
Dataset generation for the 2D DOA estimation system.

Three separate generators are provided (2, 3, and 4 targets).
Each generator:
  1. Iterates over amplitude ratio *a* and angle-separation Δθ (or Δθ_min).
  2. Draws SAMPLES_PER_COMB random valid angle configurations per combination.
  3. Repeats each configuration for every SNR in SNR_LIST.
  4. Computes the TRUE covariance matrix (used for training, following the paper).
  5. Vectorises the covariance and assembles labels via `label_generator`.
  6. Yields batches of (X, y) for downstream HDF5 writing.

Validity constraints on angle configurations
---------------------------------------------
A configuration is valid iff:
  * The Euclidean angular separation between ALL target pairs ≥ Δθ (or Δθ_min).
  * All K integer azimuths are DISTINCT  (|round(az_i) − round(az_j)| ≥ 1).
  * All K integer elevations are DISTINCT (|round(el_i) − round(el_j)| ≥ 1).
The last two constraints ensure clean single-peak stage-1 outputs for both
azimuth and elevation, enabling unambiguous stage-2 per-target decoding.
Samples violating these constraints are rejected and resampled.

Label format (z2_az, z2_el shape change vs. 1D baseline)
----------------------------------------------------------
z2_az : (K, N_DECIMAL)  — row k holds the decimal-az label for the k-th
                          azimuth-sorted target (single peak per row).
z2_el : (K, N_DECIMAL)  — row k holds the decimal-el label for the k-th
                          elevation-sorted target.
z3    : (K!,)           — one-hot az→el pairing label.

See `generate_dataset.py` for the top-level script.

Allocation strategies
---------------------
2 targets  (follow paper strategy):
    a ∈ A_VALUES (100), Δθ ∈ DELTA_ANGLE_VALUES_2T (181), 5000 samples/combo.
    Total raw samples = 100 × 181 × 5000 × 7 ≈ 633.5 M.

3 targets:
    a ∈ A_VALUES (100), Δθ_min ∈ DELTA_ANGLE_VALUES_3T (181), 5000 samples/combo.
    Total raw samples = 100 × 181 × 5000 × 7 ≈ 633.5 M.

4 targets:
    a ∈ A_VALUES (100), Δθ_min ∈ DELTA_ANGLE_VALUES_4T (91), 5000 samples/combo.
    Total raw samples = 100 × 91 × 5000 × 7 ≈ 318.5 M.
"""

import math
from typing import Generator, Tuple

import numpy as np

from config import (
    AZ_MIN, AZ_MAX, EL_MIN, EL_MAX,
    A_VALUES, SNR_LIST,
    DELTA_ANGLE_VALUES_2T, SAMPLES_PER_COMB_2T,
    DELTA_ANGLE_VALUES_3T, SAMPLES_PER_COMB_3T,
    DELTA_ANGLE_VALUES_4T, SAMPLES_PER_COMB_4T,
    INPUT_DIM,
)
from signal_model import true_covariance, vectorize_covariance
from label_generator import generate_labels


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------

def _angular_separation(az1, el1, az2, el2) -> float:
    """Euclidean distance in the 2D angle space [degrees]."""
    return math.sqrt((az1 - az2) ** 2 + (el1 - el2) ** 2)


def _distinct_integers(az_list, el_list) -> bool:
    """True iff all integer azimuths AND all integer elevations are distinct."""
    az_ints = [int(round(a)) for a in az_list]
    el_ints = [int(round(e)) for e in el_list]
    return (len(set(az_ints)) == len(az_ints)
            and len(set(el_ints)) == len(el_ints))


# ---------------------------------------------------------------------------
# 2-target angle sampling
# ---------------------------------------------------------------------------

def _sample_pair(rng: np.random.Generator, delta: float,
                 max_tries: int = 500) -> Tuple:
    """Return (az1, el1, az2, el2) satisfying the validity constraints.

    Returns None if max_tries exceeded without finding a valid pair.
    """
    for _ in range(max_tries):
        az1 = rng.uniform(AZ_MIN, AZ_MAX)
        el1 = rng.uniform(EL_MIN, EL_MAX)
        alpha = rng.uniform(0.0, 2.0 * math.pi)
        az2 = az1 + delta * math.cos(alpha)
        el2 = el1 + delta * math.sin(alpha)
        # Clamp to search space
        az2 = max(AZ_MIN, min(AZ_MAX, az2))
        el2 = max(EL_MIN, min(EL_MAX, el2))
        if _distinct_integers([az1, az2], [el1, el2]):
            return az1, el1, az2, el2
    return None


# ---------------------------------------------------------------------------
# 3-target angle sampling
# ---------------------------------------------------------------------------

def _sample_triplet(rng: np.random.Generator, delta_min: float,
                    max_tries: int = 500) -> Tuple:
    """Return (az1,el1, az2,el2, az3,el3) satisfying validity constraints."""
    for _ in range(max_tries):
        az = rng.uniform(AZ_MIN, AZ_MAX, 3)
        el = rng.uniform(EL_MIN, EL_MAX, 3)
        sep_12 = _angular_separation(az[0], el[0], az[1], el[1])
        sep_13 = _angular_separation(az[0], el[0], az[2], el[2])
        sep_23 = _angular_separation(az[1], el[1], az[2], el[2])
        if min(sep_12, sep_13, sep_23) >= delta_min:
            if _distinct_integers(az, el):
                return (az[0], el[0], az[1], el[1], az[2], el[2])
    return None


# ---------------------------------------------------------------------------
# 4-target angle sampling
# ---------------------------------------------------------------------------

def _sample_quadruplet(rng: np.random.Generator, delta_min: float,
                       max_tries: int = 1000) -> Tuple:
    """Return (az1,el1, …, az4,el4) satisfying validity constraints."""
    for _ in range(max_tries):
        az = rng.uniform(AZ_MIN, AZ_MAX, 4)
        el = rng.uniform(EL_MIN, EL_MAX, 4)
        ok = all(
            _angular_separation(az[i], el[i], az[j], el[j]) >= delta_min
            for i in range(4) for j in range(i + 1, 4)
        )
        if ok and _distinct_integers(az, el):
            return (az[0], el[0], az[1], el[1],
                    az[2], el[2], az[3], el[3])
    return None


# ---------------------------------------------------------------------------
# Buffer helper
# ---------------------------------------------------------------------------

def _new_label_buf():
    return {'z1_az': [], 'z1_el': [], 'z2_az': [], 'z2_el': [], 'z3': []}


def _flush(X_buf, label_buf, meta_buf, n):
    """Stack buffer contents into arrays and return (X, labels) dict."""
    out = {
        'z1_az':    np.stack(label_buf['z1_az'][:n]),
        'z1_el':    np.stack(label_buf['z1_el'][:n]),
        'z2_az':    np.stack(label_buf['z2_az'][:n]),   # (n, K, N_DECIMAL)
        'z2_el':    np.stack(label_buf['z2_el'][:n]),   # (n, K, N_DECIMAL)
        'z3':       np.stack(label_buf['z3'][:n]),
        'az_sorted': np.stack(meta_buf['az'][:n]),
        'el_sorted': np.stack(meta_buf['el'][:n]),
    }
    return X_buf[:n].copy(), out


def _add_sample(X_buf, label_buf, meta_buf, buf_pos,
                x, labels, batch_size):
    """Insert one processed sample; return updated buf_pos and any ready batch."""
    X_buf[buf_pos] = x
    label_buf['z1_az'].append(labels['z1_az'])
    label_buf['z1_el'].append(labels['z1_el'])
    label_buf['z2_az'].append(labels['z2_az'])   # (K, N_DECIMAL)
    label_buf['z2_el'].append(labels['z2_el'])   # (K, N_DECIMAL)
    label_buf['z3'].append(labels['z3'])
    meta_buf['az'].append(labels['az_sorted'])
    meta_buf['el'].append(labels['el_sorted'])
    buf_pos += 1

    if buf_pos == batch_size:
        batch = _flush(X_buf, label_buf, meta_buf, buf_pos)
        buf_pos = 0
        label_buf.clear()
        label_buf.update(_new_label_buf())
        meta_buf.clear()
        meta_buf.update({'az': [], 'el': []})
        return buf_pos, label_buf, meta_buf, batch

    return buf_pos, label_buf, meta_buf, None


# ---------------------------------------------------------------------------
# 2-target dataset generator
# ---------------------------------------------------------------------------

def gen_samples_2targets(batch_size: int = 10000,
                          seed: int = 42) -> Generator:
    """Yield batches of (X, labels) for the 2-target dataset.

    Args:
        batch_size : Number of samples per yielded batch.
        seed       : Random seed for reproducibility.

    Yields:
        (X_batch, labels_batch) where
            X_batch             : float32 (batch_size, INPUT_DIM=9)
            labels_batch['z1_az']: float32 (batch_size, 61)
            labels_batch['z1_el']: float32 (batch_size, 61)
            labels_batch['z2_az']: float32 (batch_size, 2, 100)
            labels_batch['z2_el']: float32 (batch_size, 2, 100)
            labels_batch['z3']   : float32 (batch_size, 2)
            labels_batch['az_sorted']: float32 (batch_size, 2)
            labels_batch['el_sorted']: float32 (batch_size, 2)
    """
    rng = np.random.default_rng(seed)
    X_buf = np.zeros((batch_size, INPUT_DIM), dtype=np.float32)
    label_buf = _new_label_buf()
    meta_buf = {'az': [], 'el': []}
    buf_pos = 0

    for a in A_VALUES:
        for delta in DELTA_ANGLE_VALUES_2T:
            n_drawn = 0
            while n_drawn < SAMPLES_PER_COMB_2T:
                pair = _sample_pair(rng, float(delta))
                if pair is None:
                    continue
                az1, el1, az2, el2 = pair
                n_drawn += 1
                for snr in SNR_LIST:
                    R = true_covariance([az1, az2], [el1, el2],
                                        [1.0, float(a)], snr)
                    x = vectorize_covariance(R)
                    labels = generate_labels([az1, az2], [el1, el2])
                    buf_pos, label_buf, meta_buf, batch = _add_sample(
                        X_buf, label_buf, meta_buf, buf_pos, x, labels, batch_size)
                    if batch is not None:
                        yield batch

    # Flush remainder
    if buf_pos > 0:
        yield _flush(X_buf, label_buf, meta_buf, buf_pos)


# ---------------------------------------------------------------------------
# 3-target dataset generator
# ---------------------------------------------------------------------------

def gen_samples_3targets(batch_size: int = 10000,
                          seed: int = 42) -> Generator:
    """Yield batches of (X, labels) for the 3-target dataset.

    Allocation strategy (see module docstring).

    Yields:
        Same structure as gen_samples_2targets but z2_az/z2_el have shape
        (batch_size, 3, 100) and z3 has shape (batch_size, 6).
    """
    rng = np.random.default_rng(seed)
    X_buf = np.zeros((batch_size, INPUT_DIM), dtype=np.float32)
    label_buf = _new_label_buf()
    meta_buf = {'az': [], 'el': []}
    buf_pos = 0

    for a in A_VALUES:
        for delta_min in DELTA_ANGLE_VALUES_3T:
            n_drawn = 0
            while n_drawn < SAMPLES_PER_COMB_3T:
                triplet = _sample_triplet(rng, float(delta_min))
                if triplet is None:
                    continue
                az1, el1, az2, el2, az3, el3 = triplet
                n_drawn += 1
                for snr in SNR_LIST:
                    R = true_covariance(
                        [az1, az2, az3], [el1, el2, el3],
                        [1.0, float(a), float(a)], snr)
                    x = vectorize_covariance(R)
                    labels = generate_labels([az1, az2, az3], [el1, el2, el3])
                    buf_pos, label_buf, meta_buf, batch = _add_sample(
                        X_buf, label_buf, meta_buf, buf_pos, x, labels, batch_size)
                    if batch is not None:
                        yield batch

    if buf_pos > 0:
        yield _flush(X_buf, label_buf, meta_buf, buf_pos)


# ---------------------------------------------------------------------------
# 4-target dataset generator
# ---------------------------------------------------------------------------

def gen_samples_4targets(batch_size: int = 10000,
                          seed: int = 42) -> Generator:
    """Yield batches of (X, labels) for the 4-target dataset.

    Allocation strategy (see module docstring).

    Yields:
        Same structure as gen_samples_2targets but z2_az/z2_el have shape
        (batch_size, 4, 100) and z3 has shape (batch_size, 24).
    """
    rng = np.random.default_rng(seed)
    X_buf = np.zeros((batch_size, INPUT_DIM), dtype=np.float32)
    label_buf = _new_label_buf()
    meta_buf = {'az': [], 'el': []}
    buf_pos = 0

    for a in A_VALUES:
        for delta_min in DELTA_ANGLE_VALUES_4T:
            n_drawn = 0
            while n_drawn < SAMPLES_PER_COMB_4T:
                quad = _sample_quadruplet(rng, float(delta_min))
                if quad is None:
                    continue
                az1, el1, az2, el2, az3, el3, az4, el4 = quad
                n_drawn += 1
                for snr in SNR_LIST:
                    R = true_covariance(
                        [az1, az2, az3, az4],
                        [el1, el2, el3, el4],
                        [1.0, float(a), float(a), float(a)], snr)
                    x = vectorize_covariance(R)
                    labels = generate_labels(
                        [az1, az2, az3, az4],
                        [el1, el2, el3, el4])
                    buf_pos, label_buf, meta_buf, batch = _add_sample(
                        X_buf, label_buf, meta_buf, buf_pos, x, labels, batch_size)
                    if batch is not None:
                        yield batch

    if buf_pos > 0:
        yield _flush(X_buf, label_buf, meta_buf, buf_pos)
