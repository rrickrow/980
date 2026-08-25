"""
Label generation for the 2D TS-MLP DOA estimation system.

Design:  OPTION-B  —  per-target decimal vectors + single cross-dimension pairing

Label structure per sample
==========================

  z1_az   : Stage-1 integer-azimuth  label  [N_AZ_INT]       (61 values)
  z1_el   : Stage-1 integer-elevation label  [N_EL_INT]       (61 values)
  z2_az   : Stage-2 decimal-azimuth   labels [K × N_DECIMAL]  (K × 100 values)
             Row k → decimal az of the k-th AZIMUTH-sorted target (single peak).
  z2_el   : Stage-2 decimal-elevation labels [K × N_DECIMAL]  (K × 100 values)
             Row k → decimal el of the k-th ELEVATION-sorted target (single peak).
  z3      : Stage-3 pairing label            [K!]             (2 / 6 / 24 values)
             One-hot encoding of the permutation p such that the k-th az-sorted
             target has elevation rank p[k] in the elevation-sorted list.

Why separate decimal vectors?
------------------------------
In the original 1D TS-MLP the z2 output is a multi-peak vector (K peaks) and
an extra z3 resolves the int–dec pairing ambiguity.  When extending to 2D we
need to resolve THREE ambiguities:
  1. az int  ↔ az dec  (within azimuth dimension)
  2. el int  ↔ el dec  (within elevation dimension)
  3. az DOA  ↔ el DOA  (cross-dimension)

Using K SEPARATE single-peak decimal vectors per dimension (z2_az, z2_el)
removes ambiguities (1) and (2) completely: the k-th vector by construction
belongs to the k-th sorted target.  Only ambiguity (3) remains, resolved by z3.

Prerequisites enforced by the dataset generator
------------------------------------------------
  * All K integer azimuths must be DISTINCT  (|round(az_i) − round(az_j)| ≥ 1)
  * All K integer elevations must be DISTINCT (|round(el_i) − round(el_j)| ≥ 1)
These constraints guarantee a clean 1-to-1 mapping between integer peaks in
z1_az (z1_el) and the K targets.
"""

import math
from itertools import permutations

import numpy as np

from config import (
    AZ_MIN, EL_MIN,
    N_AZ_INT, N_EL_INT,
    N_DECIMAL, DECIMAL_GRID, DECIMAL_STEP,
    LABEL_PEAK_VALUE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _int_index(angle_deg: float, grid_min: float) -> int:
    """Stage-1 grid index for an angle (nearest integer degree)."""
    return int(round(angle_deg)) - int(grid_min)


def _dec_index(angle_deg: float) -> int:
    """Stage-2 grid index for the decimal part of an angle."""
    integer_part = round(angle_deg)
    decimal_part = angle_deg - integer_part          # in [-0.5, 0.5)
    decimal_part = max(DECIMAL_GRID[0], min(DECIMAL_GRID[-1], decimal_part))
    idx = int(round((decimal_part - DECIMAL_GRID[0]) / DECIMAL_STEP))
    return max(0, min(N_DECIMAL - 1, idx))


def _check_distinct_integers(az_list, el_list):
    """Return True iff all integer azimuths and all integer elevations are distinct."""
    az_ints = [int(round(a)) for a in az_list]
    el_ints = [int(round(e)) for e in el_list]
    return len(set(az_ints)) == len(az_ints) and len(set(el_ints)) == len(el_ints)


# Pre-build permutation lookup tables for K = 2, 3, 4
_PERMS = {K: list(permutations(range(K))) for K in (2, 3, 4)}
# Inverse lookup: tuple → index
_PERM_INDEX = {K: {p: i for i, p in enumerate(_PERMS[K])} for K in (2, 3, 4)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_labels(az_list, el_list) -> dict:
    """Generate all TS-MLP labels for K targets.

    The targets are sorted internally:
      * By AZIMUTH  for z1_az and each row of z2_az.
      * By ELEVATION for z1_el and each row of z2_el.
    z3 encodes the permutation that maps az-sorted rank → el-sorted rank.

    Args:
        az_list : Azimuth angles [degrees], length K ∈ {2, 3, 4}.
        el_list : Elevation angles [degrees], length K.

    Returns:
        dict with keys:
            'z1_az'    : (N_AZ_INT,)  float32  — multi-hot, K peaks
            'z1_el'    : (N_EL_INT,)  float32  — multi-hot, K peaks
            'z2_az'    : (K, N_DECIMAL) float32 — per-target, az-sorted order
            'z2_el'    : (K, N_DECIMAL) float32 — per-target, el-sorted order
            'z3'       : (K!,)        float32  — one-hot az→el pairing
            'az_sorted': (K,) sorted azimuths (for diagnostics)
            'el_sorted': (K,) elevations in az-sorted order (for diagnostics)

    Raises:
        ValueError  if K not in {2, 3, 4}.
        ValueError  if integer az or el parts are not all distinct.
    """
    K = len(az_list)
    if K not in (2, 3, 4):
        raise ValueError(f"K must be 2, 3 or 4; got {K}")

    az_arr = np.asarray(az_list, dtype=float)
    el_arr = np.asarray(el_list, dtype=float)

    if not _check_distinct_integers(az_arr, el_arr):
        raise ValueError(
            "All integer azimuths AND all integer elevations must be distinct. "
            f"az={az_arr.tolist()}, el={el_arr.tolist()}"
        )

    # Sort by azimuth
    az_order = np.argsort(az_arr)
    az_s = az_arr[az_order]          # ascending az
    el_s = el_arr[az_order]          # el in az-sorted order

    # Sort by elevation
    el_order = np.argsort(el_arr)
    el_sorted = el_arr[el_order]     # ascending el
    az_for_el = az_arr[el_order]     # az in el-sorted order (for diagnostics)

    # ---- Stage-1: integer peaks ----
    z1_az = np.zeros(N_AZ_INT, dtype=np.float32)
    z1_el = np.zeros(N_EL_INT, dtype=np.float32)
    for k in range(K):
        i_az = _int_index(az_s[k], AZ_MIN)
        i_el = _int_index(el_sorted[k], EL_MIN)
        if 0 <= i_az < N_AZ_INT:
            z1_az[i_az] = LABEL_PEAK_VALUE
        if 0 <= i_el < N_EL_INT:
            z1_el[i_el] = LABEL_PEAK_VALUE

    # ---- Stage-2: per-target decimal vectors ----
    z2_az = np.zeros((K, N_DECIMAL), dtype=np.float32)    # rows = az-sorted targets
    z2_el = np.zeros((K, N_DECIMAL), dtype=np.float32)    # rows = el-sorted targets
    for k in range(K):
        j_az = _dec_index(az_s[k])
        j_el = _dec_index(el_sorted[k])
        z2_az[k, j_az] = LABEL_PEAK_VALUE
        z2_el[k, j_el] = LABEL_PEAK_VALUE

    # ---- Stage-3: az→el pairing ----
    # For each az-sorted target (rank k), find its rank in the el-sorted list.
    # Build a map: original index → el rank
    el_rank_of_orig = np.empty(K, dtype=int)
    for el_rank, orig_idx in enumerate(el_order):
        el_rank_of_orig[orig_idx] = el_rank
    # Compose with az ordering: az_rank k → original index az_order[k] → el_rank
    pairing = tuple(int(el_rank_of_orig[az_order[k]]) for k in range(K))

    n_perms = math.factorial(K)
    z3 = np.zeros(n_perms, dtype=np.float32)
    z3[_PERM_INDEX[K][pairing]] = 1.0

    return {
        'z1_az': z1_az,
        'z1_el': z1_el,
        'z2_az': z2_az,            # (K, N_DECIMAL), row k = decimal az for az-sorted target k
        'z2_el': z2_el,            # (K, N_DECIMAL), row k = decimal el for el-sorted target k
        'z3': z3,
        'az_sorted': az_s,
        'el_sorted': el_s,
    }


def decode_prediction(z1_az_pred, z1_el_pred,
                      z2_az_pred, z2_el_pred,
                      z3_pred, K: int):
    """Decode network outputs into estimated (azimuth, elevation) pairs.

    Args:
        z1_az_pred : (N_AZ_INT,)     — Stage-1 az output.
        z1_el_pred : (N_EL_INT,)     — Stage-1 el output.
        z2_az_pred : (K, N_DECIMAL)  — Stage-2 az outputs, one row per az-sorted target.
        z2_el_pred : (K, N_DECIMAL)  — Stage-2 el outputs, one row per el-sorted target.
        z3_pred    : (K!,)           — Stage-3 pairing output.
        K          : Number of targets.

    Returns:
        List of K (azimuth_deg, elevation_deg) tuples, in azimuth-sorted order.
    """
    z2_az_pred = np.asarray(z2_az_pred)   # (K, 100)
    z2_el_pred = np.asarray(z2_el_pred)   # (K, 100)

    # Stage-1: integer degrees (top-K peaks, sort indices ascending)
    az_int_top = np.argsort(z1_az_pred)[-K:]
    az_int_sorted_idx = np.sort(az_int_top)
    az_ints = AZ_MIN + az_int_sorted_idx.astype(float)   # (K,) ascending az

    el_int_top = np.argsort(z1_el_pred)[-K:]
    el_int_sorted_idx = np.sort(el_int_top)
    el_ints = EL_MIN + el_int_sorted_idx.astype(float)   # (K,) ascending el

    # Stage-2: decimal degrees (argmax of each per-target vector)
    az_decs = np.array([DECIMAL_GRID[int(np.argmax(z2_az_pred[k]))] for k in range(K)])
    el_decs = np.array([DECIMAL_GRID[int(np.argmax(z2_el_pred[k]))] for k in range(K)])

    az_final = az_ints + az_decs                # (K,) full az, ascending
    el_final_sorted = el_ints + el_decs         # (K,) full el, ascending (el-sorted order)

    # Stage-3: az→el pairing
    perm_idx = int(np.argmax(z3_pred))
    pairing = _PERMS[K][perm_idx]

    results = []
    for k in range(K):
        az_est = float(az_final[k])
        el_est = float(el_final_sorted[pairing[k]])
        results.append((az_est, el_est))
    return results
