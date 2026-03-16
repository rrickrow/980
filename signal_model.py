"""
Signal model for the 2D pulse-radar DOA estimation system.

Key changes from the original TS-MLP paper:
  1. Signal model: s(t) is a single-frequency pulse (not complex Gaussian).
  2. Array geometry: 3-element L-shaped planar array for 2D (az + el) DOA.
  3. Covariance structure: unchanged for TRAINING (true covariance has the same
     algebraic form; the pulse model only affects sample-covariance generation
     used in TESTING / inference).

Pulse signal parameters:
    Pulse carrier frequency : 5 GHz
    Pulse width             : 50 ns
    Sampling frequency      : 250 MHz
    Pulse repetition freq.  : 1000 Hz
    Beam pointing           : (0°, 0°)  [azimuth, elevation]

Array:
    3-element L-shaped array in the x-y plane.
    Element 1 at (0, 0), Element 2 at (d, 0), Element 3 at (0, d),
    where d = λ/2 (half-wavelength spacing).
"""

import numpy as np
from config import (
    C_LIGHT, WAVELENGTH, D_ELEMENT,
    N_ELEMENTS, N_SNAPSHOTS, ARRAY_POS,
    F_SAMPLE, PULSE_WIDTH,
)


# ---------------------------------------------------------------------------
# Steering vectors and manifold matrix
# ---------------------------------------------------------------------------

def steering_vector(az_deg: float, el_deg: float,
                    array_pos: np.ndarray = ARRAY_POS,
                    wavelength: float = WAVELENGTH) -> np.ndarray:
    """Compute the complex steering vector for one target direction.

    Uses the far-field, narrow-band model.  For an element at position
    (x_n, y_n) in the horizontal plane and a target at azimuth φ and
    elevation θ:

        a_n = exp( j · 2π/λ · (x_n · sin(φ)cos(θ) + y_n · sin(θ)) )

    Args:
        az_deg    : Azimuth angle [degrees], measured from boresight in
                    the horizontal plane.
        el_deg    : Elevation angle [degrees], measured from the
                    horizontal plane.
        array_pos : Element positions [N × 2], in metres.
        wavelength: Signal wavelength [m].

    Returns:
        Steering vector of shape (N,), dtype complex128.
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    phase = (2.0 * np.pi / wavelength) * (
        array_pos[:, 0] * np.sin(az) * np.cos(el) +
        array_pos[:, 1] * np.sin(el)
    )
    return np.exp(1j * phase)


def manifold_matrix(az_list, el_list,
                    array_pos: np.ndarray = ARRAY_POS,
                    wavelength: float = WAVELENGTH) -> np.ndarray:
    """Assemble the array manifold matrix A = [a(φ_1,θ_1) … a(φ_K,θ_K)].

    Args:
        az_list   : Iterable of K azimuth angles [degrees].
        el_list   : Iterable of K elevation angles [degrees].
        array_pos : Element positions [N × 2], in metres.
        wavelength: Signal wavelength [m].

    Returns:
        Manifold matrix of shape (N, K), dtype complex128.
    """
    K = len(az_list)
    N = array_pos.shape[0]
    A = np.zeros((N, K), dtype=complex)
    for k in range(K):
        A[:, k] = steering_vector(az_list[k], el_list[k], array_pos, wavelength)
    return A


# ---------------------------------------------------------------------------
# Covariance matrices
# ---------------------------------------------------------------------------

def true_covariance(az_list, el_list, amplitudes, snr_db: float,
                    array_pos: np.ndarray = ARRAY_POS,
                    wavelength: float = WAVELENGTH,
                    noise_power: float = 1.0) -> np.ndarray:
    """Compute the analytical (true) covariance matrix R.

    Model:  R = A · Rs · A^H + σ²·I_N

    where Rs = diag(P_1, …, P_K) with P_k = snr_linear · σ² · amplitude_k².

    For the single-frequency pulse signal model:
        s_k(t_l) = A_k · exp(j·φ_{k,l}),  φ_{k,l} ~ Uniform[0, 2π) i.i.d.

    Because the phases are independent across sources and snapshots,
    E[s(t)·s(t)^H] = diag(A_1², …, A_K²).  This is algebraically identical
    to the zero-mean complex-Gaussian model used in the original paper, so
    the true covariance formula is unchanged.  The pulse parameters govern
    sample-covariance behaviour (see `sample_covariance`).

    Args:
        az_list    : Azimuth angles [degrees], length K.
        el_list    : Elevation angles [degrees], length K.
        amplitudes : Relative amplitudes [A_1=1 reference, A_k for k≥2].
        snr_db     : SNR in dB (defined for reference target with amplitude=1).
        array_pos  : Element positions [N × 2], in metres.
        wavelength : Signal wavelength [m].
        noise_power: Noise power σ² (default 1.0, normalised).

    Returns:
        True covariance matrix of shape (N, N), dtype complex128.
    """
    A_mat = manifold_matrix(az_list, el_list, array_pos, wavelength)
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_powers = np.array([snr_linear * noise_power * (amp ** 2)
                               for amp in amplitudes])
    Rs = np.diag(signal_powers)
    N = array_pos.shape[0]
    R = A_mat @ Rs @ A_mat.conj().T + noise_power * np.eye(N)
    return R


def sample_covariance(az_list, el_list, amplitudes, snr_db: float,
                      n_snapshots: int = N_SNAPSHOTS,
                      array_pos: np.ndarray = ARRAY_POS,
                      wavelength: float = WAVELENGTH,
                      noise_power: float = 1.0) -> np.ndarray:
    """Estimate the sample covariance matrix by simulating the pulse signal.

    Pulse signal model (complex baseband, after range-gating):
        For snapshot l:  y(t_l) = A · s(t_l) + w(t_l)
        s_k(t_l) = sqrt(P_k) · exp(j·φ_{k,l}),  φ_{k,l} ~ U[0, 2π)
        w(t_l)   ~ CN(0, σ²·I_N)

    The random initial phase per snapshot models independent pulse
    realisations (e.g. different range positions within the pulse, or
    different PRIs with random target phase).

    Sample covariance:  R̂ = (1/L) · Y · Y^H

    This function is used for TEST / inference data generation only.
    TRAINING uses `true_covariance`.

    Args:
        az_list     : Azimuth angles [degrees], length K.
        el_list     : Elevation angles [degrees], length K.
        amplitudes  : Relative amplitudes, length K.
        snr_db      : SNR in dB (reference target).
        n_snapshots : Number of snapshots L (= number of pulse samples).
        array_pos   : Element positions [N × 2], in metres.
        wavelength  : Signal wavelength [m].
        noise_power : Noise power σ².

    Returns:
        Sample covariance matrix of shape (N, N), dtype complex128.
    """
    K = len(az_list)
    N = array_pos.shape[0]
    A_mat = manifold_matrix(az_list, el_list, array_pos, wavelength)

    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_amps = np.array([np.sqrt(snr_linear * noise_power) * amp
                            for amp in amplitudes])

    Y = np.zeros((N, n_snapshots), dtype=complex)
    for l in range(n_snapshots):
        phases = np.random.uniform(0.0, 2.0 * np.pi, K)
        s = signal_amps * np.exp(1j * phases)
        noise_std = np.sqrt(noise_power / 2.0)
        w = noise_std * (np.random.randn(N) + 1j * np.random.randn(N))
        Y[:, l] = A_mat @ s + w

    R_hat = (Y @ Y.conj().T) / n_snapshots
    return R_hat


# ---------------------------------------------------------------------------
# Covariance vectorisation
# ---------------------------------------------------------------------------

def vectorize_covariance(R: np.ndarray) -> np.ndarray:
    """Convert a Hermitian covariance matrix to a real input vector.

    Only the lower-triangular part (including diagonal) is used, since
    the upper part contains redundant information.

    Layout (for N = 3):
        [R[0,0],                         <- diagonal (real), 3 values
         R[1,1],
         R[2,2],
         Re(R[1,0]), Im(R[1,0]),         <- lower off-diagonal (complex→2 real)
         Re(R[2,0]), Im(R[2,0]),
         Re(R[2,1]), Im(R[2,1])]
    Total length = N² = 9  (for N=3).

    Args:
        R: Hermitian covariance matrix of shape (N, N).

    Returns:
        Real feature vector of length N², dtype float32.
    """
    N = R.shape[0]
    result = []
    for i in range(N):                    # diagonal (real)
        result.append(float(R[i, i].real))
    for i in range(1, N):                 # lower triangular (complex → 2 real)
        for j in range(i):
            result.append(float(R[i, j].real))
            result.append(float(R[i, j].imag))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Convenience: pulse-signal range gate
# ---------------------------------------------------------------------------

def pulse_range_gate_index(target_range: float,
                           f_sample: float = F_SAMPLE,
                           c_light: float = C_LIGHT) -> int:
    """Return the sample index (range bin) corresponding to `target_range`.

    two-way propagation delay τ = 2·R / c
    sample index = floor(τ · f_s)

    Args:
        target_range: Slant range [m].
        f_sample    : ADC sampling frequency [Hz].
        c_light     : Speed of light [m/s].

    Returns:
        Integer sample index.
    """
    tau = 2.0 * target_range / c_light
    return int(tau * f_sample)
