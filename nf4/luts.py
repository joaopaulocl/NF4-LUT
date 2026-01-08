from __future__ import annotations

import numpy as np
import scipy.stats as stats

from .constants import NF4_POS_MAG


def build_5bit_acc_lut(values: np.ndarray) -> dict[str, float]:
    """Returns a 5-bit LUT for keeping precise accumulator values."""
    lut = {}
    for i, val in enumerate(values):
        lut[f"{i:05b}"] = val
    return lut


def gaussian_max_lloyd_lut(products: np.ndarray, bits: int = 3) -> dict[str, float]:
    """
    Builds a LUT by projecting a uniform Gaussian quantizer onto the product
    distribution and normalizing the selected entries to [-1, 1].
    """
    mean = products.mean()
    std = products.std()
    levels = 2 * 2**bits
    phi_inv = np.array([stats.norm.ppf(k / levels) for k in range(1, levels)])
    codebook = mean + std * phi_inv
    codebook /= np.max(np.abs(codebook))

    lut = {}
    offset = 2**bits - 1
    for i in range(2**bits):
        lut[f"{i:0{bits}b}"] = codebook[i + offset]
    return lut


def lloyd_max_empirical(samples: np.ndarray, K: int = 16, max_iters: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples)
    samples = np.sort(samples)

    c = np.quantile(samples, np.linspace(0, 1, K))

    for _ in range(max_iters):
        c_old = c.copy()
        b = np.zeros(K + 1)
        b[0] = -np.inf
        b[-1] = np.inf
        b[1:-1] = 0.5 * (c[:-1] + c[1:])

        indices = np.digitize(samples, b) - 1

        for i in range(K):
            assigned = samples[indices == i]
            if len(assigned):
                c[i] = assigned.mean()

        if np.max(np.abs(c - c_old)) < tol:
            break

    return c, b


def empirical_lloyd_lut(samples: np.ndarray, bits: int = 3) -> dict[int, float]:
    """
    Trains a Lloyd–Max quantizer on empirical samples and returns a LUT for
    the selected output levels.
    """
    K = 2 * 2**bits
    c, _ = lloyd_max_empirical(samples, K=K)

    c = c / np.max(np.abs(c))
    codebook = np.round(c, 3)

    lut = {}
    for i in range(2**bits):
        lut[i] = codebook[i + 2**bits]
    return lut


def closest_value(x: float, lut: dict) -> float:
    """Finds the LUT entry closest to x."""
    values = np.array(list(lut.values()))
    idx = np.argmin(np.abs(values - x))
    return values[idx]

def build_nf4_lut(magnitudes: np.ndarray) -> dict[int, float]:
    result = {}
    for i in range(0,magnitudes.shape[0]):
        result[i] = magnitudes[i]
    return result

def build_nf4_mul_lut(magnitudes: np.ndarray, lut: dict) -> dict[int, float]:
    """Builds a 6-bit NF4 multiplication LUT."""
    result = {}
    len = magnitudes.shape[0]
    shift = np.log2(len).astype(int) 
    for i in range(len):
        for j in range(len):
            key = (i << shift) | j
            result[key] = closest_value(magnitudes[i] * magnitudes[j], lut)
    return result


def nf4_array_multiply(a: np.ndarray, b: np.ndarray, lut: dict[int, float]) -> np.ndarray:
    """
    Elementwise NF4 × NF4 multiplication via a pre-computed LUT returning
    signed floats.
    """
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)

    sign_a = (a >> 3) & 0b1
    sign_b = (b >> 3) & 0b1

    idx_a = a & 0b111
    idx_b = b & 0b111

    out_sign = sign_a ^ sign_b
    keys = (idx_a << 3) | idx_b

    mag_prod = np.vectorize(lut.get, otypes=[float])(keys)

    return np.where(out_sign, -mag_prod, mag_prod)


def nf4_matmul(A: np.ndarray, B: np.ndarray, lut: dict[int, float]) -> np.ndarray:
    """
    Matrix multiplication C = A @ B using NF4 × NF4 multiplication via LUT.

    A: (M, K) uint8 NF4-encoded
    B: (K, N) uint8 NF4-encoded
    Returns:
        C: (M, N) float
    """
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)

    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    _, N = B.shape

    C = np.zeros((M, N), dtype=np.float64)

    # Compute C[i, j] = sum_k A[i, k] * B[k, j]
    for k in range(K):
        # Broadcast:
        # A[:, k] -> (M, 1)
        # B[k, :] -> (1, N)
        prod = nf4_array_multiply(
            A[:, k][:, None],
            B[k, :][None, :],
            lut
        )
        C += prod

    return C

def nf4_matmul_vectorized(A: np.ndarray, B: np.ndarray, lut: dict[int, float]) -> np.ndarray:
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)

    # Shapes: (M, K, 1) and (1, K, N)
    A_exp = A[:, :, None]
    B_exp = B.T[None, :, :]  # transpose so K aligns

    prod = nf4_array_multiply(A_exp, B_exp, lut)
    return prod.sum(axis=1)


def default_nf4_mul_lut(bits: int = 3) -> dict[int, float]:
    """Utility to get a default empirical LUT for NF4 multiplication."""
    return build_nf4_mul_lut(NF4_POS_MAG, empirical_lloyd_lut(NF4_POS_MAG, bits=bits))
