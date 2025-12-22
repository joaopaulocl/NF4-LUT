import numpy as np

from .constants import NF4_MAG


def pairwise_product_matrix(magnitudes: np.ndarray | None = None) -> np.ndarray:
    """
    Returns the outer product of the provided magnitudes.
    """
    if magnitudes is None:
        magnitudes = NF4_MAG
    return np.outer(magnitudes, magnitudes)


def flatten_products(products: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the flattened product array, the sorted unique values, and the
    subset of unique positive values.
    """
    flattened = products.flatten()
    unique_vals = np.unique(flattened)
    unique_pos = unique_vals[unique_vals >= 0]
    return flattened, unique_vals, unique_pos

