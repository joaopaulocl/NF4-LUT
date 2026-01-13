import numpy as np

from .constants import NF4_MAG


def pairwise_product_matrix(magnitudes: np.ndarray | None = None) -> np.ndarray:
    """
    Returns the outer product of the provided magnitudes.
    """
    if magnitudes is None:
        magnitudes = NF4_MAG
    result = []
    len = magnitudes.shape[0]
    for i in range(len):
        for j in range(len):
            result.append(magnitudes[i] * magnitudes[j])    

    return np.array(result).reshape((len, len))


def flatten_products(products: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the flattened product array, the sorted unique values, and the
    subset of unique positive values.
    """
    flattened = products.flatten()
    unique_vals = np.unique(flattened)
    unique_pos = np.unique(np.abs(unique_vals)) 
    return flattened, unique_vals, unique_pos

