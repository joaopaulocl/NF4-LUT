from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nf4.constants import NF4_MAG
from nf4.luts import (
    build_mul_lut,
    build_nf4_lut,
    empirical_lloyd_lut,
    gaussian_max_lloyd_lut,
    nf4_array_multiply,
    build_nf4_mul_lut,
    nf4_matmul,
)

from nf4.products import pairwise_product_matrix, flatten_products


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(fig, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def dump_lut(lut: dict, name: str) -> None:
    path = OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        serializable = {str(k): float(v) for k, v in lut.items()}
        json.dump(serializable, fh, indent=2)
    print(f"wrote LUT {path}")


def main() -> None:
    products = pairwise_product_matrix()
    flattened, unique_vals, unique_pos = flatten_products(products)

    print("NF4 pairwise products shape:", products.shape)
    print("unique values:", unique_vals.shape[0])
    print("positive unique values:", unique_pos.shape[0])

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(products, aspect="equal")
    plt.colorbar(label="Value")
    plt.title("NF4 Pairwise Product Matrix (16×16)")
    plt.xlabel("Index")
    plt.ylabel("Index")
    save_figure(fig, "nf4_products_heatmap.png")

    fig = plt.figure()
    plt.hist(flattened, bins=30)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of NF4 Pairwise Products")
    save_figure(fig, "nf4_products_histogram.png")

    fig = plt.figure()
    plt.hist(unique_vals, bins=30)
    plt.xlabel("Unique Product Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Unique NF4 Product Values")
    save_figure(fig, "nf4_unique_histogram.png")

    lut_nf4 = build_nf4_lut(NF4_MAG)
    lut_8b = build_mul_lut(flattened, bits=8) 
    lut_gaussian = gaussian_max_lloyd_lut(flattened)
    lut_empirical = empirical_lloyd_lut(flattened, bits=3)

    #print("Empirical LUT values:", lut_empirical)
    mul_lut = build_nf4_mul_lut(NF4_MAG, lut_8b)

    mul_lut_approx = build_nf4_mul_lut(NF4_MAG, lut_empirical)

    #print(mul_lut)
    dump_lut(lut_8b, "lut_8b_accumulator.json")
    dump_lut(lut_gaussian, "lut_3b_gaussian.json")
    dump_lut(lut_empirical, "lut_3b_empirical.json")
    dump_lut(mul_lut, "nf4_mul_lut.json")
    dump_lut(mul_lut_approx, "nf4_mul_approx_lut.json")
    print("LUT approx", [float(v) for v in mul_lut_approx.values()]) 
    print("LUT accurate", [float(v) for v in lut_8b.values()]) 

    vals4b = np.arange(16, dtype=np.uint8)
    a = np.repeat(vals4b, 16)
    b = np.tile(vals4b, 16)
    products = nf4_array_multiply(a, b, mul_lut)

    print("NF4 multiplication coverage:", np.unique(products).shape[0], "values")
    a_ = np.vectorize(lut_nf4.get, otypes=[float])(a)
    b_ = np.vectorize(lut_nf4.get, otypes=[float])(b)

    #assert np.allclose(a_*b_, products, 1e-2), "NF4 element-wise mult does not match reference!"


    matmul_result_2 = nf4_matmul(
        a.reshape(16, 16), b.reshape(16, 16), mul_lut
    )
    C_ref = a_.reshape(16, 16) @ b_.reshape(16, 16)
    print(matmul_result_2.flatten()[0:10])
    print(C_ref.flatten()[0:10])

    #print("NF4 matmul result (4x4):\n", matmul_result_2)
    #print("Reference matmul result (4x4):\n", C_ref)
    assert np.allclose(matmul_result_2, C_ref, 1e-1), "NF4 matmul does not match reference!"

if __name__ == "__main__":
    main()


