"""A benchmark testing several approaches to implementing fast sparse-by-dense matrix
multiplication.

We ultimately find that a manually unrolled tiled approach provides a nice speedup!
"""

import logging
from functools import partial
from time import perf_counter

import numba
import numpy as np
import pyarrow as pa
import scipy.sparse as sp
from numpy.typing import NDArray
from tqdm.auto import tqdm

from luxical.embedder import Embedder
from luxical.misc_utils import find_project_root
from luxical.ngrams import bow_matrix_ngrams
from luxical.tokenization import (
    arrow_tokenize_texts,
    load_arrow_tokenizer_from_pretrained,
)

logger = logging.getLogger(__name__)


#
# ------ Candidate Implementations Of Sparse-By-Dense Matrix Multiplication ------
# See: https://github.com/scipy/scipy/blob/c6427325eb40c56d71579c41ba4a830c7c1fc4c8/scipy/sparse/sparsetools/csr.h#L1139
#

#
# --- Setup and Type Hinting ---
#
# This setup is common to all implementations. We define the expected
# array types for clarity.
#

A_indptr_t = NDArray[np.int64]
A_indices_t = NDArray[np.int64]
A_data_t = NDArray[np.float32]
X_t = NDArray[np.float32]
Y_t = NDArray[np.float32]


#
# --- Variation 0: Call The SciPy Builtin ---
#


def csr_builtin(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
) -> None:
    A = sp.csr_matrix(
        (A_data, A_indices, A_indptr), shape=(A_indptr.shape[0] - 1, X.shape[0])
    )
    Y += A @ X


#
# --- Variation 1: Numba Parallelized Over Rows Of A ---
#


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_baseline(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
) -> None:
    """
    Baseline Y += A @ X. Parallelized over the rows of A.
    Loop order: i (rows) -> j (non-zeros) -> k (vectors).
    """
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    # Parallelize the loop over rows of A. Each thread works on a separate
    # row of Y, so there are no race conditions.
    for i in numba.prange(n_row):
        # Iterate over the non-zero elements in the i-th row of A.
        for jj in range(A_indptr[i], A_indptr[i + 1]):
            j = A_indices[jj]  # Column index in A, row index in X
            a = A_data[jj]  # Non-zero value A[i, j]

            # --- Performance Analysis ---
            # This inner loop is an AXPY operation: Y[i,:] += a * X[j,:].
            # Access to Y[i,:] is sequential and cache-friendly.
            # However, the `j` index changes with each iteration of the `jj` loop,
            # meaning we jump to a new row of X each time. This can cause
            # cache misses if the rows of X needed for a given row of A are
            # not all in the cache. This is the primary bottleneck we will
            # try to address.
            for k in range(n_vecs):
                Y[i, k] += a * X[j, k]


#
# --- Variation 2: Tiled with Tuned Block Size ---
#
# This version introduces tiling (or blocking) on the innermost loop over `k`.
# The goal is to improve the temporal locality of the accesses to matrix X.
# This is often the most effective optimization for this type of problem.
#


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_tiled(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
    block_size: int = 4,  # Tunable parameter
) -> None:
    """
    Tiled version of Y += A @ X. Improves cache reuse of X.
    Loop order: k_block -> i (rows) -> j (non-zeros) -> k (vectors).
    """
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    for k_start in range(0, n_vecs, block_size):
        k_end = min(k_start + block_size, n_vecs)
        for i in numba.prange(n_row):
            for jj in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[jj]
                a = A_data[jj]
                for k in range(k_start, k_end):
                    Y[i, k] += a * X[j, k]


#
# --- Variation 3: Tiled with Block Accumulator ---
#
# This variation builds on the tiled approach by using a small, temporary
# array for each block of the output row Y[i, :]. This accumulator is
# likely to be allocated on the stack and remain in the L1 cache, reducing
# read-modify-write traffic to the main Y matrix.
#


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_tiled_acc(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
    block_size: int = 4,
) -> None:
    """
    Tiled version using a temporary accumulator for each Y row-block.
    """
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    for k_start in range(0, n_vecs, block_size):
        k_end = min(k_start + block_size, n_vecs)
        current_block_size = k_end - k_start

        for i in numba.prange(n_row):
            # Temporary accumulator for the Y[i, k_start:k_end] block.
            y_block_acc = np.zeros(current_block_size, dtype=np.float32)

            # Accumulate results for the block into the temporary array.
            for jj in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[jj]
                a = A_data[jj]
                for k_idx in range(current_block_size):
                    k = k_start + k_idx
                    y_block_acc[k_idx] += a * X[j, k]

            # Add the accumulated block back to the main Y matrix once.
            for k_idx in range(current_block_size):
                Y[i, k_start + k_idx] += y_block_acc[k_idx]


#
# --- Variation 4: Tiled with Manual Unrolling ---
#
# This is a specialized and more aggressive optimization. It manually unrolls
# the innermost loop for a fixed block size. Instead of a temporary array,
# it uses four separate scalar variables, which gives a strong hint to the
# compiler to keep them in CPU registers. This eliminates array indexing
# overhead in the inner loop and creates a very clear pattern for SIMD vectorization.
#


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_tiled_unrolled_4(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
) -> None:
    """
    Tiled version with manual loop unrolling for block_size=4.
    """
    BLOCK_SIZE = 4
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    # Process all vectors in blocks. The last block might be smaller.
    for k_start in range(0, n_vecs, BLOCK_SIZE):
        k_end = min(k_start + BLOCK_SIZE, n_vecs)

        # Use the fast, unrolled path if the block is full-sized.
        if (k_end - k_start) == BLOCK_SIZE:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    # Add directly to Y to maintain numerical stability.
                    # This avoids using temporary accumulators.
                    Y[i, k_start + 0] += a * X[j, k_start + 0]
                    Y[i, k_start + 1] += a * X[j, k_start + 1]
                    Y[i, k_start + 2] += a * X[j, k_start + 2]
                    Y[i, k_start + 3] += a * X[j, k_start + 3]
        # Use a generic tiled loop for the smaller remainder block.
        else:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    for k in range(k_start, k_end):
                        Y[i, k] += a * X[j, k]


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_tiled_unrolled_8(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
) -> None:
    """
    Tiled version with manual loop unrolling for block_size=8.
    """
    BLOCK_SIZE = 8
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    # Process all vectors in blocks. The last block might be smaller.
    for k_start in range(0, n_vecs, BLOCK_SIZE):
        k_end = min(k_start + BLOCK_SIZE, n_vecs)

        # Use the fast, unrolled path if the block is full-sized.
        if (k_end - k_start) == BLOCK_SIZE:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    # Add directly to Y to maintain numerical stability.
                    # Unrolled for a block of 8.
                    Y[i, k_start + 0] += a * X[j, k_start + 0]
                    Y[i, k_start + 1] += a * X[j, k_start + 1]
                    Y[i, k_start + 2] += a * X[j, k_start + 2]
                    Y[i, k_start + 3] += a * X[j, k_start + 3]
                    Y[i, k_start + 4] += a * X[j, k_start + 4]
                    Y[i, k_start + 5] += a * X[j, k_start + 5]
                    Y[i, k_start + 6] += a * X[j, k_start + 6]
                    Y[i, k_start + 7] += a * X[j, k_start + 7]
        # Use a generic tiled loop for the smaller remainder block.
        else:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    for k in range(k_start, k_end):
                        Y[i, k] += a * X[j, k]


#
# --- Variation 5: Tiled Unrolled with Accumulators (Block Size 8) ---
#
# This version combines unrolling with local scalar accumulators. This is
# often the fastest approach as it encourages the compiler to keep the hot
# variables (y0-y7) in CPU registers, minimizing memory access to the main
# Y array within the innermost loop.
#
# NOTE on AVX512: For a true AVX512 CPU, unrolling to a block size of 16
# would be an even better match for the hardware's 512-bit vector units.
#
# NOTE on Numerics: This method changes the order of floating-point additions
# compared to adding directly to Y, which can lead to small numerical
# differences. For most applications this is acceptable for the performance gain.
#


@numba.njit(nogil=True, parallel=True, error_model="numpy")
def csr_matvecs_tiled_unrolled_acc(
    A_indptr: A_indptr_t,
    A_indices: A_indices_t,
    A_data: A_data_t,
    X: X_t,
    Y: Y_t,
) -> None:
    """
    Tiled version with unrolling and accumulators for block_size=8.
    """
    BLOCK_SIZE = 8
    n_row = A_indptr.shape[0] - 1
    if n_row == 0:
        return
    n_vecs = X.shape[1]

    # Process all vectors in blocks. The last block might be smaller.
    for k_start in range(0, n_vecs, BLOCK_SIZE):
        k_end = min(k_start + BLOCK_SIZE, n_vecs)

        # Use the fast, unrolled path if the block is full-sized.
        if (k_end - k_start) == BLOCK_SIZE:
            for i in numba.prange(n_row):
                # Scalar accumulators are very likely to be kept in registers.
                y0, y1, y2, y3 = 0.0, 0.0, 0.0, 0.0
                y4, y5, y6, y7 = 0.0, 0.0, 0.0, 0.0

                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    # Accumulate into local variables.
                    y0 += a * X[j, k_start + 0]
                    y1 += a * X[j, k_start + 1]
                    y2 += a * X[j, k_start + 2]
                    y3 += a * X[j, k_start + 3]
                    y4 += a * X[j, k_start + 4]
                    y5 += a * X[j, k_start + 5]
                    y6 += a * X[j, k_start + 6]
                    y7 += a * X[j, k_start + 7]

                # Write back to main memory once per row-block.
                Y[i, k_start + 0] += y0
                Y[i, k_start + 1] += y1
                Y[i, k_start + 2] += y2
                Y[i, k_start + 3] += y3
                Y[i, k_start + 4] += y4
                Y[i, k_start + 5] += y5
                Y[i, k_start + 6] += y6
                Y[i, k_start + 7] += y7
        # Use a generic tiled loop for the smaller remainder block.
        else:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    for k in range(k_start, k_end):
                        Y[i, k] += a * X[j, k]


#
# ------ END Candidate Implementations Of Sparse-By-Dense Matrix Multiplication ------
#


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting benchmarking...")

    # Find the data directory.
    data_dir = find_project_root() / "data"

    # Initialize the Arrow Tokenizer from a pretrained tokenizer state.
    logger.info("Loading tokenizer...")
    tokenizer_id = "google-bert/bert-base-uncased"
    arrow_tokenizer = load_arrow_tokenizer_from_pretrained(tokenizer_id)

    # Load the embedder.
    logger.info("Loading embedder...")
    embedder_path = data_dir / "full_embedder_state_v2.json"
    embedder_state = Embedder.from_str(embedder_path.read_text())

    # Load the data.
    logger.info("Loading data...")
    with pa.ipc.open_file(data_dir / "train.arrow") as reader:
        table_train = reader.read_all()
    with pa.ipc.open_file(data_dir / "test.arrow") as reader:
        table_test = reader.read_all()
    text_train = table_train["text"].to_pylist()
    text_test = table_test["text"].to_pylist()
    y_train = table_train["y"].to_numpy()
    y_test = table_test["y"].to_numpy()

    # Tokenize all the data.
    logger.info("Tokenizing data...")
    tokens_train = arrow_tokenize_texts(text_train, arrow_tokenizer)
    tokens_test = arrow_tokenize_texts(text_test, arrow_tokenizer)

    # BoW all the data.
    logger.info("BoW'ing data...")
    bow_train = bow_matrix_ngrams(
        tokens_train.to_numpy(),
        max_ngram_length=embedder_state.max_ngram_length,
        ngram_hash_to_idx=embedder_state.ngram_hash_to_ngram_idx,
    )
    bow_test = bow_matrix_ngrams(
        tokens_test.to_numpy(),
        max_ngram_length=embedder_state.max_ngram_length,
        ngram_hash_to_idx=embedder_state.ngram_hash_to_ngram_idx,
    )

    logger.info("Benchmarking...")
    total_size = 1024 * 16  # Total number of rows to benchmark on.
    B = embedder_state.bow_to_dense_embedder.layers[0].T
    expected = bow_train[:total_size].astype(np.float32) @ B

    def eval_fn(fn, batch_size: int, name: str | None = None):
        out = np.zeros((total_size, B.shape[1]), dtype=np.float32)
        with tqdm(total=total_size, desc=name, unit="seq", unit_scale=True) as pbar:
            start = perf_counter()
            for i in range(0, total_size, batch_size):
                batch = bow_train[i : i + batch_size].astype(np.float32)
                fn(
                    batch.indptr,
                    batch.indices,
                    batch.data,
                    B,
                    out[i : i + batch_size],
                )
                pbar.update(batch.shape[0])
            elapsed = perf_counter() - start
        name_str = f"{name:>12} " if name else ""
        print(f"{name_str}time: {elapsed:.3f}s")
        # Test closeness.
        if not np.isclose(out, expected, rtol=1e-4, atol=1e-4).all():
            print("WARNING: Output is not close to the baseline.")

    for batch_size in [256, 1024, 4096, 16384]:
        print(f"Batch size: {batch_size}")
        for name, fn in [
            ("builtin", csr_builtin),
            ("baseline", csr_matvecs_baseline),
            ("tiled_4", partial(csr_matvecs_tiled, block_size=4)),
            ("tiled_4_acc", partial(csr_matvecs_tiled_acc, block_size=4)),
            ("tiled_4_unrolled", csr_matvecs_tiled_unrolled_4),
            ("tiled_8", partial(csr_matvecs_tiled, block_size=8)),
            ("tiled_8_acc", partial(csr_matvecs_tiled_acc, block_size=8)),
            ("tiled_8_unrolled", csr_matvecs_tiled_unrolled_8),
            ("tiled_8_unrolled_acc", csr_matvecs_tiled_unrolled_acc),
            ("tiled_16", partial(csr_matvecs_tiled, block_size=16)),
        ]:
            eval_fn(fn, batch_size=batch_size, name=name)
            print()
        print("=" * 80)
        print("\n" * 3)
