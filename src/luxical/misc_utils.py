"""Miscellaneous utilities for the Luxical project."""

import logging
from pathlib import Path
from typing import Any, Iterable, cast

import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def find_project_root(marker: str = ".git") -> Path:
    """Finds the project root directory by searching for a marker file or directory."""
    current_path = Path.cwd().resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root from {current_path}")


@numba.njit(error_model="numpy", parallel=True)
def fast_8bit_uniform_scalar_quantize(
    emb_matrix: NDArray[np.float32], limit: float
) -> NDArray[np.uint8]:
    num_row, num_col = emb_matrix.shape
    assert limit > 0
    out = np.empty((num_row, num_col), dtype=np.uint8)
    bin_width = 2 * limit / 255
    for i in numba.prange(num_row):
        for j in range(num_col):
            out[i, j] = round(
                max(0, min(2 * limit, limit + emb_matrix[i, j])) / bin_width
            )
    return out


def dequantize_8bit_uniform_scalar_quantized(
    emb_quantized: NDArray[np.uint8], limit: float
) -> NDArray[np.float32]:
    assert limit > 0
    bin_width = 2 * limit / 255
    return -np.float32(limit) + np.float32(bin_width) * emb_quantized


def stream_id_text_pairs_from_parquet(
    in_paths: list[Path],
    chunk_size: int,
    id_column_name: str,
    text_column_name: str,
) -> Iterable[tuple[pa.StringArray, pa.StringArray]]:
    """Streams (id, text) pairs from a list of Parquet files."""
    logger.info(f"Streaming from {len(in_paths)} Parquet files.")
    for in_path in in_paths:
        logger.info(f"Processing file: {in_path}")
        with pq.ParquetFile(in_path) as reader:
            for chunk in reader.iter_batches(batch_size=chunk_size):
                yield chunk[id_column_name], chunk[text_column_name]


def numpy_ndarray_to_pyarrow_fixed_size_list_array(
    ndarray: NDArray[Any],
) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        np.ascontiguousarray(ndarray).ravel(),
        list_size=ndarray.shape[1],
    )


def pyarrow_fixed_size_list_array_to_numpy_ndarray(
    pa_array: pa.ChunkedArray | pa.FixedSizeListArray,
) -> NDArray[Any]:
    embed_dim = len(pa_array[0])
    if hasattr(pa_array, "combine_chunks"):
        pa_array = pa_array.combine_chunks()
    res = pa_array.flatten().to_numpy().reshape(-1, embed_dim)
    return cast(NDArray[Any], res)


def gini_score(y_true: NDArray[np.bool_], y_score: NDArray[np.float32]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError(
            "Sklearn is not installed, please install `luxical[sklearn] to use this function."
        )

    auc = float(roc_auc_score(y_true, y_score))
    return 2 * auc - 1


def normalize_inplace(matrix: NDArray[np.float32]) -> None:
    assert matrix.ndim == 2
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.divide(matrix, norms, out=matrix)


def relu_inplace(matrix: NDArray[np.float32]) -> None:
    assert matrix.ndim == 2
    np.maximum(matrix, 0, out=matrix)
