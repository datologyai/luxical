from pathlib import Path

import numba
import numpy as np
import pyarrow as pa
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def csr_matrix_to_npz(matrix: csr_matrix, npz_path: str | Path) -> None:
    """Saves a SciPy CSR matrix to a .npz file."""
    np.savez_compressed(
        npz_path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,  # type: ignore[arg-type]
    )


def csr_matrix_from_npz(npz_path: str | Path) -> csr_matrix:
    """Loads a SciPy CSR matrix from a .npz file."""
    loader = np.load(npz_path)
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=loader["shape"],  # type: ignore[arg-type]
    )


def csr_matrix_to_arrow_batch(
    matrix: csr_matrix,
    indices_column: str = "indices",
    values_column: str = "values",
    indices_range_column: str = "indices_range",
) -> pa.RecordBatch:
    """Converts a SciPy CSR matrix to an Arrow record batch."""
    values = pa.LargeListArray.from_arrays(offsets=matrix.indptr, values=matrix.data)
    indices = pa.LargeListArray.from_arrays(
        offsets=matrix.indptr, values=matrix.indices
    )
    record_batch = pa.record_batch(
        {
            indices_column: indices,
            values_column: values,
            indices_range_column: pa.repeat(matrix.shape[1], matrix.shape[0]),  # type: ignore[index]
        }
    )
    return record_batch


def csr_matrix_from_arrow_batch(
    table_or_batch: pa.Table | pa.RecordBatch,
    indices_column: str = "indices",
    values_column: str = "values",
    indices_range_column: str = "indices_range",
) -> csr_matrix:
    """Convert an Arrow table or record batch into a SciPy CSR matrix."""
    # Combine chunks if necessary to get pa.Array not pa.ChunkedArray objects.
    indices = table_or_batch[indices_column]
    values = table_or_batch[values_column]
    indices_range = table_or_batch[indices_range_column]
    if isinstance(indices, pa.ChunkedArray):
        indices = indices.combine_chunks()
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(indices_range, pa.ChunkedArray):
        indices_range = indices_range.combine_chunks()

    # Pull out the offsets.
    indptr = indices.offsets.to_numpy()
    assert np.allclose(indptr, values.offsets.to_numpy())

    # Ensure indices range is constant across all rows.
    indices_range_value = indices_range[0].as_py()
    assert np.allclose(indices_range, indices_range_value)

    # Construct the SciPy CSR matrix.
    matrix = csr_matrix(
        (values.values.to_numpy(), indices.values.to_numpy(), indptr),
        shape=(len(table_or_batch), indices_range_value),
    )
    return matrix


def csr_matrix_from_torch(tensor: torch.Tensor) -> csr_matrix:
    """Converts a PyTorch sparse CSR tensor to a SciPy CSR matrix."""
    if not tensor.is_sparse_csr:
        raise ValueError("Input tensor must be in sparse CSR format.")
    crow_indices = tensor.crow_indices().cpu().numpy()
    col_indices = tensor.col_indices().cpu().numpy()
    values = tensor.values().cpu().numpy()
    size = tensor.shape
    return sp.csr_matrix((values, col_indices, crow_indices), shape=size)


def csr_matrix_to_torch(sparse_matrix: csr_matrix) -> torch.Tensor:
    """Converts a SciPy CSR matrix to a PyTorch sparse CSR tensor."""
    if not sp.isspmatrix_csr(sparse_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix.")
    return torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(sparse_matrix.indptr),
        col_indices=torch.from_numpy(sparse_matrix.indices),
        values=torch.from_numpy(sparse_matrix.data),
        size=sparse_matrix.shape,
    )


# Fast parallelized sparse-by-dense matrix multiplication.
#
# Here are several different signatures of inputs that we can precompile for.
#
# The dense matrix is expected to be in Fortran-order, e.g. pre-transposed, but we allow
# it to be readonly or not (helpful when loading a matrix from serialized form).
csr_matvecs_tiled_unrolled_8_type_signatures = [
    numba.void(
        numba.int32[:],
        numba.int32[:],
        numba.float32[:],
        numba.types.Array(numba.float32, 2, "F", readonly=False),
        numba.float32[:, :],
    ),
    numba.void(
        numba.int32[:],
        numba.int32[:],
        numba.float32[:],
        numba.types.Array(numba.float32, 2, "F", readonly=True),
        numba.float32[:, :],
    ),
    numba.void(
        numba.int64[:],
        numba.int64[:],
        numba.float32[:],
        numba.types.Array(numba.float32, 2, "F", readonly=False),
        numba.float32[:, :],
    ),
    numba.void(
        numba.int64[:],
        numba.int64[:],
        numba.float32[:],
        numba.types.Array(numba.float32, 2, "F", readonly=True),
        numba.float32[:, :],
    ),
]


@numba.njit(
    # csr_matvecs_tiled_unrolled_8_type_signatures,  # Uncomment to precompile.
    nogil=True,
    parallel=True,
    error_model="numpy",
)
def csr_matvecs_tiled_unrolled_8(
    A_indptr: NDArray[np.int64],
    A_indices: NDArray[np.int64],
    A_data: NDArray[np.float32],
    B_dense: NDArray[np.float32],
    Y_out_dense: NDArray[np.float32],
) -> None:
    """Fast multithreaded sparse-by-dense matrix multiplication.

    Computes Y_out_dense += A @ B_dense

    Uses tiling and manual loop unrolling with block_size 8 for high performance.
    """
    BLOCK_SIZE = 8
    n_row = A_indptr.shape[0] - 1
    n_vecs = B_dense.shape[1]
    if n_row == 0:
        return

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
                    Y_out_dense[i, k_start + 0] += a * B_dense[j, k_start + 0]
                    Y_out_dense[i, k_start + 1] += a * B_dense[j, k_start + 1]
                    Y_out_dense[i, k_start + 2] += a * B_dense[j, k_start + 2]
                    Y_out_dense[i, k_start + 3] += a * B_dense[j, k_start + 3]
                    Y_out_dense[i, k_start + 4] += a * B_dense[j, k_start + 4]
                    Y_out_dense[i, k_start + 5] += a * B_dense[j, k_start + 5]
                    Y_out_dense[i, k_start + 6] += a * B_dense[j, k_start + 6]
                    Y_out_dense[i, k_start + 7] += a * B_dense[j, k_start + 7]
        # Use a generic tiled loop for the smaller remainder block.
        else:
            for i in numba.prange(n_row):
                for jj in range(A_indptr[i], A_indptr[i + 1]):
                    j = A_indices[jj]
                    a = A_data[jj]
                    for k in range(k_start, k_end):
                        Y_out_dense[i, k] += a * B_dense[j, k]


class _CsrMatVecs(torch.autograd.Function):
    """
    Custom autograd Function to wrap our fast sparse-by-dense matmul implementation.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.layout == torch.sparse_csr  # Sparse CSR layout.
        assert b.layout == torch.strided  # Dense layout.
        # Save tensors for the backward pass
        ctx.save_for_backward(a, b)
        a_scipy_csr = csr_matrix_from_torch(a)
        b_np_ndarray = b.cpu().numpy()
        result = np.zeros(shape=(a.shape[0], b.shape[1]), dtype=b_np_ndarray.dtype)
        csr_matvecs_tiled_unrolled_8(
            A_indptr=a_scipy_csr.indptr,  # type: ignore[arg-type]
            A_indices=a_scipy_csr.indices,  # type: ignore[arg-type]
            A_data=a_scipy_csr.data,  # type: ignore[arg-type]
            B_dense=b_np_ndarray,
            Y_out_dense=result,
        )
        result_torch = torch.from_numpy(result)
        return result_torch

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        The gradient of a matrix product Y = A @ B is given by:
        d_loss/d_A = d_loss/d_Y @ B.T
        d_loss/d_B = A.T @ d_loss/d_Y

        Args:
            ctx: Context object with saved tensors.
            grad_output (torch.Tensor): The gradient of the loss with respect to the output of this function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to the input tensors a and b.
        """
        a, b = ctx.saved_tensors
        grad_output = grad_outputs[0]
        grad_a = grad_b = None

        # Calculate gradient with respect to a, if needed.
        # NOTE: `grad_a` will not be sparse, but `grad_a.to_sparse_csr * a.to(torch.bool)` should
        # sparsify it.
        if ctx.needs_input_grad[0]:
            grad_a = grad_output @ b.T

        # Calculate gradient with respect to b, if needed.
        if ctx.needs_input_grad[1]:
            a_transpose_scipy_csr = csr_matrix_from_torch(a).T.tocsr()
            grad_output_np_ndarray = grad_output.cpu().numpy()
            grad_b_np_ndarray = np.zeros(
                shape=b.size(), dtype=grad_output_np_ndarray.dtype
            )
            csr_matvecs_tiled_unrolled_8(
                A_indptr=a_transpose_scipy_csr.indptr,  # type: ignore[arg-type]
                A_indices=a_transpose_scipy_csr.indices,  # type: ignore[arg-type]
                A_data=a_transpose_scipy_csr.data,  # type: ignore[arg-type]
                B_dense=grad_output_np_ndarray,
                Y_out_dense=grad_b_np_ndarray,
            )
            grad_b = torch.from_numpy(grad_b_np_ndarray)

        return grad_a, grad_b


# Create a nicely-type-annotated function `csr_matvecs_torch` from our custom autograd
# function implementation.
class _CsrMatVecsWrapper:
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _CsrMatVecs.apply(a, b)  # type: ignore[no-any-return]


csr_matvecs_torch = _CsrMatVecsWrapper()
