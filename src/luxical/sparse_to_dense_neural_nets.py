"""
This submodule implements feedforward neural networks that project from sparse
representations (like BoW or TF-IDF) to dense representations (embedding vectors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from luxical.csr_matrix_utils import csr_matvecs_tiled_unrolled_8, csr_matvecs_torch
from luxical.misc_utils import normalize_inplace, relu_inplace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseToDenseEmbedder:
    layers: list[NDArray[np.float32]]

    @classmethod
    def create(
        cls,
        dims: Sequence[int],
        dtype: type[np.floating] = np.float32,
        seed: int = 0,
    ) -> SparseToDenseEmbedder:
        assert len(dims) > 1
        logger.info(
            f"Initializing sparse to dense embedder with dims {dims} and dtype {dtype}."
        )
        rng = np.random.default_rng(seed)
        layers: list[NDArray[np.floating]] = []
        for i in range(len(dims) - 1):
            kaiming_uniform_scale = np.sqrt(6 / dims[i])
            layer = rng.uniform(
                -kaiming_uniform_scale,
                kaiming_uniform_scale,
                size=(dims[i + 1], dims[i]),
            ).astype(dtype)
            layers.append(layer)
        return cls(layers)

    @property
    def output_dim(self) -> int:
        return self.layers[-1].shape[0]

    @property
    def input_dim(self) -> int:
        return self.layers[0].shape[1]

    @property
    def dims(self) -> Sequence[int]:
        return [layer.shape[1] for layer in self.layers]

    def to_torch(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> SparseToDenseEmbedderTorchModule:
        return SparseToDenseEmbedderTorchModule(
            layers=[torch.tensor(x, device=device, dtype=dtype) for x in self.layers],
        )

    @classmethod
    def from_torch(
        cls, module: SparseToDenseEmbedderTorchModule
    ) -> SparseToDenseEmbedder:
        return cls(
            layers=[layer.detach().float().cpu().numpy() for layer in module.layers],
        )

    def __call__(
        self,
        bow_csr_matrix: csr_matrix,
        batch_size: int = 4096,
        progress_bar: bool = False,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Embed BoW-encoded text into a dense vector space.

        Internally batches data to avoid memory overhead. Optionally provides a
        `tqdm` progress bar.
        """
        assert bow_csr_matrix.ndim == 2
        assert bow_csr_matrix.shape[1] == self.layers[0].shape[1]  # type: ignore[index]
        assert len(self.layers) > 0
        num_rows = bow_csr_matrix.shape[0]  # type: ignore[index]

        # Allocate the final output and chunks for intermediate activations.
        chunk_activations = [
            np.zeros((batch_size, layer.shape[0]), dtype=np.float32)
            for layer in self.layers[:-1]
        ]
        if out is not None:
            assert out.shape == (num_rows, self.output_dim)
            assert out.dtype == np.float32
        else:
            out = np.empty((num_rows, self.output_dim), dtype=np.float32)

        # Go batch by batch.
        with (
            tqdm(
                total=num_rows,
                disable=not progress_bar,
                desc="BoW to dense",
                unit="row",
                unit_scale=True,
            ) as pbar,
            np.testing.suppress_warnings() as sup,
        ):
            # Suppress matmul warnings, which seem to be hair-trigger on some systems.
            sup.filter(RuntimeWarning, message=r".*encountered in matmul.*")
            for start in range(0, num_rows, batch_size):
                # Take the next batch of rows as a floating point matrix.
                end = min(start + batch_size, num_rows)
                batch_size = end - start
                bow_batch = bow_csr_matrix[start:end].astype(np.float32)

                # Sparse-by-dense matrix multiplication of the first layer.
                chunk_activations[0][:] = 0
                csr_matvecs_tiled_unrolled_8(
                    bow_batch.indptr,
                    bow_batch.indices,
                    bow_batch.data,
                    self.layers[0].T,
                    chunk_activations[0][:batch_size],
                )

                # ReLU and normalize.
                relu_inplace(chunk_activations[0][:batch_size])
                normalize_inplace(chunk_activations[0][:batch_size])

                # Do the hidden layers.
                for i in range(1, len(self.layers) - 1):
                    chunk_activations[i][:] = 0
                    np.matmul(
                        chunk_activations[i - 1][:batch_size],
                        self.layers[i].T,
                        out=chunk_activations[i][:batch_size],
                    )
                    relu_inplace(chunk_activations[i][:batch_size])
                    normalize_inplace(chunk_activations[i][:batch_size])

                # Do the final layer.
                np.matmul(
                    chunk_activations[-1][:batch_size],
                    self.layers[-1].T,
                    out=out[start:end],
                )
                normalize_inplace(out[start:end])
            pbar.update(batch_size)

        return out


class SparseToDenseEmbedderTorchModule(nn.Module):
    """Torch module implementing the same functionality as `SparseToDenseEmbedder`.

    This is useful for training the model using automatic differentiation with GPUs.
    """

    def __init__(self, layers: Sequence[torch.Tensor]):
        super().__init__()
        assert len(layers) > 0
        self.layers = nn.ParameterList(nn.Parameter(layer) for layer in layers)

    @property
    def output_dim(self) -> int:
        return self.layers[-1].shape[0]

    @property
    def input_dim(self) -> int:
        return self.layers[0].shape[1]

    @classmethod
    def create(
        cls,
        dims: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> SparseToDenseEmbedderTorchModule:
        assert len(dims) > 1
        rng = torch.Generator(device=device).manual_seed(seed)
        layers = []
        for i in range(len(dims) - 1):
            layer = torch.empty(dims[i + 1], dims[i], dtype=dtype, device=device)
            nn.init.kaiming_uniform_(layer, generator=rng, nonlinearity="relu")
            layers.append(layer)
        return cls(layers)

    def forward(self, bow_csr_tensor: torch.Tensor) -> torch.Tensor:
        assert bow_csr_tensor.ndim == 2
        assert bow_csr_tensor.shape[1] == self.layers[0].shape[1]
        assert bow_csr_tensor.device == self.layers[0].device

        # Project our sparse high-dimensional bag of words features to hidden space.
        # If we're on CPU, use our own custom fast sparse-by-dense matmul function.
        if self.layers[0].is_cpu:
            hidden = csr_matvecs_torch(bow_csr_tensor, self.layers[0].T)
        else:
            hidden = bow_csr_tensor @ self.layers[0].T
        hidden = F.normalize(torch.relu(hidden), dim=1)

        # Run all the hidden-to-hidden transformations.
        for layer in self.layers[1:-1]:
            hidden = F.normalize(torch.relu(hidden @ layer.T), dim=1)

        # Project our final hidden activations to the output vector (no nonlinearity).
        out = F.normalize(hidden @ self.layers[-1].T, dim=-1)
        return out
