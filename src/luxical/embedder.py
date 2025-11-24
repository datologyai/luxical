from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar, cast

import numba
import numpy as np
import pyarrow as pa
from numba.typed import Dict as NumbaTypedDict
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from luxical.ngrams import (
    SpaceSavingNgramSummary,
    bow_matrix_ngrams,
    build_ngram_hash_to_idx_map,
)
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder
from luxical.tokenization import ArrowTokenizer

logger = logging.getLogger(__name__)

# Embedder serialization format version
EMBEDDER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Embedder:
    tokenizer: ArrowTokenizer
    recognized_ngrams: NDArray[np.int64]
    ngram_hash_to_ngram_idx: NumbaTypedDict[np.int64, np.uint32]
    idf_values: NDArray[np.float32]
    bow_to_dense_embedder: SparseToDenseEmbedder

    @property
    def max_ngram_length(self) -> int:
        return self.recognized_ngrams.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.bow_to_dense_embedder.output_dim

    def save(self, path: str | Path) -> None:
        path = Path(path).resolve()

        # Convert the ngram hash to ngram index map to a pair of numpy arrays.
        ngram_idx_keys, ngram_idx_values = _unpack_int_dict(
            self.ngram_hash_to_ngram_idx
        )

        # Name the individual neural network layers.
        neural_layers = self.bow_to_dense_embedder.layers
        layers_dict = {f"nn_layer_{i}": layer for i, layer in enumerate(neural_layers)}

        # Store the tokenizer state JSON string as a numpy array of bytes.
        tokenizer_bytes_arr = np.frombuffer(
            self.tokenizer.to_str().encode("utf-8"), dtype=np.uint8
        )

        # Serialize the embedder state to a numpy file.
        np.savez(
            path,
            tokenizer=tokenizer_bytes_arr,
            version=np.int64(EMBEDDER_FORMAT_VERSION),
            recognized_ngrams=self.recognized_ngrams,
            ngram_hash_to_ngram_idx_keys=ngram_idx_keys,
            ngram_hash_to_ngram_idx_values=ngram_idx_values,
            idf_values=self.idf_values,
            num_nn_layers=np.array([len(neural_layers)]),
            **layers_dict,
            allow_pickle=False,
        )

    @classmethod
    def load(cls, path: str | Path) -> Embedder:
        path = Path(path).resolve()

        with np.load(path, allow_pickle=False) as npzfile:
            # Ensure supported version.
            version = npzfile["version"].item()
            if version != EMBEDDER_FORMAT_VERSION:
                raise NotImplementedError(
                    f"Unsupported embedder format version {version}. "
                    f"This code supports version {EMBEDDER_FORMAT_VERSION}."
                )

            # Deserialize the tokenizer.
            tokenizer_bytes_arr = npzfile["tokenizer"]
            tokenizer_str = tokenizer_bytes_arr.tobytes().decode("utf-8")
            tokenizer = ArrowTokenizer(tokenizer_str)

            # Deserialize the recognized ngrams.
            recognized_ngrams = npzfile["recognized_ngrams"]

            # Deserialize the ngram hash to ngram index map.
            ngram_idx_keys = npzfile["ngram_hash_to_ngram_idx_keys"]
            ngram_idx_values = npzfile["ngram_hash_to_ngram_idx_values"]
            ngram_hash_to_ngram_idx = _pack_int_dict(ngram_idx_keys, ngram_idx_values)

            # Deserialize the IDF values.
            idf_values = npzfile["idf_values"]

            # Deserialize the neural network layers into a `SparseToDenseEmbedder`.
            num_nn_layers = npzfile["num_nn_layers"][0]
            nn_layers = [npzfile[f"nn_layer_{i}"] for i in range(num_nn_layers)]
            bow_to_dense_embedder = SparseToDenseEmbedder(layers=nn_layers)

            return cls(
                tokenizer=tokenizer,
                recognized_ngrams=recognized_ngrams,
                ngram_hash_to_ngram_idx=ngram_hash_to_ngram_idx,
                bow_to_dense_embedder=bow_to_dense_embedder,
                idf_values=idf_values,
            )

    def tokenize(
        self, texts: pa.StringArray | Sequence[str]
    ) -> Sequence[NDArray[np.uint32]]:
        token_pa = self.tokenizer.tokenize(pa.array(texts), add_special_tokens=False)
        return token_pa.to_numpy(zero_copy_only=False)

    def bow_from_tokens(
        self, tokenized_docs: Iterable[NDArray[np.uint32]], progress_bar: bool = False
    ) -> csr_matrix:
        return bow_matrix_ngrams(
            tokenized_docs=tokenized_docs,
            max_ngram_length=self.max_ngram_length,
            ngram_hash_to_idx=self.ngram_hash_to_ngram_idx,
            progress_bar=progress_bar,
        )

    def bow_from_texts(self, texts: Sequence[str]) -> csr_matrix:
        tokenized_docs = self.tokenize(texts)
        return self.bow_from_tokens(tokenized_docs, progress_bar=False)

    def tfidf_from_bow(self, bow: csr_matrix) -> csr_matrix:
        tfidf_data = np.empty_like(bow.data, dtype=np.float32)
        _fast_tfidf_from_bow(
            tfidf_data=tfidf_data,
            bow_data=cast(NDArray[np.integer[Any]], bow.data),
            bow_indices=bow.indices,
            bow_indptr=bow.indptr,
            idf_values=self.idf_values,
        )
        return csr_matrix((tfidf_data, bow.indices, bow.indptr), shape=bow.shape)

    def replace_sparse_to_dense_embedder(
        self, new_embedder: SparseToDenseEmbedder
    ) -> Embedder:
        """Replaces the neural network parameters as a copy operation."""
        return Embedder(
            tokenizer=self.tokenizer,
            recognized_ngrams=self.recognized_ngrams,
            ngram_hash_to_ngram_idx=self.ngram_hash_to_ngram_idx,
            idf_values=self.idf_values,
            bow_to_dense_embedder=new_embedder,
        )

    def __call__(
        self,
        texts: pa.StringArray | Sequence[str],
        batch_size: int = 4096,
        progress_bars: bool = False,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Embed a sequence of texts into a dense vector space.

        Internally batches data to avoid memory overhead. Optionally provides a
        `tqdm` progress bar.
        """
        n = len(texts)
        out = np.empty((n, self.embedding_dim), dtype=np.float32)
        with (
            tqdm(
                total=n,
                desc="Embedding",
                unit="text",
                unit_scale=True,
                disable=not progress_bars,
            ) as pbar,
            tqdm(
                unit="token", unit_scale=True, position=1, disable=not progress_bars
            ) as pbar_tokens,
        ):
            for start in range(0, n, batch_size):
                # Take the next batch of input texts.
                end = min(start + batch_size, n)
                batch_size = end - start
                batch_text = texts[start:end]

                # Tokenize the batch of texts.
                tokenized_docs = self.tokenize(batch_text)
                batch_size_in_tokens = np.vectorize(len)(tokenized_docs).sum()

                # Convert the tokenized batch to a BoW matrix.
                batch_bow = self.bow_from_tokens(tokenized_docs, progress_bar=False)

                # TF-IDF.
                batch_tfidf = self.tfidf_from_bow(batch_bow)

                # Project the BoW matrix to a dense vector space.
                self.bow_to_dense_embedder(batch_tfidf, out=out[start:end])

                # Update the progress bars.
                pbar.update(batch_size)
                pbar_tokens.update(batch_size_in_tokens)

        return out


def initialize_embedder_from_ngram_summary(
    ngram_summary: SpaceSavingNgramSummary[np.int64],
    tokenizer: ArrowTokenizer,
    sparse_to_dense_embedder_dims: Sequence[int],
    min_ngram_count_multiple: float = 10.0,  # Minimum multiple of the lowest approximate count to keep an ngram.
    max_vocabulary_size: int = 2_000_000,  # Maximum number of ngrams to keep as features.
    dtype: type[np.floating] = np.float32,
    embedder_init_seed: int = 0,
) -> Embedder:
    """Construct an `Embedder` from the ngram statistics of a `SpaceSavingNgramSummary`.

    Since the counts are approximate and the approximation is worse for the rarer ngrams,
    let's throw out all ngrams with a count that is less than a certain multiple of the
    lowest approximate count. We'll also limit the vocabulary size to a certain upper
    bound.
    """
    logger.info("Initializing embedder from features based on ngram summary")
    min_approx_count = ngram_summary.approximate_counts.min()
    keep_threshold = min_ngram_count_multiple * min_approx_count
    is_above_threshold = ngram_summary.approximate_counts >= keep_threshold
    fraction_above_threshold = is_above_threshold.mean()
    num_above_threshold = int(is_above_threshold.sum())
    vocabulary_size = min(max_vocabulary_size, num_above_threshold)
    kept_counts = ngram_summary.approximate_counts[-vocabulary_size:]
    kept_ngrams = ngram_summary.ngrams[-vocabulary_size:]
    logger.info(
        f"There are {num_above_threshold:,d} ngrams ({fraction_above_threshold:.2%}) "
        f"with approximate count greater than {keep_threshold:,}, while the max "
        f"vocabulary size is {max_vocabulary_size:,d} ngrams. This gives us a "
        f"vocabulary of {vocabulary_size:,d} ngrams to track in our embedder."
    )

    # Pseudo-IDF (inverse log-scaled term frequency, ignoring the concept of "documents").
    idf_values = (np.log(kept_counts.sum()) - np.log(kept_counts)).astype(dtype)
    logger.info(
        "Determined pseudo-IDF vector for the given vocabulary size of "
        f"{vocabulary_size:,} ngrams with min value {idf_values.min():.2f} and "
        f"max value {idf_values.max():.2f}."
    )

    # Also pull out the ngrams themselves and construct the ngram hash to index map for
    # saving the full embedder state.
    kept_ngrams = ngram_summary.ngrams[-vocabulary_size:]
    ngram_hash_to_ngram_idx = NumbaTypedDict.empty(
        numba.types.int64, numba.types.uint32
    )
    build_ngram_hash_to_idx_map(kept_ngrams, ngram_hash_to_ngram_idx)

    # Randomly initialize the sparse to dense embedder.
    sparse_to_dense_embedder = SparseToDenseEmbedder.create(
        dims=[vocabulary_size] + list(sparse_to_dense_embedder_dims),
        dtype=dtype,
        seed=embedder_init_seed,
    )

    return Embedder(
        tokenizer=tokenizer,
        recognized_ngrams=kept_ngrams,
        ngram_hash_to_ngram_idx=ngram_hash_to_ngram_idx,
        idf_values=idf_values,
        bow_to_dense_embedder=sparse_to_dense_embedder,
    )


@numba.njit(error_model="numpy", parallel=True, nogil=True)
def _fast_tfidf_from_bow(
    bow_data: NDArray[np.integer[Any]],
    bow_indices: NDArray[np.integer[Any]],
    bow_indptr: NDArray[np.integer[Any]],
    idf_values: NDArray[np.floating[Any]],
    tfidf_data: NDArray[np.floating[Any]],
    l2_normalize: bool = True,
) -> None:
    """Fast and memory-efficient TF-IDF transformation of a Bag-of-Words CSR matrix."""
    num_rows = bow_indptr.shape[0] - 1
    for i in numba.prange(num_rows):
        # Get the slice of data and indices for the current row.
        start, end = bow_indptr[i], bow_indptr[i + 1]
        row_tfidf = tfidf_data[start:end]
        row_bow = bow_data[start:end]
        row_indices = bow_indices[start:end]

        # In the first pass over the row, apply IDF weighting and calculate the
        # squared L2 norm of the resulting TF-IDF vector.
        l2_norm_sq = 0.0
        for j in range(row_bow.shape[0]):
            weighted_val = row_bow[j] * idf_values[row_indices[j]]
            row_tfidf[j] = weighted_val
            l2_norm_sq += weighted_val * weighted_val

        # In the second pass, normalize the row by the L2 norm.
        if l2_normalize and l2_norm_sq > 0.0:
            l2_norm = np.sqrt(l2_norm_sq)
            row_tfidf /= l2_norm


# Serialization/deserialization helpers for numba typed dicts.
T_IntDictKey = TypeVar("T_IntDictKey", np.int32, np.int64)
T_IntDictValue = TypeVar("T_IntDictValue", np.int32, np.int64)


def _unpack_int_dict(
    d: NumbaTypedDict[T_IntDictKey, T_IntDictValue],
) -> tuple[NDArray[T_IntDictKey], NDArray[T_IntDictValue]]:
    key_dtype = numba.np.numpy_support.as_dtype(d._numba_type_.key_type)  # type: ignore[attr-defined]
    value_dtype = numba.np.numpy_support.as_dtype(d._numba_type_.value_type)  # type: ignore[attr-defined]
    keys = np.empty(len(d), dtype=key_dtype)
    values = np.empty(len(d), dtype=value_dtype)
    _unpack_int_dict_numba(d, keys, values)  # type: ignore[arg-type]
    return keys, values  # type: ignore[return-value]


@numba.njit
def _unpack_int_dict_numba(
    d: NumbaTypedDict[T_IntDictKey, T_IntDictValue],
    keys: NDArray[T_IntDictKey],
    values: NDArray[T_IntDictValue],
) -> None:
    for i, (k, v) in enumerate(d.items()):
        keys[i] = k
        values[i] = v


def _pack_int_dict(
    keys: NDArray[T_IntDictKey], values: NDArray[T_IntDictValue]
) -> NumbaTypedDict[T_IntDictKey, T_IntDictValue]:
    assert len(keys) == len(values)
    n = len(keys)
    d = NumbaTypedDict.empty(
        key_type=numba.from_dtype(keys.dtype),
        value_type=numba.from_dtype(values.dtype),
        n_keys=n,
    )
    _pack_int_dict_numba(d, keys, values)
    return d


@numba.njit
def _pack_int_dict_numba(
    d: NumbaTypedDict[T_IntDictKey, T_IntDictValue],
    keys: NDArray[T_IntDictKey],
    values: NDArray[T_IntDictValue],
) -> None:
    for k, v in zip(keys, values):
        d[k] = v
