"""
This module contains tools for identifying and Bag-of-Words representing ngrams
in pre-tokenized documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar

import numba
import numpy as np
import scipy.sparse as sp
from numba.typed import Dict as NumbaTypedDict
from numpy.typing import NDArray
from tqdm.auto import tqdm

T_Token = TypeVar("T_Token", bound=np.integer[Any])


@dataclass(frozen=True, kw_only=True)
class SpaceSavingNgramSummary(Generic[T_Token]):
    """Summary of approximate counts of the most frequent ngrams in a collection of
    documents.
    """

    ngrams: NDArray[T_Token]
    approximate_counts: NDArray[np.int64]
    total_ngrams_seen: int
    hash_collisions_skipped: int

    def save_npz(self, path: Path | str) -> None:
        np.savez(path, **self.__dict__)

    @classmethod
    def load_npz(cls, path: Path | str) -> SpaceSavingNgramSummary[T_Token]:
        with np.load(path) as npz:
            return cls(
                ngrams=npz["ngrams"],
                approximate_counts=npz["approximate_counts"],
                total_ngrams_seen=npz["total_ngrams_seen"],
                hash_collisions_skipped=npz["hash_collisions_skipped"],
            )


@dataclass
class _SpaceSavingState(Generic[T_Token]):
    """Internal state for the space-saving summary algorithm."""

    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32]
    ngs: NDArray[T_Token]
    nghs: NDArray[np.int64]
    sorted_counts: NDArray[np.int64]
    collisions: int = 0
    total_seen: int = 0

    def to_summary(self) -> SpaceSavingNgramSummary[T_Token]:
        # If we got fewer than `num_top_items` ngrams, we need to resize the arrays.
        num_top_items = len(self.ngram_hash_to_idx)
        return SpaceSavingNgramSummary(
            ngrams=self.ngs[-num_top_items:],
            approximate_counts=self.sorted_counts[-num_top_items:],
            total_ngrams_seen=self.total_seen,
            hash_collisions_skipped=self.collisions,
        )


@numba.njit(error_model="numpy")
def _space_saving_summary_single_document_update_jit(
    tokens: NDArray[T_Token],
    ng: NDArray[T_Token],
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32],
    ngs: NDArray[T_Token],
    nghs: NDArray[np.int64],
    sorted_counts: NDArray[np.int64],
    empty_token_val: int,
) -> tuple[int, int]:
    """The update step for a single document.

    This is the "hot path" of the algorithm, so we compile it with numba.

    Instead of using linked-list-like data structures to enable fast updates to the
    lowest-count item in the summary (as in the paper), we instead split our
    ngram-to-count mapping into an ngram-to-index mapping and an always-sorted counts
    array which allows for constant-time identification of the lowest-count item in
    exchange for requiring a binary search at every update. This is theoretically
    slower but easier to implement in the array-centric numba paradigm.
    """
    hash_collisions = 0
    assert not np.any(tokens == empty_token_val), "reserved value"
    max_ngram_length = ngs.shape[1]
    total_ngrams_seen = 0
    for ngram_length in range(1, max_ngram_length + 1):
        num_ngrams_of_this_len = len(tokens) - ngram_length + 1
        for i in range(num_ngrams_of_this_len):
            ng[:] = empty_token_val
            ng[:ngram_length] = tokens[i : i + ngram_length]
            # We work with a hash of the ngram array because dictionaries don't
            # allow arrays as keys.
            ngh = fnv1a_hash_array_to_int64(ng)

            # If the ngram is already in the summary, update the count.
            # Since the counts array must stay sorted, this involves a swap.
            if ngh in ngram_hash_to_idx:
                current_idx = ngram_hash_to_idx[ngh]

                # In the unlikely event of a hash-collision, track it and skip this item.
                if not np.array_equal(ngs[current_idx], ng):
                    hash_collisions += 1
                    continue

                # Determine the new count and index.
                current_count = sorted_counts[current_idx]
                new_count = current_count + 1
                new_idx = np.searchsorted(sorted_counts, new_count, side="left") - 1
                new_idx = np.uint32(new_idx)

                # Swap the current and new indices.
                # The target ngram moves to the current idx and easily for us the
                # current count is already correct.
                target_ng = ngs[new_idx]
                target_ngh = nghs[new_idx]
                assert sorted_counts[new_idx] == current_count
                ngs[current_idx] = target_ng
                nghs[current_idx] = target_ngh
                ngram_hash_to_idx[target_ngh] = current_idx

                # The currently updated ngram moves to the new idx and gets a new count.
                ngs[new_idx] = ng
                nghs[new_idx] = ngh
                ngram_hash_to_idx[ngh] = new_idx
                sorted_counts[new_idx] = new_count
            else:
                # Evict the lowest-count ngram, which is at the front of the array
                # since the array is sorted by count.
                evict_idx = np.uint32(0)
                evict_ngh = nghs[evict_idx]
                evict_count = sorted_counts[evict_idx]
                if evict_count > 0:
                    del ngram_hash_to_idx[evict_ngh]

                # Replace the evicted ngram with the new one, performing a swap
                # to keep the count array sorted.
                new_count = evict_count + 1
                new_idx = np.searchsorted(sorted_counts, new_count, side="left") - 1
                new_idx = np.uint32(new_idx)
                assert sorted_counts[new_idx] == evict_count

                # Move the ngram currently at the new idx to the evicted slot.
                target_ng = ngs[new_idx]
                if not np.all(target_ng == empty_token_val):
                    target_ngh = nghs[new_idx]
                    ngs[evict_idx] = target_ng
                    nghs[evict_idx] = target_ngh
                    ngram_hash_to_idx[target_ngh] = evict_idx
                    sorted_counts[evict_idx] = evict_count

                # Place the new ngram into the new idx slot.
                ngs[new_idx] = ng
                nghs[new_idx] = ngh
                ngram_hash_to_idx[ngh] = new_idx
                sorted_counts[new_idx] = new_count
        total_ngrams_seen += num_ngrams_of_this_len
    return hash_collisions, total_ngrams_seen


def merge_summaries(
    summaries: list[SpaceSavingNgramSummary[T_Token]], num_top_items: int
) -> SpaceSavingNgramSummary[T_Token]:
    """Merge multiple summaries into a single summary."""
    # Use a dictionary to map ngram hashes to indices in the arrays.
    ngh_to_idx: dict[int, int] = {}

    # Calculate total stats
    total_ngrams_seen = sum(s.total_ngrams_seen for s in summaries)
    hash_collisions_skipped = sum(s.hash_collisions_skipped for s in summaries)

    max_ngram_length = summaries[0].ngrams.shape[1]
    dtype = summaries[0].ngrams.dtype
    assert all(s.ngrams.shape[1] == max_ngram_length for s in summaries)
    assert all(s.ngrams.dtype == dtype for s in summaries)

    # Pre-allocate arrays and resize as needed.
    initial_capacity = 4096
    merged_ngrams = np.empty((initial_capacity, max_ngram_length), dtype=dtype)
    merged_counts = np.empty(initial_capacity, dtype=np.int64)
    current_size = 0

    for summary in summaries:
        for i, ngram in enumerate(summary.ngrams):
            count = summary.approximate_counts[i]
            ngh = int(fnv1a_hash_array_to_int64(ngram))
            if ngh in ngh_to_idx:
                idx = ngh_to_idx[ngh]
                if not np.array_equal(merged_ngrams[idx], ngram):
                    hash_collisions_skipped += 1
                    continue
                merged_counts[idx] += int(count)
            else:
                if current_size == len(merged_counts):
                    new_capacity = len(merged_counts) * 2
                    merged_ngrams = np.resize(
                        merged_ngrams, (new_capacity, max_ngram_length)
                    )
                    merged_counts = np.resize(merged_counts, new_capacity)

                idx = current_size
                ngh_to_idx[ngh] = idx
                merged_ngrams[idx] = ngram
                merged_counts[idx] = int(count)
                current_size += 1

    # Trim arrays to actual size
    merged_ngrams = merged_ngrams[:current_size]
    merged_counts = merged_counts[:current_size]

    # Get top ngrams
    num_to_keep = min(num_top_items, current_size)
    top_indices = np.argpartition(merged_counts, -num_to_keep)[-num_to_keep:]
    final_ngrams = merged_ngrams[top_indices]
    final_counts = merged_counts[top_indices]
    sort_indices = np.argsort(final_counts)
    final_ngrams = final_ngrams[sort_indices]
    final_counts = final_counts[sort_indices]

    return SpaceSavingNgramSummary(
        ngrams=final_ngrams,
        approximate_counts=final_counts,
        total_ngrams_seen=total_ngrams_seen,
        hash_collisions_skipped=hash_collisions_skipped,
    )


def space_saving_ngram_summary(
    tokenized_texts: Iterable[NDArray[T_Token]],
    max_ngram_length: int = 5,
    num_top_items: int = 10_000_000,
) -> SpaceSavingNgramSummary[T_Token]:
    """Apply the space-saving summary algorithm [1] to all ngrams up to a certain length to
    derive approximate counts for the top items.

    If `num_workers` is not 1, the documents will be split between several parallel
    summaries which will be merged at the end. Set `num_workers` to -1 to use one worker
    per visible CPU core.

    [1] https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf
    """
    # Peek at tokens to get dtype.
    tokenized_texts_iter = iter(tokenized_texts)
    try:
        first_doc = next(tokenized_texts_iter)
    except StopIteration:
        # Handle empty input.
        return SpaceSavingNgramSummary(
            ngrams=np.empty((0, max_ngram_length), dtype=np.uint32),  # type: ignore
            approximate_counts=np.empty(0, dtype=np.int64),
            total_ngrams_seen=0,
            hash_collisions_skipped=0,
        )
    docs_for_processing: Iterable[NDArray[T_Token]] = chain(
        [first_doc], tokenized_texts_iter
    )
    dtype = first_doc.dtype

    # We use a special token value to signify empty slots in the ngram arrays.
    empty_token_val = np.iinfo(dtype).max

    # Pre-allocate arrays for the summary.
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32] = NumbaTypedDict.empty(
        key_type=numba.int64, value_type=numba.uint32, n_keys=num_top_items
    )
    ngs = np.full((num_top_items, max_ngram_length), empty_token_val, dtype=dtype)
    nghs = np.zeros(num_top_items, dtype=np.int64)
    sorted_counts = np.zeros(num_top_items, dtype=np.int64)

    # We work with a temporary buffer for the current ngram to avoid re-allocating it.
    ng = np.full(max_ngram_length, empty_token_val, dtype=dtype)

    state = _SpaceSavingState(
        ngram_hash_to_idx=ngram_hash_to_idx,
        ngs=ngs,
        nghs=nghs,
        sorted_counts=sorted_counts,
    )

    for tokens in docs_for_processing:
        hash_collisions, total_ngrams_seen = (
            _space_saving_summary_single_document_update_jit(
                tokens=tokens,
                ng=ng,
                ngram_hash_to_idx=state.ngram_hash_to_idx,
                ngs=state.ngs,
                nghs=state.nghs,
                sorted_counts=state.sorted_counts,
                empty_token_val=empty_token_val,
            )
        )
        state.collisions += hash_collisions
        state.total_seen += total_ngrams_seen

    return state.to_summary()


@numba.njit(parallel=True, error_model="numpy")
def build_ngram_hash_to_idx_map(
    ngrams: NDArray[T_Token],
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32],
) -> None:
    # Parallelized ngram hashing.
    nghs = np.empty(ngrams.shape[0], dtype=np.int64)
    for i in numba.prange(ngrams.shape[0]):
        nghs[i] = fnv1a_hash_array_to_int64(ngrams[i])

    # Non-parallelized dictionary construction.
    for i in range(nghs.shape[0]):
        ngram_hash_to_idx[nghs[i]] = np.uint32(i)


@numba.njit()
def update_ngram_counts(
    document_tokens: NDArray[T_Token],
    max_ngram_length: int,
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32],
    counts: NDArray[np.uint32],
) -> None:
    empty_token_val = np.iinfo(document_tokens.dtype).max
    assert not np.any(document_tokens == empty_token_val), "reserved value"
    for ngram_length in range(1, max_ngram_length + 1):
        num_ngrams_of_this_len = len(document_tokens) - ngram_length + 1
        for i in range(num_ngrams_of_this_len):
            ng = np.full(max_ngram_length, empty_token_val, dtype=document_tokens.dtype)
            ng[:ngram_length] = document_tokens[i : i + ngram_length]
            ngh = fnv1a_hash_array_to_int64(ng)
            if ngh in ngram_hash_to_idx:
                idx = ngram_hash_to_idx[ngh]
                counts[idx] += 1


# Using `nogil=True` allows us to parallelize calls to this function across CPU cores
# using plain Python threads -- no multiprocessing required!
@numba.njit(nogil=True)
def sparse_count_unigram_in_document(
    tokens: NDArray[T_Token],
) -> tuple[NDArray[T_Token], NDArray[np.uint32]]:
    """Gets the indices and counts of the unique tokens in the document."""
    count_dict = NumbaTypedDict.empty(key_type=tokens.dtype, value_type=numba.uint32)
    for token in tokens:
        if token in count_dict:
            count_dict[token] += 1
        else:
            count_dict[token] = 1
    idx_array = np.empty(len(count_dict), dtype=tokens.dtype)
    count_array = np.empty(len(count_dict), dtype=np.uint32)
    for i, (k, v) in enumerate(count_dict.items()):
        idx_array[i] = k
        count_array[i] = v
    return idx_array, count_array


# Using `nogil=True` allows us to parallelize calls to this function across CPU cores
# using plain Python threads -- no multiprocessing required!
@numba.njit(nogil=True)
def sparse_count_ngram_in_document(
    max_ngram_length: int,
    tokens: NDArray[T_Token],
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32],
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Gets the indices and counts of the ngrams in the document."""
    count_dict = NumbaTypedDict.empty(key_type=numba.uint32, value_type=numba.uint32)
    empty_token_val = np.iinfo(tokens.dtype).max
    assert not np.any(tokens == empty_token_val), "reserved value"
    ng = np.full(max_ngram_length, empty_token_val, dtype=tokens.dtype)
    for ngram_length in range(1, max_ngram_length + 1):
        num_ngrams_of_this_len = len(tokens) - ngram_length + 1
        for i in range(num_ngrams_of_this_len):
            ng[:] = empty_token_val
            ng[:ngram_length] = tokens[i : i + ngram_length]
            ngh = fnv1a_hash_array_to_int64(ng)
            if ngh in ngram_hash_to_idx:
                idx = ngram_hash_to_idx[ngh]
                if idx in count_dict:
                    count_dict[idx] = np.uint32(count_dict[idx] + np.uint32(1))
                else:
                    count_dict[idx] = np.uint32(1)
    idx_array = np.empty(len(count_dict), dtype=np.uint32)
    count_array = np.empty(len(count_dict), dtype=np.uint32)
    for i, (k, v) in enumerate(count_dict.items()):
        idx_array[i] = k
        count_array[i] = v
    return idx_array, count_array


def bow_matrix_ngrams(
    tokenized_docs: Iterable[NDArray[T_Token]],
    max_ngram_length: int,
    ngram_hash_to_idx: NumbaTypedDict[np.int64, np.uint32],
    num_workers: int = -1,
    worker_chunksize: int = 8,
    progress_bar: bool = True,
    size_hint: int | None = None,
) -> sp.csr_matrix:
    """Create a CSR sparse matrix of ngram counts."""
    offset = 0
    values_chunks: list[NDArray[np.uint32]] = []
    j_indices_chunks: list[NDArray[np.uint32]] = []
    indptr: list[int] = [offset]
    num_workers = cpu_count() if num_workers == -1 else num_workers
    if size_hint is None and hasattr(tokenized_docs, "__len__"):
        size_hint = len(tokenized_docs)  # type: ignore
    with (
        ThreadPool(processes=num_workers) as pool,
        tqdm(
            desc="BoW",
            unit="seq",
            unit_scale=True,
            disable=not progress_bar,
            total=size_hint,
        ) as pbar,
        tqdm(
            unit="token", position=1, unit_scale=True, disable=not progress_bar
        ) as pbar_tok,
    ):
        it = pool.imap(
            lambda tokens: sparse_count_ngram_in_document(
                max_ngram_length=max_ngram_length,
                tokens=tokens,
                ngram_hash_to_idx=ngram_hash_to_idx,
            ),
            tokenized_docs,
            chunksize=worker_chunksize,
        )

        for indices, counts in it:
            # Update the CSR matrix state.
            j_indices_chunks.append(indices)
            values_chunks.append(counts)
            offset += len(indices)
            indptr.append(offset)
            pbar.update()
            pbar_tok.update(counts.sum())

    # Construct the CSR matrix, just as above in the unigram case.
    j_indices = np.concatenate(j_indices_chunks)
    values = np.concatenate(values_chunks)
    vocab_size = len(ngram_hash_to_idx)
    matrix = sp.csr_matrix(
        (values, j_indices, indptr), shape=(len(indptr) - 1, vocab_size)
    )
    return matrix


def bow_matrix_unigrams(
    tokenized_docs: Iterable[NDArray[T_Token]],
    vocab_size: int,
    num_workers: int = -1,
    progress_bar: bool = True,
    worker_chunksize: int = 8,
    size_hint: int | None = None,
) -> sp.csr_matrix:
    """Create a CSR sparse matrix of unigram counts."""
    offset = 0
    values_chunks: list[NDArray[np.uint32]] = []
    j_indices_chunks: list[NDArray[T_Token]] = []
    indptr: list[int] = [offset]
    num_workers = cpu_count() if num_workers == -1 else num_workers
    if size_hint is None and hasattr(tokenized_docs, "__len__"):
        size_hint = len(tokenized_docs)  # type: ignore
    with (
        ThreadPool(processes=num_workers) as pool,
        tqdm(
            desc="BoW",
            unit="seq",
            unit_scale=True,
            disable=not progress_bar,
            total=size_hint,
        ) as pbar,
        tqdm(
            unit="token", position=1, unit_scale=True, disable=not progress_bar
        ) as pbar_tok,
    ):
        # Since `sparse_count_ngram_in_document` releases the GIL, we can use multiple
        # threads to saturate multiple CPU cores and linearly accelerate the computation.
        it = pool.imap(
            sparse_count_unigram_in_document,
            tokenized_docs,
            chunksize=worker_chunksize,
        )
        for indices, counts in it:
            # Update the CSR matrix state.
            j_indices_chunks.append(indices)
            values_chunks.append(counts)
            offset += len(indices)
            indptr.append(offset)
            pbar.update()
            pbar_tok.update(counts.sum())

    # Construct the CSR matrix, just as above in the unigram case.
    j_indices = np.concatenate(j_indices_chunks)
    values = np.concatenate(values_chunks)
    matrix = sp.csr_matrix(
        (values, j_indices, indptr), shape=(len(indptr) - 1, vocab_size)
    )
    return matrix


@numba.njit(error_model="numpy", inline="always")
def fnv1a_hash_array_to_int64(ndarray: NDArray[T_Token]) -> np.int64:
    """A fast hash funtion which can be called from within `numba.njit()` to hash
    numpy arrays.

    Reference: https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
    """
    FNV_OFFSET_BASIS_64 = np.uint64(14695981039346656037)
    FNV_PRIME_64 = np.uint64(1099511628211)
    byte_view = ndarray.ravel().view(np.uint8)
    hash_val = FNV_OFFSET_BASIS_64
    for byte_val in byte_view:
        hash_val ^= np.uint64(byte_val)
        hash_val *= FNV_PRIME_64
    return np.int64(hash_val)
