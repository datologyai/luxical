import numpy as np
from collections import Counter

from luxical.ngrams import (
    space_saving_ngram_summary,
    merge_summaries,
    SpaceSavingNgramSummary,
)


def test_space_saving_ngram_summary_basic():
    """Test basic functionality with a simple case where counts are exact."""
    tokenized_texts = [
        np.array([1, 2, 3, 1, 2, 4], dtype=np.uint16),
        np.array([1, 2, 3, 5, 6, 7], dtype=np.uint16),
    ]
    max_ngram_length = 2
    # Set num_top_items high enough to get exact counts. There are 14 unique ngrams.
    summary = space_saving_ngram_summary(
        tokenized_texts, max_ngram_length=max_ngram_length, num_top_items=50
    )

    # Determine the padding value used
    dtype = tokenized_texts[0].dtype
    pad_val = np.iinfo(dtype).max

    # Calculate expected counts
    all_ngrams = []
    for doc in tokenized_texts:
        # 1-grams
        for i in range(len(doc)):
            ngram = np.full(max_ngram_length, pad_val, dtype=dtype)
            ngram[0] = doc[i]
            all_ngrams.append(tuple(ngram))
        # 2-grams
        for i in range(len(doc) - max_ngram_length + 1):
            ngram = np.full(max_ngram_length, pad_val, dtype=dtype)
            ngram[:max_ngram_length] = doc[i : i + max_ngram_length]
            all_ngrams.append(tuple(ngram))

    expected_counts = Counter(all_ngrams)

    # Format summary results for comparison
    summary_counts_dict = {
        tuple(ng): count
        for ng, count in zip(
            (tuple(x) for x in summary.ngrams.tolist()), summary.approximate_counts
        )
    }

    assert summary_counts_dict == expected_counts


def test_space_saving_ngram_summary_empty_input():
    """Test behavior with empty input."""
    summary = space_saving_ngram_summary([], max_ngram_length=5, num_top_items=10)
    assert summary.ngrams.shape[0] == 0
    assert summary.approximate_counts.shape[0] == 0
    assert summary.total_ngrams_seen == 0
    assert summary.hash_collisions_skipped == 0


def test_space_saving_ngram_summary_num_top_items():
    """Test if `num_top_items` is respected."""
    tokenized_texts = [np.arange(20) for _ in range(5)]
    # There are more than 10 unique ngrams.
    summary = space_saving_ngram_summary(
        tokenized_texts, max_ngram_length=2, num_top_items=10
    )
    assert summary.ngrams.shape[0] <= 10


def test_docs_shorter_than_max_ngram_length():
    """Test documents that are shorter than max_ngram_length."""
    tokenized_texts = [
        np.array([1, 2]),
        np.array([3, 4, 5]),
    ]
    summary = space_saving_ngram_summary(
        tokenized_texts, max_ngram_length=5, num_top_items=10
    )
    # The summary should not be empty
    assert summary.ngrams.shape[0] > 0
    assert summary.total_ngrams_seen > 0


def test_ngram_summary_merge():
    """Test merging two NgramSummary objects with exact counts."""
    tokenized_texts1 = [
        np.array([1, 2, 3, 1, 2, 4], dtype=np.uint16),
    ]
    tokenized_texts2 = [
        np.array([1, 2, 3, 5, 6, 7], dtype=np.uint16),
    ]
    max_ngram_length = 2
    num_top_items = 50  # High enough to get exact counts

    summary1 = space_saving_ngram_summary(
        tokenized_texts1,
        max_ngram_length=max_ngram_length,
        num_top_items=num_top_items,
    )
    summary2 = space_saving_ngram_summary(
        tokenized_texts2,
        max_ngram_length=max_ngram_length,
        num_top_items=num_top_items,
    )

    merged_summary = merge_summaries([summary1, summary2], num_top_items=num_top_items)

    combined_texts = tokenized_texts1 + tokenized_texts2
    expected_summary = space_saving_ngram_summary(
        combined_texts,
        max_ngram_length=max_ngram_length,
        num_top_items=num_top_items,
    )

    assert merged_summary.total_ngrams_seen == expected_summary.total_ngrams_seen
    assert (
        merged_summary.hash_collisions_skipped
        == expected_summary.hash_collisions_skipped
    )

    merged_counts_dict = {
        tuple(ng): count
        for ng, count in zip(
            (tuple(x) for x in merged_summary.ngrams.tolist()),
            merged_summary.approximate_counts,
        )
    }
    expected_counts_dict = {
        tuple(ng): count
        for ng, count in zip(
            (tuple(x) for x in expected_summary.ngrams.tolist()),
            expected_summary.approximate_counts,
        )
    }
    assert merged_counts_dict == expected_counts_dict


def test_ngram_summary_merge_truncation():
    """Test that merging correctly truncates to `num_top_items`."""
    ngrams1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.uint16)
    counts1 = np.array([10, 9, 3, 2, 1], dtype=np.int64)
    summary1 = SpaceSavingNgramSummary(
        ngrams=ngrams1,
        approximate_counts=counts1,
        total_ngrams_seen=np.sum(counts1).item(),
        hash_collisions_skipped=0,
    )

    ngrams2 = np.array(
        [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=np.uint16
    )
    counts2 = np.array([8, 7, 6, 5, 4], dtype=np.int64)
    summary2 = SpaceSavingNgramSummary(
        ngrams=ngrams2,
        approximate_counts=counts2,
        total_ngrams_seen=np.sum(counts2).item(),
        hash_collisions_skipped=0,
    )

    merged_summary = merge_summaries([summary1, summary2], num_top_items=5)

    assert merged_summary.ngrams.shape[0] == 5

    expected_top_counts = {
        (1, 2): 10,
        (3, 4): 9,
        (11, 12): 8,
        (13, 14): 7,
        (15, 16): 6,
    }

    merged_counts_dict = {
        tuple(ng): count
        for ng, count in zip(
            (tuple(x) for x in merged_summary.ngrams.tolist()),
            merged_summary.approximate_counts,
        )
    }

    assert merged_counts_dict == expected_top_counts
    assert (
        merged_summary.total_ngrams_seen
        == summary1.total_ngrams_seen + summary2.total_ngrams_seen
    )
