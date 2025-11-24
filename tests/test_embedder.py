import tempfile
from pathlib import Path
from unittest.mock import Mock

import numba
import numpy as np
from numba.typed import Dict as NumbaTypedDict

from luxical.embedder import (
    EMBEDDER_FORMAT_VERSION,
    Embedder,
    _fast_tfidf_from_bow,
    _pack_int_dict,
    _unpack_int_dict,
    initialize_embedder_from_ngram_summary,
)
from luxical.ngrams import SpaceSavingNgramSummary
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder
from luxical.tokenization import ArrowTokenizer


def test_numba_dict_serialization():
    """Test serialization and deserialization of numba typed dicts."""
    d = NumbaTypedDict.empty(numba.types.int64, numba.types.uint32)
    d[12345] = np.uint32(100)
    d[67890] = np.uint32(200)
    d[123] = np.uint32(50)
    keys, values = _unpack_int_dict(d)
    reconstructed_d = _pack_int_dict(keys, values)
    assert reconstructed_d == d


def test_tfidf_computation():
    """Test the TF-IDF computation with known input/output."""
    # Create a simple BOW matrix: 2 docs, 3 features
    # Doc 1: [2, 0, 1] (term frequencies)
    # Doc 2: [1, 3, 0]
    bow_data = np.array([2, 1, 1, 3], dtype=np.int32)
    bow_indices = np.array([0, 2, 0, 1], dtype=np.int32)
    bow_indptr = np.array([0, 2, 4], dtype=np.int32)

    # IDF values (log-scaled inverse frequencies)
    idf_values = np.array([0.5, 1.0, 2.0], dtype=np.float32)

    # Expected TF-IDF values
    # Doc 1: [2*0.5, 0*1.0, 1*2.0] = [1.0, 0.0, 2.0]
    # Doc 2: [1*0.5, 3*1.0, 0*2.0] = [0.5, 3.0, 0.0]
    # After L2 normalization:
    # Doc 1 norm = sqrt(1.0^2 + 2.0^2) = sqrt(5) ≈ 2.236
    # Doc 2 norm = sqrt(0.5^2 + 3.0^2) = sqrt(9.25) ≈ 3.041

    tfidf_data = np.empty_like(bow_data, dtype=np.float32)
    _fast_tfidf_from_bow(
        bow_data=bow_data,
        bow_indices=bow_indices,
        bow_indptr=bow_indptr,
        idf_values=idf_values,
        tfidf_data=tfidf_data,
        l2_normalize=True,
    )

    # Verify shapes and approximate values
    assert tfidf_data.shape == bow_data.shape
    assert np.all(np.isfinite(tfidf_data))

    # Check first document values (normalized)
    expected_doc1 = np.array([1.0, 2.0]) / np.linalg.norm([1.0, 2.0])
    np.testing.assert_allclose(tfidf_data[:2], expected_doc1, rtol=1e-5)


def test_embedder_save_load_roundtrip():
    """Test that embedder can be saved and loaded correctly."""
    # Create minimal embedder components
    tokenizer = Mock(spec=ArrowTokenizer)
    tokenizer.to_str.return_value = '{"test": "tokenizer"}'

    recognized_ngrams = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    ngram_hash_to_idx = NumbaTypedDict.empty(numba.types.int64, numba.types.uint32)
    ngram_hash_to_idx[123] = np.uint32(0)
    ngram_hash_to_idx[456] = np.uint32(1)
    ngram_hash_to_idx[789] = np.uint32(2)

    idf_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Create neural network layers
    rng = np.random.default_rng(0)
    layers = [
        rng.normal(size=(3, 4)).astype(np.float32),
        rng.normal(size=(4, 2)).astype(np.float32),
    ]
    bow_to_dense = SparseToDenseEmbedder(layers=layers)

    embedder = Embedder(
        tokenizer=tokenizer,
        recognized_ngrams=recognized_ngrams,
        ngram_hash_to_ngram_idx=ngram_hash_to_idx,
        idf_values=idf_values,
        bow_to_dense_embedder=bow_to_dense,
    )

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        embedder.save(f.name)

        # Mock the tokenizer constructor for loading
        def mock_tokenizer_init(json_str):
            mock_tok = Mock(spec=ArrowTokenizer)
            mock_tok.to_str.return_value = json_str
            return mock_tok

        # Patch ArrowTokenizer constructor
        import luxical.embedder

        original_tokenizer = luxical.embedder.ArrowTokenizer
        try:
            luxical.embedder.ArrowTokenizer = mock_tokenizer_init
            loaded_embedder = Embedder.load(f.name)
        finally:
            luxical.embedder.ArrowTokenizer = original_tokenizer

        # Verify data integrity
        np.testing.assert_array_equal(
            loaded_embedder.recognized_ngrams, recognized_ngrams
        )
        np.testing.assert_array_equal(loaded_embedder.idf_values, idf_values)
        assert len(loaded_embedder.ngram_hash_to_ngram_idx) == len(ngram_hash_to_idx)
        for k in ngram_hash_to_idx.keys():
            assert loaded_embedder.ngram_hash_to_ngram_idx[k] == ngram_hash_to_idx[k]

        # Verify neural network layers
        assert len(loaded_embedder.bow_to_dense_embedder.layers) == len(layers)
        for orig, loaded in zip(layers, loaded_embedder.bow_to_dense_embedder.layers):
            np.testing.assert_array_equal(orig, loaded)


def test_embedder_version_checking():
    """Test that version checking works correctly during load."""
    # Create minimal embedder
    tokenizer = Mock(spec=ArrowTokenizer)
    tokenizer.to_str.return_value = '{"test": "tokenizer"}'

    recognized_ngrams = np.array([[1, 2]], dtype=np.int64)
    ngram_hash_to_idx = NumbaTypedDict.empty(numba.types.int64, numba.types.uint32)
    ngram_hash_to_idx[123] = np.uint32(0)
    idf_values = np.array([1.0], dtype=np.float32)
    layers = [np.random.randn(1, 2).astype(np.float32)]
    bow_to_dense = SparseToDenseEmbedder(layers=layers)

    embedder = Embedder(
        tokenizer=tokenizer,
        recognized_ngrams=recognized_ngrams,
        ngram_hash_to_ngram_idx=ngram_hash_to_idx,
        idf_values=idf_values,
        bow_to_dense_embedder=bow_to_dense,
    )

    with (
        tempfile.NamedTemporaryFile(suffix=".npz") as f,
        tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as bad_f,
    ):
        embedder.save(f.name)

        # Verify the version is saved correctly
        with np.load(f.name) as npzfile:
            saved_version = npzfile["version"].item()
            assert saved_version == EMBEDDER_FORMAT_VERSION

        # Test loading unsupported version by manually creating a bad file
        # Load original data and modify version
        data = dict(np.load(f.name))
        data["version"] = np.array([999], dtype=np.int64)  # Unsupported version
        np.savez(bad_f.name, **data)

        try:
            import luxical.embedder

            original_tokenizer = luxical.embedder.ArrowTokenizer
            luxical.embedder.ArrowTokenizer = lambda x: Mock(spec=ArrowTokenizer)

            # Should raise NotImplementedError for unsupported version
            try:
                Embedder.load(bad_f.name)
                assert False, "Should have raised NotImplementedError"
            except NotImplementedError as e:
                assert "999" in str(e)
                assert str(EMBEDDER_FORMAT_VERSION) in str(e)
            finally:
                luxical.embedder.ArrowTokenizer = original_tokenizer
        finally:
            Path(bad_f.name).unlink()


def test_initialize_embedder_from_ngram_summary():
    """Test embedder initialization from ngram summary."""
    # Create test ngram summary
    ngrams = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    counts = np.array([10, 25, 50, 100], dtype=np.int64)  # Sorted ascending
    summary = SpaceSavingNgramSummary(
        ngrams=ngrams,
        approximate_counts=counts,
        total_ngrams_seen=200,
        hash_collisions_skipped=1,
    )

    tokenizer = Mock(spec=ArrowTokenizer)
    embedder = initialize_embedder_from_ngram_summary(
        ngram_summary=summary,
        tokenizer=tokenizer,
        sparse_to_dense_embedder_dims=[5, 4],
        min_ngram_count_multiple=2.0,  # Keep ngrams with count >= 2*10 = 20
        max_vocabulary_size=10,
        embedder_init_seed=42,
    )

    # Should keep 3 ngrams (counts 100, 50, 25) and drop the one with count 10
    assert embedder.recognized_ngrams.shape[0] == 3
    assert embedder.idf_values.shape[0] == 3
    assert len(embedder.ngram_hash_to_ngram_idx) == 3

    # Check neural network dimensions (layers are stored as output_dim x input_dim)
    assert embedder.bow_to_dense_embedder.layers[0].shape == (5, 3)
    assert embedder.bow_to_dense_embedder.layers[1].shape == (4, 5)

    # Verify basic IDF properties (actual computation is complex with thresholding)
    assert np.all(embedder.idf_values > 0)  # IDF values should be positive
    assert np.all(np.isfinite(embedder.idf_values))  # Should be finite values
    # Rarer terms should have higher IDF values
    assert embedder.idf_values[0] > embedder.idf_values[-1]
