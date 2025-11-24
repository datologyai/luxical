from __future__ import annotations

import numpy as np
import pyarrow as pa

from luxical.misc_utils import (
    numpy_ndarray_to_pyarrow_fixed_size_list_array,
    pyarrow_fixed_size_list_array_to_numpy_ndarray,
)


def test_numpy_to_pyarrow_roundtrip() -> None:
    """Test round-trip conversion: numpy -> pyarrow -> numpy."""
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    pa_array = numpy_ndarray_to_pyarrow_fixed_size_list_array(original)
    result = pyarrow_fixed_size_list_array_to_numpy_ndarray(pa_array)

    assert np.allclose(original, result)
    assert original.shape == result.shape
    assert original.dtype == result.dtype


def test_numpy_to_pyarrow_roundtrip_different_shapes() -> None:
    """Test round-trip with different array shapes."""
    test_cases = [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.random.randn(10, 128).astype(np.float32),
        np.random.randn(100, 64).astype(np.float32),
    ]

    for original in test_cases:
        pa_array = numpy_ndarray_to_pyarrow_fixed_size_list_array(original)
        result = pyarrow_fixed_size_list_array_to_numpy_ndarray(pa_array)

        assert np.allclose(original, result), f"Failed for shape {original.shape}"
        assert original.shape == result.shape
        assert original.dtype == result.dtype


def test_numpy_to_pyarrow_roundtrip_chunked_array() -> None:
    """Test round-trip with ChunkedArray input."""
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    pa_array = numpy_ndarray_to_pyarrow_fixed_size_list_array(original)
    chunked = pa.chunked_array([pa_array])

    result = pyarrow_fixed_size_list_array_to_numpy_ndarray(chunked)

    assert np.allclose(original, result)
    assert original.shape == result.shape
    assert original.dtype == result.dtype


def test_numpy_to_pyarrow_roundtrip_different_dtypes() -> None:
    """Test round-trip with different numpy dtypes."""
    test_cases = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    ]

    for original in test_cases:
        pa_array = numpy_ndarray_to_pyarrow_fixed_size_list_array(original)
        result = pyarrow_fixed_size_list_array_to_numpy_ndarray(pa_array)

        assert np.allclose(original, result), f"Failed for dtype {original.dtype}"
        assert original.shape == result.shape
