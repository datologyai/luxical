from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from luxical.csr_matrix_utils import csr_matrix_from_npz, csr_matrix_to_npz


def test_csr_roundtrip() -> None:
    m = csr_matrix(np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        csr_matrix_to_npz(m, p)
        m2 = csr_matrix_from_npz(p)
    assert np.allclose(m.toarray(), m2.toarray())
