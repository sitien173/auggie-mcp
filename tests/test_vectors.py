from __future__ import annotations

import numpy as np

from auggie_mcp.vectors import normalize_rows


def test_normalize_rows_returns_l2_normalized_float_matrix() -> None:
    actual = normalize_rows(np.array([[3, 4], [0, 2]], dtype=np.float32))
    np.testing.assert_allclose(actual, [[0.6, 0.8], [0, 1]])
