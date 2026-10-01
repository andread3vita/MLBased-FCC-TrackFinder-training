import numpy as np
import polars as pl

from src.eval.report_final_slices import nearest_truth_distance


def test_nearest_truth_distance_wraps_phi_and_stays_within_event():
    frame = pl.DataFrame({
        "seed": [1, 1, 1, 1],
        "event_id": [10, 10, 10, 11],
        "eta": [0.0, 0.0, 2.0, 0.0],
        "phi": [-np.pi + 0.01, np.pi - 0.01, 0.0, 0.0],
    })
    distances = nearest_truth_distance(frame)

    np.testing.assert_allclose(distances[:2], [0.02, 0.02], atol=1e-10)
    assert distances[2] > 2.0
    assert np.isnan(distances[3])
