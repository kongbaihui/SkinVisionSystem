import numpy as np
import pytest
from backend.skinvision.ai.reliability import analyze_reliability, occlusion_sensitivity


def test_tta_restores_flips_to_original_coordinates():
    chw = np.random.default_rng(42).random((3, 12, 16)).astype('float32')
    result, variance = analyze_reliability(lambda x: x[0], chw, rotations=())
    assert result['status'] == 'completed'
    assert result['prediction_count'] == 3
    assert result['mean_variance'] < 1e-12 and np.max(variance) < 1e-12


def test_rotation_excludes_missing_edges():
    chw = np.ones((3, 16, 16), np.float32)
    result, variance = analyze_reliability(lambda x: np.ones(x.shape[1:],np.float32)*.7, chw)
    assert result['mean_variance'] < 1e-12
    assert 0 < result['fully_covered_fraction'] < 1
    assert np.isfinite(variance).all()


def test_invalid_predictions_fail_explicitly():
    with pytest.raises(ValueError, match='finite'):
        analyze_reliability(lambda x: np.full(x.shape[1:],np.nan), np.ones((3,8,8),np.float32))


def test_occlusion_measures_real_prediction_change():
    chw = np.ones((3,8,8),np.float32)
    result, influence = occlusion_sensitivity(lambda x: x[0], chw, grid=2, fill=0.)
    assert result['prediction_count'] == 5
    np.testing.assert_allclose(influence, .25)
