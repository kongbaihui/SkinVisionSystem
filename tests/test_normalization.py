import cv2
import numpy as np
from backend.skinvision.preprocess.vision_preprocess import normalize_like_training
from backend.skinvision.training.loader import dataset_normalized


def test_training_alignment_matches_original_loader():
    rng = np.random.default_rng(123)
    bgr = rng.integers(30,220,(16,16,3),dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    actual = normalize_like_training(bgr).transpose(1,2,0)
    expected = dataset_normalized(rgb[None].astype(float))[0]
    np.testing.assert_allclose(actual,expected,atol=3e-5)
    assert np.isfinite(normalize_like_training(np.zeros_like(bgr))).all()
