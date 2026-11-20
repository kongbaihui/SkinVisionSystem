import asyncio
import numpy as np
import pytest
import torch
from backend.skinvision.ai.local_classification import LocalClassifier, adapt_state_dict


def test_checkpoint_wrapper_mapping_rejects_unknown_weights():
    weight = torch.ones(1)
    result = adapt_state_dict({'encoder.0.0.0.weight':weight, 'classifier.weight':weight})
    assert set(result) == {'features.0.0.weight', 'classifier.1.weight'}
    with pytest.raises(ValueError, match='Unexpected'):
        adapt_state_dict({'wrong.weight':weight})


def test_local_classifier_missing_and_uniform_inputs(tmp_path):
    image = np.zeros((32,32,3), np.uint8)
    assert asyncio.run(LocalClassifier('').classify(image))['status'] == 'not_configured'
    classifier = LocalClassifier(tmp_path)
    assert asyncio.run(classifier.classify(image))['status'] == 'review_required'
    image[10:20] = 128
    result = asyncio.run(classifier.classify(image))
    assert result['status'] == 'failed' and result['label'] is None
