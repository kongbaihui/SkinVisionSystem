import asyncio
import json
import httpx
import numpy as np
import pytest
from backend.skinvision.ai.classification import Classifier, parse_prediction, prepare_input


def test_valid_labels_and_review_are_separate():
    result = parse_prediction('{"status":"completed","label":"NV","candidates":["NV","BKL"],"reason":"visual candidate"}')
    assert result['label'] == 'NV'
    review = parse_prediction('{"status":"review_required","label":null,"candidates":[],"reason":"insufficient evidence"}')
    assert review['label'] is None
    for value in ['OTHER', 'nv', 'benign']:
        with pytest.raises(ValueError):
            parse_prediction(json.dumps({'status':'completed','label':value,'candidates':[value],'reason':'x'}))


def test_self_reported_confidence_is_rejected():
    with pytest.raises(ValueError):
        parse_prediction('{"status":"completed","label":"NV","candidates":["NV"],"reason":"x","confidence":0.99}')


def test_roi_uses_mask_and_empty_mask_needs_review():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6:10, 8:12] = 255
    cropped, info = prepare_input(image, mask, 'roi')
    assert cropped.shape[:2] == (6, 6) and info['bbox_xyxy'] == [7, 5, 13, 11]
    with pytest.raises(ValueError, match='empty'):
        prepare_input(image, mask * 0, 'roi')


def test_classification_request_is_independent_and_parse_failure_is_recorded():
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'output':{'choices':[{'message':{'content':[{'text':'not json'}]}}]}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await Classifier('test-key', client=client).classify(np.ones((8,8,3),np.uint8))
    result = asyncio.run(run())
    assert result['status'] == 'failed' and result['raw_response'] == 'not json'
    assert result['prompt_version'] and result['elapsed_ms'] >= 0
    assert len(requests) == 1 and 'patient' not in str(requests[0])


def test_no_key_has_explicit_status():
    result = asyncio.run(Classifier('').classify(np.ones((8,8,3),np.uint8)))
    assert result['status'] == 'not_configured' and result['label'] is None
