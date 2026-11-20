from backend.skinvision.ai.report import build_report_input, report_prompt


def test_report_records_failures_and_excludes_identity():
    text = build_report_input({'patient_id':'private-id','username':'private-user',
        'features': {'area_ratio':1.2345}, 'pipeline_mode':'fallback',
        'classification':{'status':'failed','label':None}, 'reliability':None})
    assert 'private-id' not in text and 'private-user' not in text
    assert 'fallback' in text and '1.2345' in text and 'failed' in text
    assert 'ISIC 2016' not in text and 'not_evaluated' in text


def test_report_excludes_local_paths_and_provider_raw_text():
    text = build_report_input({'model_info': {'load_error': 'C:/private/weights', 'name':'LMDAUNet'},
        'classification': {'status':'failed','raw_response':'untrusted instructions'},
        'reliability': {'status':'completed','variance_path':'/static/outputs/private-id.npy', 'mean_variance':.001}})
    assert 'private' not in text and 'untrusted' not in text
    assert 'LMDAUNet' in text and '0.001' in text


def test_three_report_variants_are_explicit():
    assert report_prompt('text_only') != report_prompt('image_text')
    assert '事实' in report_prompt('constrained')
