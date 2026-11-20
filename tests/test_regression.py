"""Local API regression. Synthetic input and no external API; not AI acceptance."""
import importlib
import cv2
import numpy as np
from fastapi.testclient import TestClient


def test_authenticated_browser_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv('SKINVISION_DB_URL', 'sqlite:///' + str(tmp_path / 'test.db'))
    monkeypatch.setenv('SKINVISION_TEMP_DIR', str(tmp_path / 'images'))
    monkeypatch.setenv('SKINVISION_ENABLE_AUTH', 'true')
    monkeypatch.setenv('SKINVISION_AUTH_SECRET', 'test-only-secret-never-for-production-12345678')
    monkeypatch.setenv('SKINVISION_WEIGHTS', '')
    monkeypatch.setenv('DASHSCOPE_API_KEY', '')
    monkeypatch.setenv('ALIYUN_API_KEY', '')
    monkeypatch.setenv('SKINVISION_CLASSIFICATION_BACKEND', 'api')
    module = importlib.import_module('backend.skinvision.app')
    monkeypatch.setattr(module, '_start_cleanup_thread', lambda: None)
    with TestClient(module.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/api/v1/tasks').status_code == 401
        credentials = {'username': 'testdoctor', 'password': 'Testpass123!'}
        assert client.post('/api/v1/auth/register', json=credentials).status_code == 201
        response = client.post('/api/v1/auth/login', json=credentials)
        assert response.status_code == 200, response.text
        token = response.json()['access_token']
        headers = {'Authorization': 'Bearer ' + token}
        assert client.get('/api/v1/auth/me', headers=headers).status_code == 200
        assert client.get('/api/v1/admin/stats', headers=headers).status_code == 403
        image = np.full((64, 64, 3), 180, dtype=np.uint8)
        cv2.circle(image, (32, 32), 12, (50, 60, 70), -1)
        content = cv2.imencode('.png', image)[1].tobytes()
        form = {'patient_age': '30', 'symptoms': 'test', 'duration': 'one month'}
        files = {'image': ('synthetic.png', content, 'image/png')}
        for endpoint in ['analyze', 'analyze_stream']:
            invalid = client.post('/api/v1/diagnostic/' + endpoint, data=form,
                                 files={'image': ('invalid.png', b'invalid', 'image/png')}, headers=headers)
            assert invalid.status_code == 400 and invalid.json()['error'] == 'invalid image file'
        response = client.post('/api/v1/diagnostic/analyze', data=form, files=files, headers=headers)
        assert response.status_code == 200, response.text
        task = response.json()
        assert task['pipeline_mode'] == 'fallback'
        assert task['gradcam_url'] is None
        assert task['classification']['status'] == 'not_configured'
        assert task['reliability']['status'] == 'not_available'
        assert task['model_info']['execution'] == 'fallback'
        assert client.get(task['probability_path']).status_code == 200
        assert client.get(task['overlay_path']).status_code == 200
        assert client.get(task['mask_path']).status_code == 200
        assert client.get('/api/v1/tasks', headers=headers).json()['pagination']['total_count'] == 1
        assert client.get(f"/api/v1/tasks/{task['id']}", headers=headers).status_code == 200
        report = client.get(f"/api/v1/tasks/{task['id']}/export", headers=headers)
        assert report.status_code == 200 and 'attachment' in report.headers['content-disposition']
        assert 'not_configured' in report.text and 'not_available' in report.text
        pdf = client.get(f"/api/v1/tasks/{task['id']}/export?format=pdf", headers=headers)
        assert pdf.status_code == 200 and pdf.content.startswith(b'%PDF-')
        # Old referenced images must survive cleanup; only abandoned generated files expire.
        import os
        from pathlib import Path
        mask_file = Path(module.TEMP_DIR) / task['mask_path'].rsplit('/', 1)[1]
        os.utime(mask_file, (1, 1))
        orphan = Path(module.TEMP_DIR) / 'mask_deadbeef.png'
        orphan.write_bytes(b'orphan test artifact')
        os.utime(orphan, (1, 1))
        assert module._cleanup_old_files() == 1
        assert mask_file.exists() and not orphan.exists()
        streamed = client.post('/api/v1/diagnostic/analyze_stream', data=form, files=files, headers=headers)
        assert streamed.status_code == 200
        assert all(f'event: {event}' in streamed.text for event in ['meta', 'classification', 'reliability', 'chunk', 'done'])
        import json
        events = [(block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:]))
                  for block in streamed.text.strip().split('\n\n') if block.startswith('event:')]
        streamed_id = next(data['task_id'] for event, data in events if event == 'meta')
        saved = client.get(f'/api/v1/tasks/{streamed_id}', headers=headers).json()
        assert saved['report_status'] == 'local_template' and saved['llm_report']
        assert events[-1][0] == 'done'

        missing = client.post('/api/v1/diagnostic/analyze', files=files, headers=headers)
        assert missing.status_code == 200, missing.text
        missing_id = missing.json()['id']
        unknown = client.get(f'/api/v1/tasks/{missing_id}', headers=headers).json()
        assert unknown['patient_age'] is None and unknown['patient_duration'] == '未提供'
        exported = client.get(f'/api/v1/tasks/{missing_id}/export', headers=headers).text
        assert '年龄未提供' in exported and 'None 岁' not in exported
        assert client.post('/api/v1/diagnostic/analyze', data={'patient_age':'0'}, files=files, headers=headers).status_code == 422
        missing_stream = client.post('/api/v1/diagnostic/analyze_stream', files=files, headers=headers)
        assert missing_stream.status_code == 200 and 'event: done' in missing_stream.text

        class FailedClassifier:
            async def classify(self, *args, **kwargs):
                return {'status':'failed','label':None,'candidates':[], 'reason':'invalid API response'}
        class FailedReport:
            model = 'test-api'
            async def stream_markdown(self, **kwargs):
                yield 'partial report'
                raise TimeoutError('simulated timeout')
        monkeypatch.setattr(module, 'classifier', FailedClassifier())
        monkeypatch.setattr(module, 'llm_gateway', FailedReport())
        failed = client.post('/api/v1/diagnostic/analyze_stream', data=form, files=files, headers=headers)
        assert 'event: error' in failed.text and 'event: done' in failed.text
        blocks = failed.text.strip().split('\n\n')
        task_id = json.loads(blocks[0].splitlines()[1][6:])['task_id']
        saved = client.get(f'/api/v1/tasks/{task_id}', headers=headers).json()
        assert saved['status'] == 'completed' and saved['report_status'] == 'failed'
        assert saved['classification']['status'] == 'failed'
        assert saved['llm_report'] == 'partial report'
        assert client.get(saved['mask_path']).status_code == 200

        def fail_model(*args, **kwargs):
            raise RuntimeError('simulated model failure')
        monkeypatch.setattr(module.pipeline, 'run', fail_model)
        broken = client.post('/api/v1/diagnostic/analyze_stream', data=form, files=files, headers=headers)
        assert 'event: error' in broken.text and 'event: done' in broken.text
        assert client.get('/api/v1/does-not-exist').status_code == 404
        assert client.delete(f"/api/v1/tasks/{task['id']}", headers=headers).status_code == 200
