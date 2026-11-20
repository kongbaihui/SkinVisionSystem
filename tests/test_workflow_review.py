"""Clinician review through the real HTTP routes using an isolated database."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from threading import Event

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def review(tmp_path, monkeypatch):
    from backend.skinvision import auth
    from backend.skinvision.ai.classification import Classifier
    from backend.skinvision.database import Base, get_db
    from backend.skinvision.db_models import User
    from backend.skinvision.pipeline import SkinVisionPipeline

    app = importlib.import_module('backend.skinvision.app')
    engine = create_engine('sqlite:///' + (tmp_path / 'review.db').as_posix())
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    def get_session():
        with factory() as session:
            yield session
    monkeypatch.setitem(app.app.dependency_overrides, get_db, get_session)
    monkeypatch.setattr(auth, 'ENABLE_AUTH', True)
    monkeypatch.setattr(auth, 'SECRET_KEY', 'isolated-case-review-secret-not-for-deployment')
    monkeypatch.setattr(app, 'TEMP_DIR', str(tmp_path))
    monkeypatch.setattr(app, 'classifier', Classifier(''))
    monkeypatch.setattr(app, 'pipeline', SkinVisionPipeline(weights_path=None, use_cpu=True))
    monkeypatch.setattr(app, 'llm_gateway', None)
    users = {}
    with factory() as session:
        for name in ('owner', 'other', 'admin'):
            user = User(username=name, password_hash='!', role='admin' if name == 'admin' else 'doctor')
            session.add(user)
            session.flush()
            users[name] = {'Authorization': 'Bearer ' + auth.create_access_token(user.id, name)}
        session.commit()
    client = TestClient(app.app)  # No lifespan creates workers or updates the application's database.
    yield app, client, factory, users
    client.close()
    engine.dispose()


def test_review_original_concurrent_edits_and_same_task_retry(review, monkeypatch):
    import asyncio
    from backend.skinvision import report_export
    from backend.skinvision.db_models import AnalysisTask
    app, client, factory, users = review
    started, release = Event(), Event()
    requests = []
    class ReportService:
        model = 'isolated-test-model'
        async def stream_markdown(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                yield 'AI partial before timeout'
                raise TimeoutError()
            started.set()
            while not release.is_set():
                await asyncio.sleep(.01)
            yield 'AI regenerated draft'
    monkeypatch.setattr(app, 'llm_gateway', ReportService())
    image = np.full((62, 93, 3), 180, dtype=np.uint8)
    cv2.circle(image, (40, 30), 15, (40, 40, 80), -1)
    original = cv2.imencode('.jpg', image)[1].tobytes()
    response = client.post('/api/v1/diagnostic/analyze_stream', headers=users['owner'],
                           data={'patient_id': 'review-test'}, files={'image': ('scan.jpg', original, 'image/jpeg')})
    assert response.status_code == 200
    meta = json.loads(response.text.split('\n\n')[0].splitlines()[1][6:])
    url = f"/api/v1/tasks/{meta['id']}"
    original_url = meta['original_path']
    assert original_url == url + '/original'
    for who in ('owner', 'admin'):
        response = client.get(original_url, headers=users[who])
        assert response.status_code == 200 and response.content == original
        assert response.headers['content-type'] == 'image/jpeg'
    for headers, status in (({}, 401), (users['other'], 403), ({'Authorization':'Bearer demo_admin_token'}, 401)):
        assert client.get(original_url, headers=headers).status_code == status
        assert client.get(url + '/export', headers=headers).status_code == status
        assert client.patch(url + '/report', headers=headers, json={'version': 0, 'text': 'edited'}).status_code == status
        assert client.post(url + '/report/retry', headers=headers).status_code == status

    def read_task():
        return client.get(url, headers=users['owner']).json()
    before = read_task()
    assert before['llm_report'] == 'AI partial before timeout' and before['report_status'] == 'failed'
    assert before['report_version'] == 0 and before['report_edited_at'] is None
    assert client.patch(url + '/report', headers=users['owner'], json={'text':'   ', 'version':0}).status_code == 422
    def save(text):
        return client.patch(url + '/report', headers=users['owner'], json={'text':text, 'version':0})
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(save, ('Clinician draft one', 'Clinician draft two')))
    assert sorted(r.status_code for r in outcomes) == [200, 409]
    winner = next(r.json() for r in outcomes if r.status_code == 200)
    assert winner['report_version'] == 1 and winner['report_edited_at']
    assert read_task()['effective_report'] == winner['edited_report']
    assert read_task()['llm_report'] == before['llm_report']

    def unexpected(*args, **kwargs):
        raise AssertionError('Retry must reuse existing segmentation and classification')
    monkeypatch.setattr(app.pipeline, 'run', unexpected)
    monkeypatch.setattr(app.classifier, 'classify', unexpected)
    with ThreadPoolExecutor(1) as pool:
        retry = pool.submit(client.post, url + '/report/retry', headers=users['admin'])
        try:
            assert started.wait(5)
            assert client.post(url + '/report/retry', headers=users['owner']).status_code == 409
            edited = client.patch(url + '/report', headers=users['owner'],
                                  json={'text':'Final clinician revision', 'version':1})
            assert edited.status_code == 200
        finally:
            release.set()
        result = retry.result(timeout=5)
    assert result.status_code == 200 and result.json() == read_task()
    task = result.json()
    assert task['id'] == before['id'] and task['request_id'] == before['request_id']
    assert task['report_status'] == 'completed' and task['llm_report'] == 'AI regenerated draft'
    assert task['edited_report'] == task['effective_report'] == 'Final clinician revision'
    assert task['report_version'] == 2 and task['classification'] == before['classification']
    assert task['reliability'] == before['reliability'] and task['features'] == before['features']
    assert requests[0] == requests[1]
    assert client.post(url + '/report/retry', headers=users['owner']).status_code == 409
    history = client.get('/api/v1/tasks', headers=users['owner']).json()['data']
    assert len(history) == 1 and history[0]['patient_id'] == 'review-test'
    assert history[0]['risk_level'] == task['risk_level'] == meta['risk_level']
    assert task['risk_source'] == meta['risk_source'] == history[0]['risk_source']
    markdown = client.get(url + '/export', headers=users['owner']).text
    assert 'Final clinician revision' in markdown and 'AI regenerated draft' not in markdown
    assert f"既有规则提示：{task['risk_level']}" in markdown and '版本 2' in markdown
    pdf_inputs = []
    render = report_export.export_pdf
    def capture(markdown_text):
        pdf_inputs.append(markdown_text)
        return render(markdown_text)
    monkeypatch.setattr(report_export, 'export_pdf', capture)
    pdf = client.get(url + '/export?format=pdf', headers=users['owner'])
    assert pdf.status_code == 200 and pdf.content.startswith(b'%PDF') and pdf_inputs == [markdown]
    with factory() as session:
        assert session.query(AnalysisTask).count() == 1


def test_legacy_review_without_original(review):
    from backend.skinvision.db_models import AnalysisTask, User
    _, client, factory, users = review
    with factory() as session:
        task = AnalysisTask(user_id=session.query(User).filter_by(username='owner').one().id,
                            status='completed', llm_report='Historical draft', irregularity_index=1.1,
                            color_variance=0.1)
        session.add(task)
        session.commit()
        url = f'/api/v1/tasks/{task.id}'
    detail = client.get(url, headers=users['owner']).json()
    assert detail['original_path'] is None and detail['effective_report'] == 'Historical draft'
    assert detail['features']['irregularity_index'] == 1.1 and detail['risk_level'] == '中风险'
    assert client.get(url + '/original', headers=users['owner']).status_code == 404
    assert client.post(url + '/report/retry', headers=users['admin']).status_code == 409
    response = client.patch(url + '/report', headers=users['admin'], json={'text':'Legacy review', 'version':0})
    assert response.status_code == 200 and response.json()['llm_report'] == 'Historical draft'


def test_failed_retry_keeps_saved_review_and_previous_partial(review, monkeypatch):
    from backend.skinvision.db_models import AnalysisTask, User
    app, client, factory, users = review
    with factory() as session:
        task = AnalysisTask(user_id=session.query(User).filter_by(username='owner').one().id,
                            status='completed', report_status='interrupted', edited_report='Saved review',
                            report_version=4, llm_report='Prior partial', report_metadata_json={'variant':'text_only'})
        session.add(task)
        session.commit()
        url = f'/api/v1/tasks/{task.id}/report/retry'
    class EmptyGateway:
        async def stream_markdown(self, **kwargs):
            yield ''
    monkeypatch.setattr(app, 'llm_gateway', EmptyGateway())
    response = client.post(url, headers=users['owner'])
    assert response.status_code == 200
    detail = response.json()
    assert detail['report_status'] == 'failed' and detail['llm_report'] == 'Prior partial'
    assert detail['effective_report'] == 'Saved review' and detail['report_version'] == 4


def test_local_demo_identities_do_not_own_registered_cases(review, monkeypatch):
    from backend.skinvision import auth
    from backend.skinvision.db_models import AnalysisTask, User
    _, client, factory, users = review
    monkeypatch.setattr(auth, 'ENABLE_AUTH', False)
    demo = {'Authorization':'Bearer demo_access_token'}
    demo_admin = {'Authorization':'Bearer demo_admin_token'}
    demo_id = client.get('/api/v1/auth/me', headers=demo).json()['id']
    with factory() as session:
        owner = session.query(User).filter_by(username='owner').one()
        registered = AnalysisTask(user_id=owner.id, status='completed', original_image=b'case', original_media_type='image/png')
        local = AnalysisTask(user_id=demo_id, status='completed', original_image=b'demo', original_media_type='image/png')
        session.add_all([registered, local])
        session.commit()
        registered_url = f'/api/v1/tasks/{registered.id}/original'
        local_url = f'/api/v1/tasks/{local.id}/original'
    for who in (demo, demo_admin):
        assert client.get(registered_url, headers=who).status_code == 403
        assert client.get(local_url, headers=who).content == b'demo'
    assert client.get(registered_url).status_code == 401
    assert client.get(registered_url, headers=users['owner']).content == b'case'
