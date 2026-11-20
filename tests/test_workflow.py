import asyncio
import pytest


def test_interrupted_report_keeps_partial_text(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.skinvision.database import Base
    from backend.skinvision.db_models import AnalysisTask
    from backend.skinvision.analysis_workflow import report_events
    engine = create_engine('sqlite:///' + str(tmp_path / 'stream.db'))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        task = AnalysisTask(status='completed', patient_age=30, patient_symptoms='synthetic',
                            patient_duration='test', classification_json={'status': 'failed'},
                            reliability_json={'status': 'not_available'})
        session.add(task)
        session.commit()
    class Gateway:
        model = 'test-double'
        async def stream_markdown(self, **kwargs):
            yield 'saved partial text'
            raise AssertionError('Must be cancelled before continuing')
    async def consume():
        stream = report_events(factory, task, {'features': {}, 'mode':'fallback', 'model_info':{}},
                               Gateway(), 'text_only', [])
        assert (await anext(stream))[0] == 'report_status'
        assert (await anext(stream))[0] == 'chunk'
        await stream.aclose()
    asyncio.run(consume())
    with factory() as session:
        saved = session.get(AnalysisTask, task.id)
        assert saved.status == 'completed'
        assert saved.report_status == 'interrupted'
        assert saved.llm_report == 'saved partial text'
        assert saved.report_metadata_json['model'] == 'test-double'


@pytest.mark.parametrize('spec_version', ['2.0', '2.4'])
@pytest.mark.parametrize('disconnect_phase', ['headers', 'chunk'])
def test_http_disconnect_closes_nested_report(tmp_path, monkeypatch, spec_version, disconnect_phase):
    import io
    import cv2
    import numpy as np
    import importlib
    from fastapi import UploadFile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.skinvision.database import Base
    from backend.skinvision.db_models import AnalysisTask
    from backend.skinvision.pipeline import SkinVisionPipeline
    from backend.skinvision.ai.classification import Classifier
    module = importlib.import_module('backend.skinvision.app')
    engine = create_engine('sqlite:///' + str(tmp_path / 'disconnect.db'))
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(module, 'TEMP_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'pipeline', SkinVisionPipeline(weights_path=None, use_cpu=True))
    monkeypatch.setattr(module, 'classifier', Classifier(''))
    class Gateway:
        model = 'test-double'
        async def stream_markdown(self, **kwargs):
            yield 'durable partial'
            await asyncio.sleep(10)
    monkeypatch.setattr(module, 'llm_gateway', Gateway())
    async def consume():
        image = cv2.imencode('.png', np.full((32,32,3), 128, np.uint8))[1].tobytes()
        with factory() as session:
            response = await module.analyze_stream(request=None, image=UploadFile(io.BytesIO(image)),
                patient_age=30, symptoms='test', duration='test', patient_id=None,
                classification_view='full', include_reliability=False, include_occlusion=False,
                report_variant='text_only', user_id=None, db=session)
        disconnected = asyncio.Event()
        async def send(message):
            if ((disconnect_phase == 'headers' and message['type'] == 'http.response.start') or
                    (disconnect_phase == 'chunk' and b'event: chunk' in message.get('body', b''))):
                disconnected.set()
                if spec_version == '2.4':
                    raise OSError('client disconnected')
                await asyncio.sleep(10)
        async def receive():
            await disconnected.wait()
            return {'type':'http.disconnect'}
        try:
            await asyncio.wait_for(response({'type':'http', 'asgi':{'spec_version':spec_version}}, receive, send), 2)
        except Exception as exc:
            from starlette.requests import ClientDisconnect
            assert isinstance(exc, ClientDisconnect), type(exc)
        # Assert immediately: no GC, loop teardown, or explicit aclose in the test.
        with factory() as session:
            saved = session.query(AnalysisTask).one()
            assert saved.status == 'completed'
            if disconnect_phase == 'chunk':
                assert saved.report_status == 'interrupted'
                assert saved.llm_report == 'durable partial'
            else:
                assert saved.classification_json['status'] == 'interrupted'
                assert saved.reliability_json['status'] == 'interrupted'
    asyncio.run(consume())
