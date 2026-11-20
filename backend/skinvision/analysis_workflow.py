"""One persisted segmentation with independently failing downstream modules."""
import hashlib
import json
import time
from datetime import datetime

import cv2
from starlette.concurrency import run_in_threadpool

from backend.skinvision.db_models import AnalysisTask
from backend.skinvision.analysis_results import save_segmentation, save_reliability
from backend.skinvision.ai.report import PROMPT_VERSION, build_report_input, report_prompt


def persist(factory, task_id, **fields):
    with factory() as session:
        task = session.get(AnalysisTask, task_id)
        if task is None:
            raise LookupError('Analysis task was removed')
        for name, value in fields.items():
            setattr(task, name, value)
        task.updated_at = datetime.utcnow()
        session.commit()


async def segment(factory, pipeline, image, directory, **task_fields):
    with factory() as session:
        task = AnalysisTask(**task_fields, status='processing', report_status='not_requested')
        session.add(task)
        session.commit()
        session.expire(task, ['original_image'])
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(pipeline.run, image)
        save_segmentation(task, result, directory, (time.perf_counter() - started) * 1000)
        task.classification_json = {'status': 'processing'}
        task.reliability_json = {'status': 'processing'}
    except Exception as exc:
        task.status = 'failed'
        task.error_message = '分割失败：' + type(exc).__name__
        result = None
    with factory() as session:
        # Reattach for a dirty-column update. A review saved while inference
        # ran must keep its version and text.
        session.add(task)
        session.commit()
    return task, result


def interrupt_pending(factory, task_id):
    # Synchronous finalizer also runs on stream disconnect/cancellation.
    with factory() as session:
        task = session.get(AnalysisTask, task_id)
        if task is None:
            return
        for name in ['classification_json', 'reliability_json']:
            value = getattr(task, name)
            if value and value.get('status') == 'processing':
                setattr(task, name, {**value, 'status': 'interrupted', 'reason': '分析连接中断'})
        # report_events finalizes its own attempt. A new retry may already
        # be running when the previous response's disconnect handler fires.
        task.updated_at = datetime.utcnow()
        session.commit()


async def additional_results(factory, task, result, image, directory, pipeline, classifier,
                             classification_view, include_reliability, include_occlusion):
    try:
        # Resize the segmentation mask back to the original image coordinates.
        mask = cv2.resize(result['mask'], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            classification = await classifier.classify(image, mask, view=classification_view)
        except Exception as exc:
            classification = {'status': 'failed', 'label': None, 'candidates': [],
                              'reason': '分类请求失败', 'error_type': type(exc).__name__}
        classification['segmentation_source'] = result['mode']
        classification['segmentation_weights_sha256'] = result['model_info'].get('weights_sha256')
        task.classification_json = classification
        persist(factory, task.id, classification_json=classification)
        yield 'classification', classification
        try:
            metadata, variance, influence = await run_in_threadpool(
                pipeline.evaluate_reliability, result, include_reliability, include_occlusion)
            reliability = save_reliability(directory, task.request_id, metadata, variance, influence)
        except Exception as exc:
            reliability = {'status': 'failed', 'reason': '可靠性计算失败', 'error_type': type(exc).__name__}
        task.reliability_json = reliability
        persist(factory, task.id, reliability_json=reliability)
        yield 'reliability', reliability
    finally:
        interrupt_pending(factory, task.id)


async def report_events(factory, task, result, gateway, variant, local_lines):
    context = {'patient_age': task.patient_age, 'symptoms': task.patient_symptoms,
               'duration': task.patient_duration, 'features': result['features'],
               'pipeline_mode': result['mode'], 'model_info': result['model_info'],
               'segmentation': result.get('segmentation') or {'status':task.status, 'mask_available':bool(task.mask_path),
                                'probability_available':bool(task.probability_path),
                                'foreground_pixels':int((result['mask'] > 0).sum()) if 'mask' in result else None},
               'classification': task.classification_json, 'reliability': task.reliability_json}
    facts = build_report_input(context)
    metadata = {'source': 'DashScope multimodal API' if gateway else 'local_template',
                'model': getattr(gateway, 'model', None), 'prompt_version': PROMPT_VERSION,
                'variant': variant if gateway else 'local_template',
                'request_settings': getattr(gateway, 'parameters', {'incremental_output': True}) if gateway else {},
                'facts': json.loads(facts), 'input_sha256': hashlib.sha256(facts.encode()).hexdigest()}
    persist(factory, task.id, report_status='processing', report_metadata_json=metadata)
    started, chunks, status = time.perf_counter(), [], 'interrupted'
    try:
        yield 'report_status', {'status': 'processing', 'metadata': metadata}
        if gateway:
            image_bytes = None
            if variant != 'text_only':
                ok, encoded = cv2.imencode('.png', result['preprocessed']['resized'])
                if not ok:
                    raise ValueError('Report image encoding failed')
                image_bytes = encoded.tobytes()
                metadata['image_sha256'] = hashlib.sha256(image_bytes).hexdigest()
            async for text in gateway.stream_markdown(prompt=report_prompt(variant), context=context,
                                                       roi_image_bytes=image_bytes):
                chunks.append(text)
                yield 'chunk', {'text': text}
            if not ''.join(chunks).strip():
                raise ValueError('Empty report response')
            status = 'completed'
        else:
            cstatus = (task.classification_json or {}).get('status', 'not_evaluated')
            rstatus = (task.reliability_json or {}).get('status', 'not_evaluated')
            for text in ['> 来源：本地规则模板；未调用报告 API。\n\n', *local_lines,
                         f'\n### 独立模块状态\n- 分割方式：{result["mode"]}\n- 分类：{cstatus}\n- 可靠性：{rstatus}\n']:
                chunks.append(text)
                yield 'chunk', {'text': text}
            status = 'local_template'
    except Exception as exc:
        status = 'failed'
        metadata['error_type'] = type(exc).__name__
        yield 'error', {'module': 'report', 'message': '报告生成失败，已保存分割结果及现有报告片段。'}
    finally:
        metadata['elapsed_ms'] = (time.perf_counter() - started) * 1000
        text = ''.join(chunks)
        if not text and status in ('failed', 'interrupted'):
            text = task.llm_report or ''
        persist(factory, task.id, llm_report=text, report_status=status,
                report_metadata_json=metadata)
    # Persistence precedes completion notification, including on API failure.
    yield 'report_status', {'status': status, 'metadata': metadata}
