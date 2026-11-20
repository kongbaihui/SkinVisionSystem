"""Explicit real service acceptance on synthetic input; no disease accuracy claim."""
import argparse
import json
import os
from pathlib import Path
import secrets
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--real-api', action='store_true', help='Send generated gray pixels and synthetic metadata to configured API')
    parser.add_argument('--classification-backend', choices=['api', 'local'], default='api')
    parser.add_argument('--image',type=Path,help='Optional public research image; never supplies a disease ground truth')
    args = parser.parse_args()
    directory = Path(args.outdir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from backend.skinvision.runtime import load_project_env
    load_project_env()
    os.environ['SKINVISION_CLASSIFICATION_BACKEND'] = args.classification_backend
    if not args.real_api:
        os.environ['DASHSCOPE_API_KEY'] = os.environ['ALIYUN_API_KEY'] = ''
    elif not (os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIYUN_API_KEY')):
        raise RuntimeError('API key must be configured in backend environment')
    os.environ['SKINVISION_DB_URL'] = 'sqlite:///' + (directory / 'acceptance.db').as_posix()
    os.environ['SKINVISION_TEMP_DIR'] = str(directory / 'images')
    os.environ['SKINVISION_ENABLE_AUTH'] = 'true'
    os.environ['SKINVISION_AUTH_SECRET'] = secrets.token_urlsafe(48)
    import cv2
    import numpy as np
    import torch
    from fastapi.testclient import TestClient
    from backend.skinvision.app import app
    torch.set_num_threads(4)
    if args.image:
        from backend.skinvision.image_io import read_image
        content=cv2.imencode('.png',read_image(args.image))[1].tobytes()
        symptoms='公开皮肤镜研究图像；年龄、症状、病程未提供'
    else:
        content = cv2.imencode('.png', np.full((256,256,3), 128, np.uint8))[1].tobytes()
        symptoms='程序生成的灰色测试图片，无患者，无真实症状'
    (directory / 'input.png').write_bytes(content)
    started = time.perf_counter()
    with TestClient(app) as client:
        credentials = {'username':'acceptance_test', 'password':secrets.token_urlsafe(24) + 'Aa1'}
        assert client.post('/api/v1/auth/register', json=credentials).status_code == 201
        token = client.post('/api/v1/auth/login', json=credentials).json()['access_token']
        headers = {'Authorization':'Bearer ' + token}
        response = client.post('/api/v1/diagnostic/analyze_stream', headers=headers,
            data={'symptoms':symptoms, 'include_reliability':'true', 'include_occlusion':'false'},
            files={'image':('synthetic-gray.png',content,'image/png')})
        assert response.status_code == 200
        (directory / 'response.sse').write_text(response.text, encoding='utf-8')
        events = [(block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:]))
                  for block in response.text.strip().split('\n\n')]
        task_id = next(data['task_id'] for event,data in events if event == 'meta')
        task = client.get(f'/api/v1/tasks/{task_id}', headers=headers).json()
        (directory / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
        for fmt, extension in [('markdown','md'),('pdf','pdf')]:
            export = client.get(f'/api/v1/tasks/{task_id}/export?format={fmt}', headers=headers)
            assert export.status_code == 200, export.text[:100]
            (directory / f'report.{extension}').write_bytes(export.content)
        for field in ['original_path','mask_path','roi_path','gradcam_path','overlay_path','probability_path']:
            assert task[field] and client.get(task[field], headers=headers).status_code == 200, field
        assert task['pipeline_mode'] == 'model' and task['status'] == 'completed'
        assert task['reliability']['status'] == 'completed'
        assert task['report_status'] == ('completed' if args.real_api else 'local_template')
        summary = {'kind':'public image integration acceptance, not diagnostic accuracy' if args.image else 'synthetic integration acceptance, not diagnostic accuracy',
                   'elapsed_ms':(time.perf_counter()-started)*1000, 'events':[event for event,_ in events],
                   'pipeline_mode':task['pipeline_mode'], 'model_info':task['model_info'],
                   'classification_status':task['classification']['status'],
                   'reliability':task['reliability'], 'report_status':task['report_status'],
                   'exports':['markdown','pdf'], 'real_api':args.real_api}
        (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key:summary[key] for key in ['kind','elapsed_ms','pipeline_mode','classification_status','report_status','real_api']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
