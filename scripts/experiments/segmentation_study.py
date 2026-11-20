"""Two normalization conditions on frozen IDs; select only on calibration data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from backend.skinvision.runtime import load_project_env, resolve_path
from backend.skinvision.pipeline import SkinVisionPipeline
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline, normalize_like_training
from backend.skinvision.evaluation.metrics import segmentation_metrics


def read_image(path, grayscale=False):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',default='outputs/research-v1/protocol-v1.json')
    parser.add_argument('--outdir',default='outputs/research-v1/segmentation')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    directory=Path(args.outdir);directory.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    load_project_env()
    pipeline=SkinVisionPipeline(resolve_path(os.getenv('SKINVISION_WEIGHTS','')),use_cpu=True)
    if not pipeline.ensure_model_loaded():
        raise RuntimeError('Real checkpoint required, no fallback experiment')
    path=Path(args.protocol);protocol=json.loads(path.read_text(encoding='utf-8'))
    configuration={'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'weights_sha256':pipeline.weights_sha256,'device':'cpu','threads':args.threads,
        'threshold':pipeline.threshold,'conditions':protocol['segmentation_comparison'],
        'torch_version':torch.__version__,'metric_resolution':[256,256],
        'mask_resize':'nearest','prediction':'mean sigmoid of four heads',
        'limitations':protocol['limitations']}
    (directory/'config.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for index,row in enumerate(protocol['segmentation']):
        image=read_image(row['image']);truth=read_image(row['mask'],True)
        if image is None or truth is None:
            raise ValueError('Unreadable source input')
        prep=preprocess_pipeline(image)
        truth=cv2.resize(truth,(256,256),interpolation=cv2.INTER_NEAREST)>127
        case_dir=directory/row['id'];case_dir.mkdir()
        cv2.imencode('.png',prep['resized'])[1].tofile(str(case_dir/'processed.png'))
        cv2.imencode('.png',truth.astype(np.uint8)*255)[1].tofile(str(case_dir/'truth.png'))
        np.save(case_dir/'truth.npy',truth,allow_pickle=False)
        for condition in configuration['conditions']:
            chw=prep['chw'] if condition=='rgb_minus1_1' else normalize_like_training(prep['resized'])
            result={'id':row['id'],'split':row['split'],'condition':condition,'status':'failed'}
            started=time.perf_counter()
            try:
                probability,mask=pipeline._infer_by_model(chw)
                result.update(status='completed',**segmentation_metrics(probability,truth))
                np.save(case_dir/(condition+'.npy'),probability,allow_pickle=False)
                cv2.imencode('.png',mask)[1].tofile(str(case_dir/(condition+'.png')))
            except Exception as exc:
                result['error_type']=type(exc).__name__
            result['inference_ms']=(time.perf_counter()-started)*1000
            records.append(result)
            with (directory/'predictions.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(f'{index+1}/{len(protocol["segmentation"])} {row["split"]} {row["id"]}',flush=True)
        if index==7:
            means={condition:np.mean([r['dice'] for r in records if r['condition']==condition and r['status']=='completed'])
                   for condition in configuration['conditions']}
            chosen=max(configuration['conditions'],key=lambda condition:means[condition])
            (directory/'calibration-choice.json').write_text(json.dumps({'normalization':chosen,
                'calibration_mean_dice':means,'n':8,'selection':'calibration only; before any test prediction'},indent=2))
            print('Calibration selected '+chosen,flush=True)
    aggregates=[]
    for split in ['calibration','test']:
        for condition in configuration['conditions']:
            selected=[r for r in records if r['split']==split and r['condition']==condition]
            valid=[r for r in selected if r['status']=='completed']
            aggregates.append({'split':split,'condition':condition,'n':len(selected),'completed':len(valid),
                'mean_dice':float(np.mean([r['dice'] for r in valid])) if valid else None,
                'mean_iou':float(np.mean([r['iou'] for r in valid])) if valid else None,
                'mean_inference_ms':float(np.mean([r['inference_ms'] for r in valid])) if valid else None})
    (directory/'summary.json').write_text(json.dumps({'results':aggregates,'limitations':protocol['limitations']},ensure_ascii=False,indent=2),encoding='utf-8')
    failures=[r for r in records if r['status']!='completed' or r.get('dice',0)<.7]
    (directory/'failure-cases.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(aggregates),flush=True)


if __name__=='__main__':
    main()
