"""Frozen 70-case paired full-image/segmentation-ROI classification experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import os
import cv2
import numpy as np
import torch
from backend.skinvision.runtime import load_project_env, resolve_path
from backend.skinvision.pipeline import SkinVisionPipeline
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline, normalize_like_training
from backend.skinvision.ai.local_classification import LocalClassifier
from backend.skinvision.evaluation.metrics import classification_metrics
from scripts.experiments.segmentation_study import read_image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='outputs/research-v1/protocol-v1.json')
    parser.add_argument('--outdir',default='outputs/research-v1/classification')
    parser.add_argument('--calibration',default='outputs/research-v1/segmentation/calibration-choice.json')
    args=parser.parse_args()
    normalization=json.loads(Path(args.calibration).read_text())['normalization']
    directory=Path(args.outdir);directory.mkdir(parents=True,exist_ok=False)
    protocol_path=Path(args.protocol);protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    load_project_env();torch.set_num_threads(4)
    pipeline=SkinVisionPipeline(resolve_path(os.getenv('SKINVISION_WEIGHTS','')),use_cpu=True,normalization=normalization)
    if not pipeline.ensure_model_loaded():
        raise RuntimeError('Real segmentation required')
    classifier=LocalClassifier(os.getenv('SKINVISION_CLASSIFICATION_DIR', 'weights/classification/efficientnet-b0-research'))
    config={'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'segmentation_weights_sha256':pipeline.weights_sha256,'normalization':normalization,
            'source':'Official ISIC2018 Task3 test, 10 frozen samples per class',
            'roi':'Predicted segmentation bounding box, 10% padding; empty mask is review_required',
            'limitations':protocol['limitations'],'timing':'Shared Windows CPU host; not a controlled speed benchmark'}
    (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for index,row in enumerate(protocol['classification']):
        if hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('Input changed after protocol freeze')
        image=read_image(row['image']);started=time.perf_counter()
        prep=preprocess_pipeline(image)
        chw=prep['chw'] if normalization=='rgb_minus1_1' else normalize_like_training(prep['resized'])
        segmentation_status='completed'
        try:
            probability,mask=pipeline._infer_by_model(chw)
            np.save(directory/(row['id']+'-prob.npy'),probability,allow_pickle=False)
            cv2.imencode('.png',mask)[1].tofile(str(directory/(row['id']+'-mask.png')))
            mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        except Exception:
            segmentation_status='failed';mask=None
        segmentation_ms=(time.perf_counter()-started)*1000
        for view in ['full','roi']:
            result=classifier._classify(image,mask,view)
            record={'id':row['id'],'truth':row['truth'],'view':view,'result':result,
                    'segmentation_status':segmentation_status,'segmentation_ms':segmentation_ms}
            records.append(record)
            with (directory/'predictions.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f'{index+1}/70 {row["id"]}',flush=True)
    summary={view:classification_metrics([r for r in records if r['view']==view]) for view in ['full','roi']}
    summary['limitations']=protocol['limitations']
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    failures=[r for r in records if r['result']['status']!='completed' or r['result']['label']!=r['truth']]
    (directory/'failure-cases.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({view:{key:summary[view][key] for key in ['n','macro_f1','accuracy','coverage']} for view in ['full','roi']}),flush=True)


if __name__=='__main__':
    main()
