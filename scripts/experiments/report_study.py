"""Three real API report conditions; preserve raw prose for technical claim audit."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from backend.skinvision.runtime import load_project_env
from backend.skinvision.ai.report import build_report_input, report_prompt, PROMPT_VERSION
from backend.skinvision.ai.local_classification import LocalClassifier
from backend.skinvision.llm_gateway import LLMGateway
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline
from backend.skinvision.metrics.lesion_metrics import extract_quant_features
from scripts.experiments.segmentation_study import read_image


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='outputs/research-v1/protocol-v1.json')
    parser.add_argument('--outdir',default='outputs/research-v1/reports')
    parser.add_argument('--segmentation',default='outputs/research-v1/segmentation')
    parser.add_argument('--reliability',default='outputs/research-v1/reliability')
    args=parser.parse_args()
    load_project_env();torch.set_num_threads(2)
    key=os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIYUN_API_KEY')
    if not key:
        raise RuntimeError('Real API key required')
    directory=Path(args.outdir);directory.mkdir(parents=True,exist_ok=False)
    protocol_path=Path(args.protocol);protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    segmentation=Path(args.segmentation)
    selected=json.loads((segmentation/'calibration-choice.json').read_text())['normalization']
    seg_config=json.loads((segmentation/'config.json').read_text(encoding='utf-8'))
    gateway=LLMGateway(key,timeout=90,model=os.getenv('SKINVISION_LLM_MODEL','qwen3-vl-plus'),temperature=0.,max_tokens=1000)
    classifier=LocalClassifier(os.getenv('SKINVISION_CLASSIFICATION_DIR', 'weights/classification/efficientnet-b0-research'))
    config={'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'model':gateway.model,'prompt_version':PROMPT_VERSION,'request_settings':gateway.parameters,
            'conditions':protocol['report']['conditions'],'segmentation_normalization':selected,
            'note':'8 cases × 3 conditions. Two deliberate fault-state cases are explicitly synthetic; no clinical validity claim.'}
    (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    cases=[row for row in protocol['segmentation'] if row['split']=='test'][:6]
    all_records=[]
    for index in range(8):
        row=cases[index] if index<6 else cases[index-6]
        record_id=row['id'] if index<6 else row['id']+'-fault-state'
        reliability_file=Path(args.reliability)/'predictions.jsonl'
        # Independent worker may still be producing the fixed cached results.
        waiting=time.monotonic()
        while True:
            rows=[json.loads(line) for line in reliability_file.read_text(encoding='utf-8').splitlines() if line.strip()] if reliability_file.exists() else []
            match=next((value for value in rows if value['id']==row['id']),None)
            if match:
                break
            if time.monotonic()-waiting>3600:
                raise TimeoutError('Fixed reliability case not available')
            await asyncio.sleep(10)
        image=read_image(row['image']);prep=preprocess_pipeline(image)
        probability=np.load(segmentation/row['id']/(selected+'.npy'),allow_pickle=False)
        mask=(probability>=.5).astype(np.uint8)*255
        classification=classifier._classify(image,cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST),'full')
        context={'patient_age':None,'symptoms':'公开研究图像，没有临床症状记录','duration':None,
                 'features':extract_quant_features(prep['resized'],mask),'pipeline_mode':'model',
                 'model_info':{'name':'LMDAUNet','execution':'model','normalization':selected,
                               'version':'lmdaunet-portable-v1','training_dataset':'unverified',
                               'weights_sha256':seg_config['weights_sha256']},
                 'segmentation':{'status':'completed','mask_available':True,'probability_available':True,
                                 'foreground_pixels':int((mask>0).sum())},
                 'classification':classification,'reliability':match}
        if index>=6:
            context['classification']={'status':'failed','label':None,'candidates':[]}
            context['reliability']={'status':'not_available'}
            context['pipeline_mode']='fallback'
            context['model_info']={'name':'Otsu','execution':'fallback','training_dataset':'not_applicable'}
            # Fault injection is a report consistency test, not a real segmentation run.
        case_dir=directory/record_id;case_dir.mkdir()
        (case_dir/'facts.json').write_text(build_report_input(context),encoding='utf-8')
        (case_dir/'source.json').write_text(json.dumps({'id':row['id'],'synthetic_fault_state':index>=6,
            'image_sha256':row['image_sha256']},indent=2))
        image_bytes=cv2.imencode('.png',prep['resized'])[1].tobytes()
        for variant in config['conditions']:
            prompt=report_prompt(variant)+'报告控制在350个汉字以内；保持五个指定小节，缺少资料应说明。'
            (case_dir/(variant+'-prompt.txt')).write_text(prompt,encoding='utf-8')
            chunks=[];started=time.perf_counter();status='completed';error=None
            try:
                async for chunk in gateway.stream_markdown(prompt=prompt,context=context,
                        roi_image_bytes=None if variant=='text_only' else image_bytes):
                    chunks.append(chunk)
            except Exception as exc:
                status='failed';error=type(exc).__name__
            text=''.join(chunks)
            (case_dir/(variant+'.md')).write_text(text,encoding='utf-8')
            record={'case':record_id,'variant':variant,'status':status,'error_type':error,
                    'elapsed_ms':(time.perf_counter()-started)*1000,'characters':len(text),
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'facts_sha256':hashlib.sha256(build_report_input(context).encode()).hexdigest(),
                    'image_sent':variant!='text_only','image_sha256':hashlib.sha256(image_bytes).hexdigest() if variant!='text_only' else None}
            all_records.append(record)
            with (directory/'requests.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(record_id+' '+variant+' '+status,flush=True)
    (directory/'requests-summary.json').write_text(json.dumps(all_records,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    asyncio.run(main())
