"""TTA and occlusion study against actual masks; optional cached reference maps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.stats import spearmanr
from backend.skinvision.runtime import load_project_env, resolve_path
from backend.skinvision.pipeline import SkinVisionPipeline
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline, normalize_like_training
from backend.skinvision.ai.reliability import analyze_reliability, occlusion_sensitivity
from backend.skinvision.evaluation.metrics import segmentation_metrics, reliability_metrics
from scripts.experiments.segmentation_study import read_image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='outputs/research-v1/protocol-v1.json')
    parser.add_argument('--outdir',default='outputs/research-v1/reliability')
    parser.add_argument('--segmentation',default='outputs/research-v1/segmentation')
    parser.add_argument('--normalization',choices=['rgb_minus1_1','train_minmax_255'])
    parser.add_argument('--occlusion-count',type=int,default=8)
    args=parser.parse_args()
    source=Path(args.segmentation)
    normalization=args.normalization or json.loads((source/'calibration-choice.json').read_text())['normalization']
    directory=Path(args.outdir);directory.mkdir(parents=True,exist_ok=False)
    path=Path(args.protocol);protocol=json.loads(path.read_text(encoding='utf-8'))
    load_project_env();torch.set_num_threads(4)
    pipeline=SkinVisionPipeline(resolve_path(os.getenv('SKINVISION_WEIGHTS','')),use_cpu=True,normalization=normalization)
    if not pipeline.ensure_model_loaded():
        raise RuntimeError('Real segmentation required')
    configuration={'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'normalization':normalization,
                   'weights_sha256':pipeline.weights_sha256,'version':'tta-flip-rotate-v1',
                   'threshold':.5,'low_quality_dice_threshold':.7,'timing':'shared Windows CPU host, not isolated benchmark',
                   'occlusion_fill':0.,'occlusion_grid':2,'occlusion_count':args.occlusion_count,'gradcam_score':'sigmoid interpolated out_head1 mean',
                   'limitations':protocol['limitations']}
    (directory/'config.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    rows=[row for row in protocol['segmentation'] if row['split']=='test']
    for index,row in enumerate(rows):
        record={'id':row['id'],'status':'failed'}
        try:
            prep=preprocess_pipeline(read_image(row['image']))
            chw=prep['chw'] if normalization=='rgb_minus1_1' else normalize_like_training(prep['resized'])
            cached=source/row['id']/(normalization+'.npy')
            reference=np.load(cached,allow_pickle=False) if cached.exists() else pipeline._infer_by_model(chw)[0]
            truth=cv2.resize(read_image(row['mask'],True),(256,256),interpolation=cv2.INTER_NEAREST)>127
            metrics=segmentation_metrics(reference,truth)
            predict=lambda x:pipeline._infer_by_model(x)[0]
            metadata,variance=analyze_reliability(predict,chw,reference=reference)
            np.save(directory/(row['id']+'-variance.npy'),variance,allow_pickle=False)
            record.update(**metadata,**metrics)
            if index<args.occlusion_count:
                occlusion,influence=occlusion_sensitivity(predict,chw,grid=2,reference=reference)
                record['occlusion']=occlusion
                np.save(directory/(row['id']+'-occlusion.npy'),influence,allow_pickle=False)
                attention,overlay=pipeline._generate_gradcam(chw,prep['resized'],return_map=True)
                record['gradcam_status']='completed' if overlay is not None else 'failed'
                if overlay is not None:
                    cv2.imencode('.png',overlay)[1].tofile(str(directory/(row['id']+'-gradcam.png')))
                    np.save(directory/(row['id']+'-gradcam.npy'),attention,allow_pickle=False)
                    attention_means=[]
                    changes=[]
                    for effect in occlusion['effects']:
                        x0,y0,x1,y1=effect['bbox_xyxy']
                        attention_means.append(float(attention[y0:y1,x0:x1].mean()))
                        changes.append(effect['mean_probability_change'])
                    record['explanation_check']={'grid_attention_mean':attention_means,
                        'occlusion_prediction_change':changes,
                        'spearman':float(spearmanr(attention_means,changes).statistic) if np.ptp(attention_means)>0 and np.ptp(changes)>0 else None,
                        'limitation':'Only four blocks per image; descriptive faithfulness probe, not clinical validation'}
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__)
        records.append(record)
        with (directory/'predictions.jsonl').open('a',encoding='utf-8') as file:
            file.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f'{index+1}/{len(rows)} {row["id"]} {record["status"]}',flush=True)
    summary=reliability_metrics(records)
    summary['mean_extra_inference_ms']=float(np.mean([row['elapsed_ms'] for row in records if row['status']=='completed']))
    summary['limitations']=protocol['limitations']
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
