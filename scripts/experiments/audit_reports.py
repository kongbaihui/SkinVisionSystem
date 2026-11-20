"""Transparent post-hoc technical audit; no medical correctness scoring."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

NUMBER = r'[0-9]+(?:\.[0-9]+)?(?:e[-−]?[0-9]+)?'
FIELDS = {
    'area_ratio': r'(?:面积占比|面积比|病灶区域占比)[^0-9。；]{0,8}('+NUMBER+r')',
    'irregularity_index': r'(?:不规则度指数|不规则指数|不规则度|irregularity_index)[^0-9。；]{0,10}('+NUMBER+r')',
    'asymmetry_index': r'(?:不对称性|不对称指数|(?<!不)对称性|asymmetry_index)[^0-9。；]{0,10}('+NUMBER+r')',
    'border_sharpness': r'(?:边界锐度值|边界锐度|锐度|border_sharpness)[^0-9。；]{0,10}('+NUMBER+r')',
    'max_diameter_ratio': r'(?:最大直径比|最大直径占比)[^0-9。；]{0,8}('+NUMBER+r')',
}


def audit(root):
    request_rows=json.loads((root/'requests-summary.json').read_text(encoding='utf-8'))
    records=[]
    for request in request_rows:
        case, variant=request['case'],request['variant']
        directory=root/case
        facts=json.loads((directory/'facts.json').read_text(encoding='utf-8'))
        text=(directory/(variant+'.md')).read_text(encoding='utf-8')
        citations=[]
        # Five common morphometry fields are the declared numeric denominator.
        # Excludes softmax/model versions, TTA counters and pixel counts.
        for field, pattern in FIELDS.items():
            for match in re.finditer(pattern,text):
                literal=match.group(1).replace('−','-'); value=float(literal)
                expected=facts['features'][field]
                suffix=text[match.end():match.end()+2]
                if field=='max_diameter_ratio' and suffix.startswith('%'):
                    expected*=100
                precision=len(literal.partition('.')[2].split('e')[0]) if '.' in literal else 0
                tolerance=.5*10**(-precision)+1e-9
                correct=abs(value-expected)<=tolerance
                reasons=[]
                if not correct:
                    reasons.append('outside stated rounding precision')
                if field=='area_ratio' and not suffix.startswith('%'):
                    correct=False;reasons.append('area percentage missing percent unit')
                if field=='asymmetry_index' and re.search(r'(?<!不)对称性',match.group(0)) and '不对称指数' not in match.group(0):
                    correct=False;reasons.append('asymmetry value mislabeled symmetry')
                citations.append({'field':field,'quote':match.group(0)+suffix,'value':value,'expected':expected,
                                  'correct':correct,'reasons':reasons})
        sections={}
        for piece in re.split(r'^###\s+',text,flags=re.M)[1:]:
            title,_,body=piece.partition('\n');sections[title.strip()]=body.strip()
        units=[]
        for heading,body in sections.items():
            for sentence in re.split(r'[。；;\n]+',body):
                sentence=sentence.strip()
                if not sentence:continue
                reasons=[]
                # Any unsupported item makes this entire sentence/clause fail once.
                if heading=='影像特征' and re.search(r'低|高|中等|轻度|良好|较对称|极小',sentence):
                    reasons.append('qualitative level without a supplied validated threshold')
                if re.search(r'DF.{0,14}脂溢',sentence):
                    reasons.append('DF label mistranslation; correct name dermatofibroma')
                if re.search(r'非真实临床场景|无真实临床背景|图像非真实|未关联真实病例验证',sentence):
                    reasons.append('public research origin does not establish nonclinical provenance')
                if '掩码覆盖率' in sentence:
                    reasons.append('TTA coordinate coverage mislabeled mask coverage')
                if '置信度较高' in sentence:
                    reasons.append('uncalibrated TTA variance promoted to confidence')
                if heading=='模型可靠性' and '属未校准softmax分数' in sentence:
                    reasons.append('TTA variance confused with classifier scores')
                if heading=='辅助摘要' and re.search(r'影像.*(?:偏良性|倾向良性|良性表征|良性表现)|图像特征倾向良性|影像参数提示低风险',sentence):
                    reasons.append('disease/risk inference not supported by supplied feature thresholds')
                if facts['classification']['status']=='failed' and '未执行疾病分类' in sentence:
                    reasons.append('failed status does not establish classification was never executed')
                if heading=='影像特征' and re.search(r'(?<!不)对称性',sentence):
                    reasons.append('asymmetry metric relabeled as symmetry')
                units.append({'section':heading,'text':sentence,'unsupported_or_contradicted':bool(reasons),'reasons':reasons})
        checklist={
            'five_required_sections':all(sections.get(x) for x in ['影像特征','分类候选','模型可靠性','缺失信息','辅助摘要']),
            'area_feature_present':bool(re.search(FIELDS['area_ratio'],text)),
            'classification_state_or_candidate':bool(re.search(r'NV|MEL|分类.*(?:失败|failed)',sections.get('分类候选',''))),
            'source_model_or_fallback':bool(re.search(r'EfficientNet|Otsu',text)),
            'reliability_result_or_missing':bool(re.search(r'方差|不可用|not_available',sections.get('模型可靠性',''))),
            'nonclinical_boundary':bool(re.search(r'非.*(?:临床|医学|诊断)|不可.*(?:诊断|确诊|临床)|不能用于诊断|无法支持.*(?:诊断|判断)|不具诊断效力|仅为辅助研究用途',text)),
            'segmentation_completed_or_available':bool(re.search(r'分割.*完成|掩膜可用|掩码可用|前景像素|分割处理|完成分割|掩膜存在',text)),
            'clinical_information_missing':all(word in sections.get('缺失信息','') for word in ['年龄','症状','病程']),
        }
        records.append({**request,'report_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'numeric_citations':citations,'information_checklist':checklist,'statement_units':units,
                        'fault_state_correct': all(re.search(p,text) for p in [r'失败|failed',r'不可用|not_available',r'Otsu']) if 'fault-state' in case else None})
    result={'rubric_version':'technical-audit-v1-posthoc','evaluation_type':'Post-hoc automated checks on generated reports; no clinician review',
            'numeric_denominator':'All detected mentions of five common morphometry fields: area, irregularity, asymmetry, border sharpness, diameter ratio. Unit/semantic mismatch counts incorrect. Other numerical fields are outside this quantitative score.',
            'statement_denominator':'Nonempty clauses split on Chinese full stop, semicolon or newline within the five sections. A clause with any unsupported technical assertion or contradiction fails once. Pure image/medical truth is not evaluated; the ratio is a lower-bound technical error screen, not total hallucination rate.',
            'completeness_denominator':'Eight explicitly saved information slots per report; presence separate from correctness.',
            'rubric_timing':'Written after seeing outputs; exploratory, potential evaluator bias; repeat with blinded dual human annotation before formal conclusions',
            'records':records,'summary':{}}
    for variant in ['text_only','image_text','constrained']:
        rows=[row for row in records if row['variant']==variant]
        cites=[x for row in rows for x in row['numeric_citations']]
        units=[x for row in rows for x in row['statement_units']]
        flags=[value for row in rows for value in row['information_checklist'].values()]
        faults=[row['fault_state_correct'] for row in rows if row['fault_state_correct'] is not None]
        result['summary'][variant]={'n':len(rows),'completed':sum(row['status']=='completed' for row in rows),
            'numeric_correct':sum(x['correct'] for x in cites),'numeric_total':len(cites),
            'numeric_correct_rate':sum(x['correct'] for x in cites)/len(cites) if cites else None,
            'complete_slots':sum(flags),'total_slots':len(flags),'completeness':sum(flags)/len(flags) if flags else None,
            'unsupported_units':sum(x['unsupported_or_contradicted'] for x in units),'total_units':len(units),
            'unsupported_statement_ratio':sum(x['unsupported_or_contradicted'] for x in units)/len(units) if units else None,
            'fault_state_correct':sum(faults),'fault_state_total':len(faults),
            'mean_generation_ms':float(np.mean([row['elapsed_ms'] for row in rows])) if rows else None}
    destination=root.parent / 'report-audit-v1.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=Path('outputs/research-v1/reports'))
    audit(parser.parse_args().reports)
