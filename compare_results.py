#!/usr/bin/env python3
"""Compare multiple model summary.json files in one Markdown/CSV table."""
import argparse
import csv
import json
from pathlib import Path

COLUMNS=[('model','모델'),('status_accuracy','상태 일치율'),('level_accuracy','필수/우대 일치율'),
         ('evidence_valid_rate','근거 인용 유효율'),('json_success_rate','JSON 성공률'),
         ('false_have_count','허위 보유 판정 수'),('perf_median_ttft_s','TTFT 중위값(s)'),
         ('perf_median_decode_tps','출력 tok/s 중위값'),('quality_median_total_s','체크리스트 총시간 중위값(s)'),
         ('gpu_peak_gib','GPU 피크 사용(GiB)')]

def fmt(key, val):
    if val is None:return 'N/A'
    if key in ('status_accuracy','level_accuracy','evidence_valid_rate','json_success_rate'):
        return f'{float(val)*100:.1f}%'
    if isinstance(val,float):return f'{val:.2f}'
    return str(val)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('paths',nargs='+',type=Path,help='summary.json files, directories, or glob-matched results')
    p.add_argument('--out',type=Path,default=Path('comparison.md'))
    args=p.parse_args()
    rows=[]
    for path in args.paths:
        fp=path/'summary.json' if path.is_dir() else path
        if not fp.is_file():
            print(f'[SKIP] {fp} not found')
            continue
        record=json.loads(fp.read_text(encoding='utf-8'))
        hardware=record.get('hardware') or {}
        peak=hardware.get('peak_total_used_mib')
        row={k:record.get(k) for k,_ in COLUMNS}
        row['gpu_peak_gib']=round(peak/1024,2) if peak is not None else None
        rows.append(row)
    if not rows:raise SystemExit('No valid summaries')
    header='| '+' | '.join(label for _,label in COLUMNS)+' |'
    divider='|'+'|'.join(['---']*len(COLUMNS))+'|'
    lines=[header,divider]
    for r in rows:
        lines.append('| '+' | '.join(fmt(k,r.get(k)) for k,_ in COLUMNS)+' |')
    md='\n'.join(lines)+'\n'
    args.out.write_text(md,encoding='utf-8')
    with args.out.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k,_ in COLUMNS]);writer.writeheader();writer.writerows(rows)
    print(md)
    print(f'[SAVED] {args.out}, {args.out.with_suffix(".csv")}')

if __name__=='__main__':main()
