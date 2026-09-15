#!/usr/bin/env python3
"""将 vLLM benchmark JSON 汇总为 CSV；不捏造 GPU 利用率或 bound 分类。"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ['phase','case','input_target','input_actual_min','input_actual_max',
          'output_target','concurrency','completed','failed','mean_ttft_ms','mean_tpot_ms',
          'mean_itl_ms','mean_e2el_ms','output_tokens_per_s','benchmark_duration_s','result']

def summarize(run: Path) -> list[dict]:
    rows = []
    for manifest in sorted(run.glob('manifest_*.csv')):
        with manifest.open() as f:
            for record in csv.DictReader(f):
                p = run / record['result']
                if record['status'] != 'ok' or not p.exists():
                    continue
                obj = json.loads(p.read_text())
                ilens = obj.get('input_lens', [])
                row = {k: record.get(k, '') for k in ('phase','case','input_target','output_target','concurrency','result')}
                row.update(input_actual_min=min(ilens) if ilens else '',
                           input_actual_max=max(ilens) if ilens else '',
                           completed=obj.get('completed',''), failed=obj.get('failed',0),
                           mean_ttft_ms=obj.get('mean_ttft_ms',''),
                           mean_tpot_ms=obj.get('mean_tpot_ms','') if int(record['output_target'])>1 else '',
                           mean_itl_ms=obj.get('mean_itl_ms','') if int(record['output_target'])>1 else '',
                           mean_e2el_ms=obj.get('mean_e2el_ms',''),
                           output_tokens_per_s=obj.get('output_throughput',''),
                           benchmark_duration_s=obj.get('duration',''))
                rows.append(row)
    with (run/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return rows

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    args=parser.parse_args()
    run=args.run_dir or Path((HERE/'active_run.txt').read_text().strip())
    rows=summarize(run)
    print(f'{run / "summary.csv"}\n成功汇总 {len(rows)} 组。')
    print('phase   case      TTFT(ms)   TPOT(ms)   E2E(ms)    output tok/s')
    for r in rows:
        def fmt(key):
            x=r[key]
            return f'{x:.3f}' if isinstance(x,(int,float)) else '-'
        print(f'{r["phase"]:<7} {r["case"]:<9} {fmt("mean_ttft_ms"):>9} '
              f'{fmt("mean_tpot_ms"):>10} {fmt("mean_e2el_ms"):>10} {fmt("output_tokens_per_s"):>12}')
    print('timing=采集关闭时的参考计时（仍可能有 nsys 注入影响）；trace=启用采集，仅用于归因。')
