#!/usr/bin/env python3
"""Validate and summarize paired filtered-search measurements."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=['arxiv','yfcc10m'],default=['arxiv'])
    a=p.parse_args()
    rows=[]
    comparisons=[]
    for dataset,large in [('arxiv',16),('yfcc10m',32)]:
        if dataset not in a.datasets:
            continue
        fixture=json.loads((a.results/f'{dataset}-fixture.json').read_text())
        assert fixture['count']==(2138591 if dataset=='arxiv' else 10000000)
        split=json.loads((a.results/f'{dataset}-split.json').read_text())
        assert set(split['calibration']).isdisjoint(split['timed'])
        for ram in [large,2]:
            paired={}
            for store in ['pgvector','pgvectorfast']:
                r=json.loads((a.results/f'{store}-{dataset}-auto-{ram}g.json').read_text())
                assert r['ram_gib']==ram and r['swap_gib']==16
                assert r['server_cpu']==2 and r['client_cpu']==0
                assert len(r['repeats'])==3
                for state in [r['memory_before'],r['memory_after']]:
                    assert int(state['memory.max'])==ram*1024**3
                    assert int(state['memory.swap.max'])==16*1024**3
                    events=dict(line.split() for line in state['memory.events'].splitlines())
                    assert int(events.get('oom_kill',0))==0
                    if store=='pgvectorfast':
                        assert state['calico_pool']['mode']=='calico'
                        assert state['calico_pool']['mapping_mismatches']==0
                if ram==2: assert r['timed_read_bytes']>=64*1024**2
                for rep in r['repeats']:
                    actual=[q['query_index'] for q in rep['queries']]
                    expected=r.get('timed_indices',split['timed'])
                    assert actual==expected
                    assert set(expected).issubset(split['timed'])
                    if ram!=2: assert expected==split['timed']
                groups=(['labels','update_date_ts','submitter','all'] if dataset=='arxiv'
                        else ['1-tag','2-tag','all'])
                assert set(groups)==set(r['repeats'][0]['metrics'])
                for group in groups:
                    ms=[rep['metrics'][group] for rep in r['repeats']]
                    if group!='all': assert {m['queries'] for m in ms}=={100 if ram==2 else 300}
                    assert max(m['recall_at_10'] for m in ms)-min(m['recall_at_10'] for m in ms)<=.001+1e-12
                    row=dict(dataset=dataset,ram_gib=ram,swap_gib=16,filter=group,store=store,
                             ef_search=r['configuration']['ef_search'],max_scan_tuples=r['configuration']['max_scan_tuples'])
                    for key in ['recall_at_10','complete_fraction','qps','mean_ms','p50_ms','p95_ms']:
                        row[key]=statistics.median(m[key] for m in ms)
                    rows.append(row)
                    paired[group,store]=row
            for group in groups:
                stock=paired[group,'pgvector']
                fast=paired[group,'pgvectorfast']
                gap=abs(stock['recall_at_10']-fast['recall_at_10'])
                complete=min(stock['complete_fraction'],fast['complete_fraction'])
                comparisons.append(dict(dataset=dataset,ram_gib=ram,filter=group,
                                        pgvector_qps=stock['qps'],pgvectorfast_qps=fast['qps'],
                                        speedup=fast['qps']/stock['qps'],recall_gap=gap,
                                        complete_fraction_min=complete,quality_matched=gap<=.01+1e-12 and complete>=.99))
    for name,data in [('summary.csv',rows),('comparison.csv',comparisons)]:
        with (a.results/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n')
            w.writeheader(); w.writerows(data)
    lines=['| Dataset | RAM | Filter | pgvector QPS | pgvectorfast QPS | Speedup | Recall gap (pp) | Quality matched |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in comparisons:
        lines.append(f"| {r['dataset']} | {r['ram_gib']} GiB | {r['filter']} | {r['pgvector_qps']:.2f} | {r['pgvectorfast_qps']:.2f} | {r['speedup']:.2f}x | {100*r['recall_gap']:.3f} | {r['quality_matched']} |")
    (a.results/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
