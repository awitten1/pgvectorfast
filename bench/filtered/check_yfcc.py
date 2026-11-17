#!/usr/bin/env python3
"""Check selected YFCC reference neighbors against their tags and L2 distances.

This validates the supplied answers' predicates and distances; exact SQL checks
in run.py separately verify nearest-neighbor ranking on sample queries.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import sparse, bin_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=a.data_root/'yfcc10m'
    qp,qt=sparse(root/'query.metadata.public.100K.spmat')
    bp,bt=sparse(root/'base.metadata.10M.spmat')
    q=bin_matrix(root/'query.public.100K.u8bin','uint8')
    b=bin_matrix(root/'base.10M.u8bin','uint8')
    gt=bin_matrix(root/'GT.public.ibin','<i4')
    gd=np.memmap(root/'GT.public.ibin',mode='r',dtype='<f4',
                 offset=8+gt.nbytes,shape=gt.shape)
    rng=np.random.default_rng(20261117)
    indices=[]
    for length in [1,2]:
        order=rng.permutation(np.flatnonzero(np.diff(qp)==length)).tolist()
        indices.extend(order[:400])
    max_error=0
    for i in indices:
        required=set(qt[qp[i]:qp[i+1]].tolist())
        ids=gt[i]
        assert len(set(ids.tolist()))==10
        for row in ids:
            assert required.issubset(set(bt[bp[row]:bp[row+1]].tolist())),(i,int(row))
        delta=np.asarray(b[ids],dtype=np.float32)-np.asarray(q[i],dtype=np.float32)
        distances=np.sum(delta*delta,axis=1)
        max_error=max(max_error,float(np.max(np.abs(distances-gd[i]))))
        assert np.allclose(distances,gd[i],rtol=1e-5,atol=1e-3),(i,distances,gd[i])
    result=dict(queries_checked=len(indices),neighbors_checked=len(indices)*10,
                max_squared_l2_error=max_error,
                all_reference_neighbors_satisfy_predicates=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
