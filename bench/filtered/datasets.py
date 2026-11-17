"""Readers for the original ArXiv and Big ANN YFCC-10M filtered workloads."""
import json
from pathlib import Path
import struct

import numpy as np


def bin_matrix(path, dtype):
    with Path(path).open('rb') as f:
        n, d = struct.unpack('<II', f.read(8))
    return np.memmap(path, mode='r', dtype=dtype, offset=8, shape=(n, d))


def sparse(path):
    with Path(path).open('rb') as f:
        n, d, nnz = np.fromfile(f, dtype='<i8', count=3)
    ptr = np.memmap(path, mode='r', dtype='<i8', offset=24, shape=int(n)+1)
    ids = np.memmap(path, mode='r', dtype='<i4', offset=24+ptr.nbytes, shape=int(nnz))
    assert ptr[-1] == nnz
    return ptr, ids


def vectors(root, dataset):
    if dataset == 'arxiv':
        return np.load(root/'arxiv/vectors.npy', mmap_mode='r')
    return bin_matrix(root/'yfcc10m/base.10M.u8bin', 'uint8')


def rows(root, dataset):
    v = vectors(root, dataset)
    if dataset == 'arxiv':
        with (root/'arxiv/payloads.jsonl').open() as f:
            for i, line in enumerate(f):
                p = json.loads(line)
                yield (i, np.asarray(v[i], dtype=np.float32), p['labels'], int(p['update_date_ts']), p.get('submitter'))
        assert i+1 == len(v)
    else:
        ptr, tags = sparse(root/'yfcc10m/base.metadata.10M.spmat')
        assert len(ptr) == len(v)+1
        for i in range(len(v)):
            yield (i, np.asarray(v[i], dtype=np.float32), tags[ptr[i]:ptr[i+1]].tolist())


def queries(root, dataset):
    if dataset == 'arxiv':
        with (root/'arxiv/tests.jsonl').open() as f:
            result = [json.loads(line) for line in f]
        assert len(result) == 10000
        return result
    v = bin_matrix(root/'yfcc10m/query.public.100K.u8bin', 'uint8')
    ptr, tags = sparse(root/'yfcc10m/query.metadata.public.100K.spmat')
    # Big ANN knn results: header, all int32 IDs, then all float32 distances.
    gt = bin_matrix(root/'yfcc10m/GT.public.ibin', '<i4')
    assert v.shape == (100000, 192) and gt.shape == (100000, 10)
    assert len(ptr) == len(v)+1
    return [{'query': np.asarray(v[i], dtype=np.float32),
             'tags': tags[ptr[i]:ptr[i+1]].tolist(),
             'closest_ids': gt[i].tolist()} for i in range(len(v))]
