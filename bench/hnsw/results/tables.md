## inmem

| Dataset | System | ef_search | Recall@10 | QPS | Mean (ms) | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT10M | pgvector | 40 | 0.9334 | 927.0 | 1.079 | 1.460 |
| SIFT10M | pgvectorfast | 40 | 0.9334 | 4,933.1 | 0.203 | 0.250 |
| SIFT10M | Chroma | 61 | 0.9335 | 847.4 | 1.180 | 1.296 |
| SIFT10M | Qdrant | 35 | 0.9335 | 1,668.5 | 0.599 | 0.675 |
| DEEP10M | pgvector | 39 | 0.8655 | 836.1 | 1.196 | 1.550 |
| DEEP10M | pgvectorfast | 39 | 0.8656 | 4,913.3 | 0.204 | 0.245 |
| DEEP10M | Chroma | 59 | 0.8664 | 862.9 | 1.159 | 1.282 |
| DEEP10M | Qdrant | 34 | 0.8673 | 1,659.1 | 0.603 | 0.668 |

## oom

| Dataset | System | ef_search | Recall@10 | QPS | Mean (ms) | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT10M | pgvector | 40 | 0.9356 | 11.4 | 87.668 | 129.250 |
| SIFT10M | pgvectorfast | 40 | 0.9355 | 65.0 | 15.393 | 19.753 |
| SIFT10M | Chroma | 61 | — | OOM at 2 GiB | — | — |
| SIFT10M | Qdrant | 35 | 0.9372 | 21.2 | 47.240 | 74.524 |
| DEEP10M | pgvector | 39 | 0.8753 | 10.4 | 96.219 | 127.511 |
| DEEP10M | pgvectorfast | 39 | 0.8753 | 61.4 | 16.294 | 19.903 |
| DEEP10M | Chroma | 59 | — | OOM at 2 GiB | — | — |
| DEEP10M | Qdrant | 34 | 0.8779 | 17.9 | 55.917 | 74.860 |
