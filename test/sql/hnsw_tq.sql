SET enable_seqscan = off;

-- L2

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq4');

SELECT * FROM t ORDER BY val <-> '[3,3,3]';
SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> (SELECT NULL::vector)) t2;

DROP TABLE t;

-- inner product

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_ip_ops) WITH (quantizer = 'tq4');

SELECT * FROM t ORDER BY val <#> '[3,3,3]';

DROP TABLE t;

-- cosine

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX ON t USING hnsw (val vector_cosine_ops) WITH (quantizer = 'tq4');

SELECT * FROM t ORDER BY val <=> '[3,3,3]';

DROP TABLE t;

-- re-rank off, and an empty index

CREATE TABLE t (val vector(3));
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq4', rerank = 'none');
SELECT * FROM t ORDER BY val <-> '[3,3,3]';
DROP TABLE t;

-- options

CREATE TABLE t (val vector(3));
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq8');
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (rerank = 'ntq16');
CREATE INDEX ON t USING hnsw (val vector_l1_ops) WITH (quantizer = 'tq4');
DROP TABLE t;

CREATE TABLE t (val vector(25));
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq4');
DROP TABLE t;

CREATE TABLE t (val vector(33));
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq4');
DROP TABLE t;

-- not supported yet

CREATE TABLE t (val vector(3));
INSERT INTO t (val) VALUES ('[1,2,3]');
CREATE INDEX ON t USING hnsw (val vector_l2_ops) WITH (quantizer = 'tq4');
INSERT INTO t (val) VALUES ('[1,1,1]');
SET hnsw.iterative_scan = relaxed_order;
SELECT * FROM t ORDER BY val <-> '[3,3,3]';
RESET hnsw.iterative_scan;
DROP TABLE t;
