\set ON_ERROR_STOP on
-- Run as superuser in a scratch database after preloading vector.
CREATE EXTENSION vector;
CREATE EXTENSION calico;

SELECT calico.status();
CREATE TABLE calico_example(id bigint PRIMARY KEY, embedding vector(3));
INSERT INTO calico_example
SELECT g, ARRAY[g::real/100, (g%17)::real, (g%7)::real]::vector
FROM generate_series(1, 5000) g;
CREATE INDEX ON calico_example USING hnsw(embedding vector_l2_ops);

CALL calico.activate(timeout_ms => 30000);
SET hnsw.enable_prefetch = on;
SELECT id FROM calico_example ORDER BY embedding <-> '[1,2,3]' LIMIT 5;
SELECT calico.pool_statistics()->>'mapping_mismatches' AS mapping_mismatches;

UPDATE calico_example SET embedding='[1,2,3]' WHERE id=1;
VACUUM (ANALYZE) calico_example;
CHECKPOINT;
SELECT id FROM calico_example ORDER BY embedding <-> '[1,2,3]' LIMIT 5;

DROP TABLE calico_example;
DROP EXTENSION vector;
DROP EXTENSION calico;
-- SQL object cleanup does not deactivate the shared buffer pool.
