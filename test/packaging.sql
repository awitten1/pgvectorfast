\set ON_ERROR_STOP on
DO $$
BEGIN
    IF calico.status() <> 'calico' THEN
        RAISE EXCEPTION 'Calico is not active';
    END IF;
END $$;

CREATE TABLE packaged_items (id integer PRIMARY KEY, embedding vector(3));
INSERT INTO packaged_items VALUES (1, '[1,2,3]'), (2, '[4,5,6]'), (3, '[7,8,9]');
CREATE INDEX ON packaged_items USING hnsw (embedding vector_l2_ops);
SET enable_seqscan = off;
DO $$
BEGIN
    IF (SELECT id FROM packaged_items ORDER BY embedding <-> '[1,2,3]' LIMIT 1) <> 1 THEN
        RAISE EXCEPTION 'packaged HNSW search returned an incorrect neighbor';
    END IF;
END $$;
UPDATE packaged_items SET embedding = '[1,2,3]' WHERE id = 2;
DELETE FROM packaged_items WHERE id = 1;
VACUUM (ANALYZE) packaged_items;
CHECKPOINT;
DO $$
BEGIN
    IF (SELECT id FROM packaged_items ORDER BY embedding <-> '[1,2,3]' LIMIT 1) <> 2 THEN
        RAISE EXCEPTION 'packaged HNSW search failed after updates and vacuum';
    END IF;
    IF (SELECT (statistics->>'mapping_mismatches')::bigint FROM calico.pool) <> 0 THEN
        RAISE EXCEPTION 'Calico mapping mismatch';
    END IF;
END $$;
