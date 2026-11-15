\echo Use "CREATE EXTENSION calico" to load this file. \quit

CREATE FUNCTION calico.status() RETURNS text
AS 'MODULE_PATHNAME', 'calico_status' LANGUAGE C VOLATILE PARALLEL RESTRICTED;

CREATE PROCEDURE calico.activate(timeout_ms integer DEFAULT 30000)
AS 'MODULE_PATHNAME', 'calico_activate' LANGUAGE C;

REVOKE ALL ON PROCEDURE calico.activate(integer) FROM PUBLIC;

CREATE FUNCTION calico.pool_statistics() RETURNS jsonb
AS 'MODULE_PATHNAME', 'calico_pool_statistics' LANGUAGE C VOLATILE PARALLEL RESTRICTED;

CREATE VIEW calico.pool AS SELECT calico.pool_statistics() AS statistics;
