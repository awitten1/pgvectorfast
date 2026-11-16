\echo Use "CREATE EXTENSION vectorfast" to load this file. \quit

-- The preloaded utility hook installs our dependencies and activates Calico
-- after this script succeeds. Check the hook before accepting an installation;
-- without preloading, even an explicit CASCADE must fail and roll everything
-- back rather than silently leave the server using its stock buffer pool.
CREATE FUNCTION vectorfast_install_check() RETURNS void
AS 'MODULE_PATHNAME', 'calico_vectorfast_install_check' LANGUAGE C;
SELECT vectorfast_install_check();
DROP FUNCTION vectorfast_install_check();
