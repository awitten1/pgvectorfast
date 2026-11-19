use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Exercise the actual utility command against a fresh server. In particular,
# failed/nested installations must not leave a globally activated buffer pool.
my $node = PostgreSQL::Test::Cluster->new('vectorfast');
$node->init;
$node->append_conf('postgresql.conf', "shared_preload_libraries = 'vectorfast'\nshared_buffers = '32MB'\n");
$node->start;

sub rejected
{
    my ($sql, $pattern, $label) = @_;
    my ($out, $err) = ('', '');
    # -c sends the entire string in one simple-query message, so a multi-command
    # implicit transaction is exercised rather than psql splitting statements.
    my $rc = $node->psql('postgres', '', stdout => \$out, stderr => \$err,
        extra_params => ['-c', $sql]);
    isnt($rc, 0, $label);
    like($err, $pattern, "$label: diagnostic");
}

sub uninstalled
{
    is($node->safe_psql('postgres', q{
        SELECT count(*) FROM pg_extension WHERE extname IN ('vector', 'calico', 'vectorfast');
    }), '0', 'failed installation leaves no extensions');
}

rejected('BEGIN; CREATE EXTENSION vectorfast; COMMIT;',
    qr/cannot run inside a transaction block/, 'explicit transaction rejected');
uninstalled();
rejected(q{DO $$ BEGIN EXECUTE 'CREATE EXTENSION vectorfast'; END $$;},
    qr/cannot be executed from a function/, 'nested command rejected');
uninstalled();
$node->safe_psql('postgres', 'CREATE ROLE vectorfast_user;');
rejected('SET ROLE vectorfast_user; CREATE EXTENSION vectorfast;',
    qr/requires superuser/, 'unprivileged activation rejected');
uninstalled();

# A script error occurs after dependency installation, but before activation.
$node->safe_psql('postgres', q{
    CREATE FUNCTION vectorfast_install_check() RETURNS void LANGUAGE sql AS 'SELECT';
});
rejected('CREATE EXTENSION vectorfast;', qr/already exists/,
    'failed installation rolls back dependencies');
uninstalled();
$node->safe_psql('postgres', 'DROP FUNCTION vectorfast_install_check(); CREATE EXTENSION calico;');
is($node->safe_psql('postgres', 'SELECT calico.status();'), 'stock',
    'failed commands never activate Calico');
$node->safe_psql('postgres', 'DROP EXTENSION calico;');

$node->safe_psql('postgres', 'CREATE EXTENSION vectorfast;');
is($node->safe_psql('postgres', 'SELECT calico.status();'), 'calico',
    'one command activates before returning');
is($node->safe_psql('postgres', q{SELECT extversion FROM pg_extension WHERE extname = 'vectorfast'}),
    '0.2.2', 'fresh installation uses the new version');
is($node->safe_psql('postgres', q{
    SELECT string_agg(extname, ',' ORDER BY extname) FROM pg_extension
    WHERE extname IN ('vector', 'calico', 'vectorfast');
}), 'calico,vector,vectorfast', 'standard dependencies installed automatically');
is($node->safe_psql('postgres', q{
    SELECT count(*) FROM pg_proc WHERE proname = 'vectorfast_install_check';
}), '0', 'temporary installation function removed');
$node->safe_psql('postgres', 'CREATE EXTENSION IF NOT EXISTS vectorfast;');
rejected('CREATE EXTENSION vectorfast;', qr/already exists/, 'duplicate command rejected');

# vector still owns its SQL objects. The wrapper works with an existing
# installation, preserving type and table OIDs as well as stored data.
$node->safe_psql('postgres', q{
    DROP EXTENSION vectorfast;
    CREATE TABLE items (id int PRIMARY KEY, embedding vector(3));
    INSERT INTO items VALUES (1, '[1,2,3]'), (2, '[4,5,6]');
    CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
    GRANT SELECT, INSERT ON items TO vectorfast_user;
});
my $oid = $node->safe_psql('postgres', q{SELECT 'vector'::regtype::oid;});
my $index_oid = $node->safe_psql('postgres', q{SELECT 'items_embedding_idx'::regclass::oid;});
$node->safe_psql('postgres', q{CREATE EXTENSION vectorfast VERSION '0.2.1';});
$node->safe_psql('postgres', 'ALTER EXTENSION vectorfast UPDATE;');
is($node->safe_psql('postgres', q{SELECT extversion FROM pg_extension WHERE extname = 'vectorfast'}),
    '0.2.2', 'version 0.2.1 upgrades to 0.2.2');
is($node->safe_psql('postgres', q{SELECT 'vector'::regtype::oid;}), $oid,
    'existing vector type retained');
is($node->safe_psql('postgres', q{SELECT 'items_embedding_idx'::regclass::oid;}), $index_oid,
    'existing HNSW index retained');
is($node->safe_psql('postgres', q{
    SET ROLE vectorfast_user;
    SELECT id FROM items ORDER BY embedding <-> '[1,2,3]' LIMIT 1;
}), '1', 'ordinary user can query after superuser setup');

# VACUUM's read stream can retain more pins than the private refcount array
# holds. A BM_VALID miss after moving an overflow entry must replenish its
# reservation before the slow-path retry (asserted by PostgreSQL 18).
$node->safe_psql('postgres', q{
    CREATE TABLE pin_retry (i int PRIMARY KEY, t text);
    CREATE INDEX ON pin_retry(t);
    ALTER TABLE pin_retry ALTER COLUMN t SET STORAGE EXTERNAL;
    INSERT INTO pin_retry SELECT g, repeat('1234567890', 269)
        FROM generate_series(1, 30) g;
    VACUUM (INDEX_CLEANUP TRUE, FULL TRUE) pin_retry;
    VACUUM (FULL TRUE) pin_retry;
    ALTER TABLE pin_retry SET (vacuum_index_cleanup = false);
    DELETE FROM pin_retry WHERE i < 15;
    VACUUM pin_retry;
    ALTER TABLE pin_retry SET (vacuum_index_cleanup = true);
    VACUUM pin_retry;
});
is($node->safe_psql('postgres', 'SELECT count(*) FROM pin_retry;'), '16',
    'VACUUM pin-overflow retry retains data and keeps server alive');

$node->restart;
is($node->safe_psql('postgres', 'SELECT calico.status();'), 'stock',
    'restart preserves existing manual activation semantics');
$node->safe_psql('postgres', 'CREATE EXTENSION IF NOT EXISTS vectorfast;');
is($node->safe_psql('postgres', 'SELECT calico.status();'), 'calico',
    'idempotent command reactivates after restart');

# Several commands in one simple-query message form an implicit transaction.
# Reject that context before installing anything or activating the pool.
$node->safe_psql('postgres', 'DROP EXTENSION vectorfast;');
rejected('CREATE EXTENSION vectorfast; SELECT 1/0;', qr/cannot run inside a transaction block/,
    'implicit transaction rejected');
is($node->safe_psql('postgres', q{
    SELECT count(*) FROM pg_extension WHERE extname = 'vectorfast';
}), '0', 'implicit transaction leaves wrapper uninstalled');
is($node->safe_psql('postgres', q{
    SELECT statistics->>'mapping_mismatches' FROM calico.pool;
}), '0', 'no mapping mismatches');

$node->safe_psql('postgres', 'CREATE DATABASE custom_schema;');
$node->safe_psql('custom_schema', 'CREATE SCHEMA ann;');
$node->safe_psql('custom_schema', 'CREATE EXTENSION vectorfast SCHEMA ann;');
is($node->safe_psql('custom_schema', q{
    SELECT '[1,2,3]'::ann.vector OPERATOR(ann.<->) '[1,2,4]'::ann.vector;
}), '1', 'custom schema propagated to vector dependency');
$node->stop;

my $plain = PostgreSQL::Test::Cluster->new('no_preload');
$plain->init;
# Docker images preload vectorfast in postgresql.conf.sample.
$plain->append_conf('postgresql.conf', "shared_preload_libraries = ''\n");
$plain->start;
my ($out, $err) = ('', '');
isnt($plain->psql('postgres', 'CREATE EXTENSION vectorfast CASCADE;',
    stdout => \$out, stderr => \$err), 0, 'installation requires preload');
like($err, qr/requires shared_preload_libraries/, 'missing preload diagnostic');
is($plain->safe_psql('postgres', q{
    SELECT count(*) FROM pg_extension WHERE extname IN ('vector', 'calico', 'vectorfast');
}), '0', 'missing preload rolls back all extensions');
$plain->safe_psql('postgres', 'CREATE EXTENSION vector;');
is($plain->safe_psql('postgres', q{SELECT '[1,2,3]'::vector;}), '[1,2,3]',
    'legacy vector installation still works without preload');
$plain->stop;

done_testing();
