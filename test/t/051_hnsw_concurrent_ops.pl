use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Verify correctness under concurrent writes, deletes, VACUUM, savepoint aborts,
# multiple open cursors, pending radix-sorted drains, duplicate-vector TID
# chains, and heavy Calico buffer eviction with shared_buffers pressure.

sub make_unit_vec
{
	my ($dim, $seed) = @_;
	my @raw = map { sin($_ * 0.43 + $seed) } (1 .. $dim);
	my $norm = 0.0;
	$norm += $_ * $_ for @raw;
	$norm = sqrt($norm);
	return '[' . join(',', map { sprintf('%.6f', $_ / $norm) } @raw) . ']';
}

my $node = PostgreSQL::Test::Cluster->new('concurrent_res');
$node->init;
$node->append_conf('postgresql.conf',
	"shared_preload_libraries = 'vectorfast'\nshared_buffers = '32MB'\nautovacuum = off\njit = off\n");
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vectorfast;');
is($node->safe_psql('postgres', 'SELECT calico.status();'), 'calico',
	'resident node activates Calico');

# ---------------------------------------------------------------------------
# Subtest 1: Multi-cursor isolation, subtransaction savepoint cleanup,
# >16K visited table promotion, pending >4096 radix-sort drain, and mid-cursor
# re-expansion across L2 (<->) and negative/mixed IP (<#>).
# ---------------------------------------------------------------------------
$node->safe_psql('postgres', q{
	SELECT setseed(0.314159);
	CREATE TABLE items_big (
		id integer PRIMARY KEY,
		cat integer NOT NULL,
		v vector(32) NOT NULL
	) WITH (autovacuum_enabled = false);
	WITH raw AS (
		SELECT i,
			   i % 211 AS cat,
			   ARRAY(
				   SELECT ((sin(i * 0.17 + j * 0.31) + (random() - 0.5) * 0.4))::real
				   FROM generate_series(1, 32) j
				   WHERE i > 0
			   ) AS a
		FROM generate_series(1, 22000) i
	)
	INSERT INTO items_big
	SELECT i,
		   cat,
		   ARRAY(
			   SELECT (val / sqrt(norm_sq))::real
			   FROM unnest(a) AS val,
					LATERAL (SELECT sum(x * x) AS norm_sq FROM unnest(a) AS x) n
		   )::vector(32)
	FROM raw;
	CREATE INDEX items_big_l2_idx ON items_big USING hnsw (v vector_l2_ops)
		WITH (m = 16, ef_construction = 48);
	CREATE INDEX items_big_ip_idx ON items_big USING hnsw (v vector_ip_ops)
		WITH (m = 16, ef_construction = 48);
});

my $q32_a = make_unit_vec(32, 1.1);
my $q32_b = make_unit_vec(32, 2.7);

my $multi_res = $node->safe_psql('postgres', qq{
	BEGIN;
	SET LOCAL enable_seqscan = off;
	SET LOCAL hnsw.ef_search = 16;
	SET LOCAL hnsw.max_scan_tuples = 25000;

	-- Mid-cursor SAVEPOINT rollback must cleanly release backend visited cache.
	SAVEPOINT sp_cur;
	SET LOCAL hnsw.iterative_scan = relaxed_order;
	DECLARE c_abort CURSOR FOR
		SELECT id FROM items_big WHERE cat = 0 ORDER BY v <-> '$q32_a';
	FETCH 5 FROM c_abort;
	ROLLBACK TO SAVEPOINT sp_cur;

	-- Interleave two live cursors in one backend (strict + relaxed order).
	SET LOCAL hnsw.iterative_scan = relaxed_order;
	DECLARE c_rel CURSOR FOR
		SELECT id, cat, (v <-> '$q32_a') AS d FROM items_big WHERE cat = 0 ORDER BY d;
	SET LOCAL hnsw.iterative_scan = strict_order;
	DECLARE c_str CURSOR FOR
		SELECT id, cat, (v <-> '$q32_b') AS d FROM items_big WHERE cat = 1 ORDER BY d;

	FETCH 12 FROM c_rel;
	FETCH 12 FROM c_str;
	FETCH 20 FROM c_rel;
	FETCH 20 FROM c_str;
	FETCH ALL FROM c_rel;
	FETCH ALL FROM c_str;
	CLOSE c_rel;
	CLOSE c_str;
	COMMIT;

	SELECT count(*) FROM items_big WHERE cat = 5;
});
like($multi_res, qr/(?:^|\n)105\z/, 'savepoint rollback and interleaved multi-cursor scans succeed');

for my $spec (['l2', '<->', $q32_a], ['ip', '<#>', $q32_b])
{
	my ($label, $op, $qvec) = @$spec;
	my $cur = $node->background_psql('postgres');

	# cat < 6 matches 626 rows (~1 in 35). Stopping expansion around 8000
	# tuples puts >4096 keys in pending; fetching 100 then 80 drains >40
	# matches across !expand batches (triggering HnswSortPendingDesc) while
	# leaving sorted pending keys before re-enabling expansion with 25000.
	$cur->query_safe(qq{
		BEGIN;
		SET LOCAL enable_seqscan = off;
		SET LOCAL hnsw.ef_search = 16;
		SET LOCAL hnsw.max_scan_tuples = 8000;
		SET LOCAL hnsw.iterative_scan = relaxed_order;
		DECLARE c_drain CURSOR FOR
			SELECT id, cat, (v $op '$qvec') AS d FROM items_big
			WHERE cat < 6 ORDER BY d;
	});
	my $b1 = $cur->query_safe('FETCH 100 FROM c_drain;');
	my $b2 = $cur->query_safe('FETCH 80 FROM c_drain;');
	$cur->query_safe('SET LOCAL hnsw.max_scan_tuples = 25000;');
	my $b3 = $cur->query_safe('FETCH ALL FROM c_drain;');
	$cur->query_safe('COMMIT;');
	$cur->quit;

	my @rows = grep { $_ ne '' } split("\n", "$b1\n$b2\n$b3");
	my %seen;
	my $bad_cat = 0;
	for my $line (@rows)
	{
		my ($id, $cat, $dist) = split(/\|/, $line);
		$seen{$id}++;
		$bad_cat++ if !defined($cat) || $cat >= 6;
	}
	is($bad_cat, 0, "$label radix drain + mid-cursor resume matches predicate");
	is(scalar(keys %seen), scalar(@rows), "$label radix drain + mid-cursor resume produces unique IDs");
	cmp_ok(scalar(@rows), '>=', 580, "$label radix drain + mid-cursor resume retrieves >= 580 matching rows");

	# Strict order monotonic check under tight max_scan_tuples + work_mem='64kB'
	my $strict_out = $node->safe_psql('postgres', qq{
		SET enable_seqscan = off;
		SET hnsw.ef_search = 8;
		SET hnsw.max_scan_tuples = 6000;
		SET work_mem = '64kB';
		SET hnsw.iterative_scan = strict_order;
		WITH res AS (
			SELECT id, cat, (v $op '$qvec') AS d
			FROM items_big
			WHERE cat < 12
			ORDER BY d
			LIMIT 80
		),
		chk AS (
			SELECT id, cat, d, lag(d) OVER () AS prev_d FROM res
		)
		SELECT count(*),
			   count(DISTINCT id),
			   coalesce(bool_and(cat < 12), false),
			   coalesce(bool_and(prev_d IS NULL OR d >= prev_d - 1e-6), false)
		FROM chk;
	});
	my ($cnt, $uniq, $cat_ok, $mono_ok) = split(/\|/, $strict_out);
	cmp_ok($cnt, '>=', 30, "$label low-work_mem strict_order returned >= 30 rows");
	is($uniq, $cnt, "$label low-work_mem strict_order unique IDs");
	is($cat_ok, 't', "$label low-work_mem strict_order predicate satisfied");
	is($mono_ok, 't', "$label low-work_mem strict_order monotonic distances");

	my $full_visit = $node->safe_psql('postgres', qq{
		SET enable_seqscan = off;
		SET hnsw.ef_search = 32;
		SET hnsw.max_scan_tuples = 22000;
		SET hnsw.iterative_scan = relaxed_order;
		SELECT count(*) = count(DISTINCT id) AND count(*) >= 100 AND coalesce(bool_and(cat = 13), true)
		FROM (
			SELECT id, cat FROM items_big WHERE cat = 13 ORDER BY v $op '$qvec' LIMIT 105
		) s;
	});
	is($full_visit, 't', "$label >16K visited promotion stays exact and deduplicated");
}

# ---------------------------------------------------------------------------
# Subtest 2: Duplicate-vector multi-TID chains (2..12 TIDs, exceeding
# HNSW_HEAPTIDS=10) under concurrent-style DELETE/UPDATE/VACUUM/INSERT across
# <->, <#>, and <=>.
# ---------------------------------------------------------------------------
my $q8_dups = make_unit_vec(8, 0.9);
for my $metric (['l2', '<->', 'vector_l2_ops'],
				['ip', '<#>', 'vector_ip_ops'],
				['cos', '<=>', 'vector_cosine_ops'])
{
	my ($name, $op, $opclass) = @$metric;
	$node->safe_psql('postgres', qq{
		DROP TABLE IF EXISTS dups_test;
		CREATE TABLE dups_test (
			id integer PRIMARY KEY,
			tag integer NOT NULL,
			v vector(8) NOT NULL
		) WITH (autovacuum_enabled = false);
		WITH raw AS (
			SELECT i, i % 5 AS tag,
				   ARRAY(SELECT sin(((i - 1) / 12) * 0.61 + j * 0.19)::real FROM generate_series(1, 8) j) AS a
			FROM generate_series(1, 600) i
			UNION ALL
			SELECT 1000 + i, i % 5 AS tag,
				   ARRAY(SELECT cos(((i - 1) / 5 + 100) * 0.47 + j * 0.23)::real FROM generate_series(1, 8) j) AS a
			FROM generate_series(1, 250) i
			UNION ALL
			SELECT 2000 + i, i % 5 AS tag,
				   ARRAY(SELECT sin(((i - 1) / 2 + 200) * 0.37 + j * 0.29)::real FROM generate_series(1, 8) j) AS a
			FROM generate_series(1, 100) i
		)
		INSERT INTO dups_test
		SELECT i, tag,
			   ARRAY(
				   SELECT (val / sqrt(norm_sq))::real
				   FROM unnest(a) AS val,
						LATERAL (SELECT sum(x * x) AS norm_sq FROM unnest(a) AS x) n
			   )::vector(8)
		FROM raw;
		CREATE INDEX dups_idx ON dups_test USING hnsw (v $opclass);
	});

	# Mutate chains: delete half of 12-TID groups, update tags & vectors, VACUUM, re-insert
	$node->safe_psql('postgres', q{
		DELETE FROM dups_test WHERE id <= 600 AND id % 2 = 0;
		UPDATE dups_test SET tag = 4 WHERE id > 1000 AND id <= 1250 AND id % 3 = 0;
		UPDATE dups_test SET v = v WHERE id <= 600 AND id % 7 = 0;
		VACUUM dups_test;
		WITH raw AS (
			SELECT 3000 + g AS id, 0 AS tag,
				   ARRAY(SELECT sin((g / 4) * 0.61 + j * 0.19)::real FROM generate_series(1, 8) j) AS a
			FROM generate_series(1, 60) g
		)
		INSERT INTO dups_test
		SELECT id, tag,
			   ARRAY(
				   SELECT (val / sqrt(norm_sq))::real
				   FROM unnest(a) AS val,
						LATERAL (SELECT sum(x * x) AS norm_sq FROM unnest(a) AS x) n
			   )::vector(8)
		FROM raw;
	});

	my $exact_ids = $node->safe_psql('postgres', q{
		SELECT string_agg(id::text, ',' ORDER BY id) FROM dups_test WHERE tag = 0;
	});

	for my $mode ('relaxed_order', 'strict_order')
	{
		my $scan_ids = $node->safe_psql('postgres', qq{
			SET enable_seqscan = off;
			SET hnsw.ef_search = 16;
			SET hnsw.max_scan_tuples = 5000;
			SET hnsw.iterative_scan = $mode;
			SELECT count(*) = count(DISTINCT id),
				   string_agg(id::text, ',' ORDER BY id)
			FROM (
				SELECT id FROM dups_test WHERE tag = 0
				ORDER BY v $op '$q8_dups' LIMIT 500
			) s;
		});
		my ($uniq_flag, $got_ids) = split(/\|/, $scan_ids, 2);
		is($uniq_flag, 't', "dups $name $mode has no duplicate TIDs after DML/VACUUM");
		is($got_ids, $exact_ids, "dups $name $mode returns exact live TIDs across overflow chains");
	}
}
$node->safe_psql('postgres', 'DROP TABLE IF EXISTS dups_test;');

# ---------------------------------------------------------------------------
# Subtest 3: Resident 8-client pgbench stress mixing concurrent INSERT, DELETE,
# UPDATE, VACUUM, and filtered scans across <->, <#>, <=> with SQL 1/0 checks.
# ---------------------------------------------------------------------------
$node->safe_psql('postgres', q{
	SELECT setseed(0.718281);
	CREATE SEQUENCE conc_res_seq START 100000;
	CREATE TABLE conc_res (
		id integer PRIMARY KEY,
		cat integer NOT NULL,
		v vector(32) NOT NULL
	) WITH (autovacuum_enabled = false);
	WITH raw AS (
		SELECT i, i % 19 AS cat,
			   ARRAY(
				   SELECT ((cos(i * 0.23 + j * 0.41) + (random() - 0.5) * 0.35))::real
				   FROM generate_series(1, 32) j
				   WHERE i > 0
			   ) AS a
		FROM generate_series(1, 6000) i
	)
	INSERT INTO conc_res
	SELECT i, cat,
		   ARRAY(
			   SELECT (val / sqrt(norm_sq))::real
			   FROM unnest(a) AS val,
					LATERAL (SELECT sum(x * x) AS norm_sq FROM unnest(a) AS x) n
		   )::vector(32)
	FROM raw;
	CREATE INDEX conc_res_l2_idx ON conc_res USING hnsw (v vector_l2_ops)
		WITH (m = 16, ef_construction = 48);
	CREATE INDEX conc_res_ip_idx ON conc_res USING hnsw (v vector_ip_ops)
		WITH (m = 16, ef_construction = 48);
	CREATE INDEX conc_res_cos_idx ON conc_res USING hnsw (v vector_cosine_ops)
		WITH (m = 16, ef_construction = 48);
});

my $rand32_norm_sql = "(SELECT ARRAY(SELECT (val / sqrt(norm_sq))::real FROM unnest(a) AS val, LATERAL (SELECT sum(x * x) AS norm_sq FROM unnest(a) AS x) n)::vector(32) FROM (SELECT ARRAY[" . join(',', map { '(random() * 2 - 1)::real' } (1 .. 32)) . "] AS a) s)";
my $q32_c = make_unit_vec(32, 3.4);

$node->pgbench(
	'--no-vacuum --client=8 --jobs=4 --time=6',
	0,
	[qr{actually processed}],
	[qr{^$}],
	'8-client resident concurrent writes, VACUUM, and filtered HNSW scans',
	{
		'051_res_insert@120' => "INSERT INTO conc_res (id, cat, v) VALUES (nextval('conc_res_seq'), floor(random() * 19)::int, $rand32_norm_sql);",
		'051_res_delete@80'  => "DELETE FROM conc_res WHERE id = (SELECT id FROM conc_res WHERE id % 2 = 0 AND id > 1000 ORDER BY random() LIMIT 1);",
		'051_res_update@80'  => "UPDATE conc_res SET cat = (cat + 1) % 19, v = $rand32_norm_sql WHERE id = (SELECT id FROM conc_res WHERE id % 2 = 1 AND id > 1000 ORDER BY random() LIMIT 1);",
		'051_res_vacuum@4'   => "VACUUM conc_res;",
		'051_res_rel_l2@120' => "SET enable_seqscan = off; SET hnsw.iterative_scan = relaxed_order; SET hnsw.ef_search = 16; WITH r AS (SELECT id, cat FROM conc_res WHERE cat = 3 ORDER BY v <-> '$q32_c' LIMIT 25) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 3), true) THEN 1 ELSE 0 END FROM r;",
		'051_res_str_ip@100' => "SET enable_seqscan = off; SET hnsw.iterative_scan = strict_order; SET hnsw.ef_search = 12; WITH r AS (SELECT id, cat, (v <#> '$q32_c') AS d FROM conc_res WHERE cat = 5 ORDER BY d LIMIT 25), chk AS (SELECT id, cat, d, lag(d) OVER () AS prev_d FROM r) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 5), true) AND coalesce(bool_and(prev_d IS NULL OR d >= prev_d - 1e-6), true) THEN 1 ELSE 0 END FROM chk;",
		'051_res_str_cos@100'=> "SET enable_seqscan = off; SET hnsw.iterative_scan = strict_order; SET hnsw.ef_search = 12; WITH r AS (SELECT id, cat, (v <=> '$q32_c') AS d FROM conc_res WHERE cat = 7 ORDER BY d LIMIT 25), chk AS (SELECT id, cat, d, lag(d) OVER () AS prev_d FROM r) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 7), true) AND coalesce(bool_and(prev_d IS NULL OR d >= prev_d - 1e-6), true) THEN 1 ELSE 0 END FROM chk;",
		'051_res_off_l2@60'  => "SET enable_seqscan = off; SET hnsw.iterative_scan = off; SET hnsw.ef_search = 24; WITH r AS (SELECT id, cat FROM conc_res WHERE cat = 9 ORDER BY v <-> '$q32_c' LIMIT 15) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 9), true) THEN 1 ELSE 0 END FROM r;"
	}
);

is($node->safe_psql('postgres',
	"SELECT (calico.pool_statistics()->>'mapping_mismatches')::bigint;"),
	'0', 'resident stress leaves 0 Calico mapping mismatches');

for my $op ('<->', '<#>', '<=>')
{
	my $hits = 0;
	for my $s (1 .. 4)
	{
		my $qv = make_unit_vec(32, 0.5 + $s * 0.7);
		my %exp = map { $_ => 1 } split("\n", $node->safe_psql('postgres',
			"SET enable_indexscan = off; SET enable_bitmapscan = off; " .
			"SELECT id FROM conc_res WHERE cat = 3 ORDER BY (v $op '$qv') + 0 LIMIT 10;"));
		my @got = split("\n", $node->safe_psql('postgres',
			"SET enable_seqscan = off; SET hnsw.ef_search = 64; SET hnsw.max_scan_tuples = 20000; " .
			"SET hnsw.iterative_scan = relaxed_order; " .
			"WITH c AS MATERIALIZED (SELECT id, (v $op '$qv') AS d FROM conc_res WHERE cat = 3 ORDER BY d LIMIT 30) " .
			"SELECT id FROM c ORDER BY d + 0 LIMIT 10;"));
		is(scalar(@got), 10, "post-stress recall $op seed $s returns 10 rows");
		$hits += scalar grep { exists $exp{$_} } @got;
	}
	cmp_ok($hits / 40, '>=', 0.90, "post-stress recall $op >= 0.90");
}
$node->stop;

# ---------------------------------------------------------------------------
# Subtest 4: Low-memory (shared_buffers = '4MB') buffer-eviction & async-wave
# stress + paused cursor across DELETE, VACUUM, and slot-reusing INSERTs.
# ---------------------------------------------------------------------------
my $low = PostgreSQL::Test::Cluster->new('concurrent_oom');
$low->init;
$low->append_conf('postgresql.conf',
	"shared_preload_libraries = 'vectorfast'\nshared_buffers = '4MB'\nautovacuum = off\njit = off\n");
$low->start;

$low->safe_psql('postgres', 'CREATE EXTENSION vectorfast;');
is($low->safe_psql('postgres', 'SELECT calico.status();'), 'calico',
	'low-memory node activates Calico');

$low->safe_psql('postgres', q{
	SELECT setseed(0.141421);
	CREATE SEQUENCE oom_seq START 100000;
	CREATE TABLE items_oom (
		id integer PRIMARY KEY,
		cat integer NOT NULL,
		v vector(64) NOT NULL
	) WITH (autovacuum_enabled = false);
	INSERT INTO items_oom
	SELECT i,
		   i % 37,
		   ARRAY(
			   SELECT ((sin(i * 0.29 + j * 0.17) + (random() - 0.5) * 0.5))::real
			   FROM generate_series(1, 64) j
			   WHERE i > 0
		   )::vector(64)
	FROM generate_series(1, 16000) i;
	CREATE INDEX items_oom_l2_idx ON items_oom USING hnsw (v vector_l2_ops)
		WITH (m = 16, ef_construction = 40);
});

my $q64 = make_unit_vec(64, 1.9);

my $oom_cur = $low->background_psql('postgres');
$oom_cur->query_safe(qq{
	BEGIN;
	SET LOCAL enable_seqscan = off;
	SET LOCAL hnsw.ef_search = 16;
	SET LOCAL hnsw.iterative_scan = relaxed_order;
	SET LOCAL hnsw.max_scan_tuples = 20000;
	DECLARE oom_scan_cur CURSOR FOR
		SELECT id FROM items_oom WHERE cat = 0 ORDER BY v <-> '$q64' LIMIT 150;
});
my $oom_first = $oom_cur->query_safe('FETCH 5 FROM oom_scan_cur;');

$low->safe_psql('postgres', q{
	DELETE FROM items_oom WHERE id <= 16000 AND id % 3 = 0;
	VACUUM items_oom;
	INSERT INTO items_oom
	SELECT 50000 + g,
		   g % 37,
		   ARRAY(
			   SELECT (random() * 2 - 1)::real
			   FROM generate_series(1, 64) j
			   WHERE g > 0
		   )::vector(64)
	FROM generate_series(1, 5500) g;
});

my $oom_rest = $oom_cur->query_safe('FETCH ALL FROM oom_scan_cur;');
$oom_cur->query_safe('COMMIT;');
$oom_cur->quit;

my @oom_rows = grep { $_ ne '' } split("\n", "$oom_first\n$oom_rest");
my %oom_seen = map { $_ => 1 } @oom_rows;
cmp_ok(scalar(@oom_rows), '>=', 100, 'low-memory paused cursor retrieves >= 100 snapshot rows');
is(scalar(keys %oom_seen), scalar(@oom_rows), 'low-memory paused cursor has no duplicate IDs');
ok(!scalar(grep { $_ >= 50000 || ($_ % 37) != 0 } @oom_rows),
	'low-memory paused cursor sees only original snapshot matching tuples');

my $rand64_expr = "ARRAY[" . join(',', map { '(random() * 2 - 1)::real' } (1 .. 64)) . "]::vector(64)";
$low->pgbench(
	'--no-vacuum --client=6 --jobs=3 --time=5',
	0,
	[qr{actually processed}],
	[qr{^$}],
	'6-client low-memory (shared_buffers=4MB) eviction + DML + VACUUM + filtered scans',
	{
		'051_oom_insert@100' => "INSERT INTO items_oom (id, cat, v) VALUES (nextval('oom_seq'), floor(random() * 37)::int, $rand64_expr);",
		'051_oom_delete@80'  => "DELETE FROM items_oom WHERE id = (SELECT id FROM items_oom WHERE id > 1000 ORDER BY random() LIMIT 1);",
		'051_oom_vacuum@4'   => "VACUUM items_oom;",
		'051_oom_rel@120'    => "SET enable_seqscan = off; SET hnsw.iterative_scan = relaxed_order; SET hnsw.ef_search = 16; WITH r AS (SELECT id, cat FROM items_oom WHERE cat = 2 ORDER BY v <-> '$q64' LIMIT 25) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 2), true) THEN 1 ELSE 0 END FROM r;",
		'051_oom_str@80'     => "SET enable_seqscan = off; SET hnsw.iterative_scan = strict_order; SET hnsw.ef_search = 12; WITH r AS (SELECT id, cat, (v <-> '$q64') AS d FROM items_oom WHERE cat = 4 ORDER BY d LIMIT 25), chk AS (SELECT id, cat, d, lag(d) OVER () AS prev_d FROM r) SELECT 1 / CASE WHEN count(*) = count(DISTINCT id) AND coalesce(bool_and(cat = 4), true) AND coalesce(bool_and(prev_d IS NULL OR d >= prev_d - 1e-6), true) THEN 1 ELSE 0 END FROM chk;"
	}
);

is($low->safe_psql('postgres',
	"SELECT (calico.pool_statistics()->>'mapping_mismatches')::bigint;"),
	'0', 'low-memory stress leaves 0 Calico mapping mismatches');
$low->stop;

done_testing();
