use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;
use File::Spec;

# HNSW with TurboQuant codes (quantizer = 'tq4'): CREATE INDEX and graph-file
# builds, search with and without Calico's optimistic reads, and what is not
# supported yet

my $node = PostgreSQL::Test::Cluster->new('hnsw_tq');
$node->init;
$node->append_conf('postgresql.conf', "shared_preload_libraries = 'vectorfast'\nshared_buffers = '64MB'\njit = off\n");
$node->start;
$node->safe_psql('postgres', 'CREATE EXTENSION vectorfast;');

# 96 dimensions = 3 blocks of 32 (r = 3); 64 = one block (r = 1)
for my $dim (96, 64)
{
	$node->safe_psql('postgres', qq{
		SELECT setseed(0.42);
		CREATE TABLE t$dim (id integer PRIMARY KEY, v vector($dim)) WITH (autovacuum_enabled = false);
		INSERT INTO t$dim SELECT i, ARRAY(SELECT (random() * 2 - 1)::real
			FROM generate_series(1, $dim) j WHERE i > 0) FROM generate_series(1, 2000) i;
	});
}

sub query_vector
{
	my ($dim, $q) = @_;
	return '[' . join(',', map { sprintf('%.6f', sin($_ * 0.37 + $q * 1.9)) } (1 .. $dim)) . ']';
}

# Recall@10 of the index on table t$dim over 10 queries, against exact search
sub recall
{
	my ($dim, $op, $settings) = @_;
	my $hits = 0;
	my $plan = $node->safe_psql('postgres', qq{
		SET enable_seqscan = off;
		$settings
		EXPLAIN (COSTS OFF) SELECT id FROM t$dim ORDER BY v $op '@{[query_vector($dim, 1)]}' LIMIT 10;
	});

	like($plan, qr/Index Scan using/, "$dim-d $op $settings: uses the index");

	for my $q (1 .. 10)
	{
		my $qv = query_vector($dim, $q);
		my $truth = $node->safe_psql('postgres',
			"SET enable_indexscan = off; SELECT id FROM t$dim ORDER BY v $op '$qv' LIMIT 10;");
		my $actual = $node->safe_psql('postgres', qq{
			SET enable_seqscan = off;
			SET hnsw.ef_search = 100;
			$settings
			SELECT id FROM t$dim ORDER BY v $op '$qv' LIMIT 10;
		});
		my @ids = split("\n", $actual);
		my %unique = map { $_ => 1 } @ids;
		is(scalar(@ids), 10, "$dim-d $op $settings: 10 results");
		is(scalar(keys %unique), 10, "$dim-d $op $settings: no duplicates");
		my %expected = map { $_ => 1 } split("\n", $truth);
		$hits += scalar grep { exists $expected{$_} } @ids;
	}
	return $hits / 100;
}

my @metrics = (['<->', 'vector_l2_ops'], ['<=>', 'vector_cosine_ops'], ['<#>', 'vector_ip_ops']);

# CREATE INDEX: all metrics, every read path
for my $metric (@metrics)
{
	my ($op, $ops) = @$metric;

	$node->safe_psql('postgres',
		"CREATE INDEX idx ON t96 USING hnsw (v $ops) WITH (m = 16, ef_construction = 64, quantizer = 'tq4');");
	for my $optimistic ('off', 'on')
	{
		for my $prefetch ('off', 'on')
		{
			my $r = recall(96, $op, "SET calico.optimistic_reads = $optimistic; SET hnsw.enable_prefetch = $prefetch;");
			cmp_ok($r, '>=', 0.9, "tq4 $op optimistic=$optimistic prefetch=$prefetch: recall $r");
		}
	}
	$node->safe_psql('postgres', 'DROP INDEX idx;');
}

# Size, and re-rank off
{
	$node->safe_psql('postgres', "CREATE INDEX idx_float ON t96 USING hnsw (v vector_l2_ops) WITH (m = 16);");
	my $float_size = $node->safe_psql('postgres', "SELECT pg_relation_size('idx_float');");
	$node->safe_psql('postgres', 'DROP INDEX idx_float;');

	$node->safe_psql('postgres', "CREATE INDEX idx ON t96 USING hnsw (v vector_l2_ops) WITH (m = 16, quantizer = 'tq4');");
	my $tq_size = $node->safe_psql('postgres', "SELECT pg_relation_size('idx');");
	my $ntq12 = recall(96, '<->', '');
	$node->safe_psql('postgres', 'DROP INDEX idx;');
	cmp_ok($tq_size, '<', $float_size, "tq4 index ($tq_size bytes) is smaller than the float index ($float_size)");

	$node->safe_psql('postgres', "CREATE INDEX idx ON t96 USING hnsw (v vector_l2_ops) WITH (m = 16, quantizer = 'tq4', rerank = 'none');");
	my $none_size = $node->safe_psql('postgres', "SELECT pg_relation_size('idx');");
	my $none = recall(96, '<->', '');
	cmp_ok($none, '<=', $ntq12, "re-rank off: recall $none, ntq12 $ntq12");
	cmp_ok($none_size, '<', $tq_size, 're-rank off: no re-rank pages');
	$node->safe_psql('postgres', 'DROP INDEX idx;');
}

# What is not supported yet
{
	$node->safe_psql('postgres', "CREATE INDEX idx ON t96 USING hnsw (v vector_l2_ops) WITH (quantizer = 'tq4');");
	my ($ret, $stdout, $stderr);

	# First, while the table has no dead rows (a failed INSERT leaves one)
	($ret, $stdout, $stderr) = $node->psql('postgres', 'VACUUM t96;');
	is($ret, 0, 'VACUUM without deleted rows works');

	($ret, $stdout, $stderr) = $node->psql('postgres', "INSERT INTO t96 VALUES (5000, (SELECT v FROM t96 WHERE id = 1));");
	like($stderr, qr/do not support inserts yet/, 'insert fails');

	($ret, $stdout, $stderr) = $node->psql('postgres', qq{
		SET enable_seqscan = off; SET hnsw.iterative_scan = relaxed_order;
		SELECT id FROM t96 ORDER BY v <-> '@{[query_vector(96, 1)]}' LIMIT 5;
	});
	like($stderr, qr/do not support iterative scans yet/, 'iterative scan fails');

	$node->safe_psql('postgres', 'DELETE FROM t96 WHERE id = 2;');
	($ret, $stdout, $stderr) = $node->psql('postgres', 'VACUUM t96;');
	like($stderr, qr/do not support vacuuming deleted rows yet/, 'VACUUM with deleted rows fails');
	$node->safe_psql('postgres', 'DROP INDEX idx;');

	($ret, $stdout, $stderr) = $node->psql('postgres', "CREATE INDEX ON t96 USING hnsw (v vector_l1_ops) WITH (quantizer = 'tq4');");
	like($stderr, qr/requires a vector column with L2, inner product or cosine distance/, 'L1 refused');

	$node->safe_psql('postgres', 'CREATE TABLE h (v halfvec(96)); INSERT INTO h SELECT v FROM t96 LIMIT 10;');
	($ret, $stdout, $stderr) = $node->psql('postgres', "CREATE INDEX ON h USING hnsw (v halfvec_l2_ops) WITH (quantizer = 'tq4');");
	like($stderr, qr/requires a vector column/, 'halfvec refused');

	$node->safe_psql('postgres', 'CREATE TABLE d (v vector(1000)); INSERT INTO d SELECT array_fill(1, ARRAY[1000])::vector;');
	($ret, $stdout, $stderr) = $node->psql('postgres', "CREATE INDEX ON d USING hnsw (v vector_l2_ops) WITH (quantizer = 'tq4');");
	like($stderr, qr/does not support 1000 dimensions/, '1000 dimensions (r = 125) refused');

	($ret, $stdout, $stderr) = $node->psql('postgres', "CREATE INDEX ON t96 USING hnsw (v vector_l2_ops) WITH (quantizer = 'tq8');");
	like($stderr, qr/invalid value for enum option "quantizer"/, 'unknown quantizer refused');
}

# A graph-file build (as from pipnn_export): each row's exact 16 nearest
# neighbours, nodes in heap order
{
	my $dim = 64;
	my $rows = $node->safe_psql('postgres', qq{
		SELECT a.id, (a.ctid::text::point)[0]::bigint, (a.ctid::text::point)[1]::bigint,
			array_to_string(ARRAY(SELECT b.id FROM t$dim b WHERE b.id <> a.id ORDER BY a.v <-> b.v LIMIT 16), ',')
		FROM t$dim a ORDER BY a.ctid;
	});
	my @lines = split("\n", $rows);
	my $n = scalar(@lines);
	my (%node_of, @tids, @neighbours);
	for my $i (0 .. $#lines)
	{
		my ($id, $blk, $off, $nb) = split(/\|/, $lines[$i]);
		$node_of{$id} = $i;
		push @tids, [$blk, $off];
		push @neighbours, [split(/,/, $nb)];
	}

	my $path = File::Spec->rel2abs($node->basedir . '/t64.graph');
	open(my $fh, '>:raw', $path) or die "cannot write $path: $!";
	print $fh pack('a8 V V Q< Q< V V', 'PIPNNGR1', 1, 16, $n, 0, $dim, 0);
	print $fh pack('V v v', $_->[0], $_->[1], 0) for @tids;
	for my $nb (@neighbours)
	{
		my @ids = map { $node_of{$_} } @$nb;
		push @ids, 0xFFFFFFFF while @ids < 16;
		print $fh pack('V V16', scalar(@$nb), @ids);
	}
	close($fh);

	my ($ret, $stdout, $stderr) = $node->psql('postgres', qq{
		SET hnsw.build_graph_file = '$path';
		CREATE INDEX idx ON t$dim USING hnsw (v vector_l2_ops) WITH (m = 8, quantizer = 'tq4');
	});
	is($ret, 0, 'tq4 graph-file build');
	like($stderr, qr/imported graph .*elements as tq4 codes/, 'the import reports tq4 codes');

	for my $optimistic ('off', 'on')
	{
		my $r = recall($dim, '<->', "SET calico.optimistic_reads = $optimistic;");
		cmp_ok($r, '>=', 0.9, "imported tq4 graph, optimistic=$optimistic: recall $r");
	}
	my $tq_size = $node->safe_psql('postgres', "SELECT pg_relation_size('idx');");
	$node->safe_psql('postgres', 'DROP INDEX idx;');

	$node->safe_psql('postgres', qq{
		SET hnsw.build_graph_file = '$path';
		CREATE INDEX idx ON t$dim USING hnsw (v vector_l2_ops) WITH (m = 8);
	});
	my $float = recall($dim, '<->', '');
	my $float_size = $node->safe_psql('postgres', "SELECT pg_relation_size('idx');");
	cmp_ok($float, '>=', 0.9, "imported float graph: recall $float");
	cmp_ok($tq_size, '<', $float_size, "imported tq4 index ($tq_size bytes) is smaller than the float one ($float_size)");
	$node->safe_psql('postgres', 'DROP INDEX idx;');
}

$node->stop;
done_testing();
