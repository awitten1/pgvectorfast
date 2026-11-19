use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('filtered_fast');
$node->init;
$node->append_conf('postgresql.conf', "shared_preload_libraries = 'vectorfast'\nshared_buffers = '16MB'\njit = off\n");
$node->start;
$node->safe_psql('postgres', 'CREATE EXTENSION vectorfast;');
$node->safe_psql('postgres', q{
    SELECT setseed(0.2718);
    CREATE TABLE items (id integer PRIMARY KEY, v vector(17)) WITH (autovacuum_enabled=false);
    INSERT INTO items SELECT i, ARRAY(SELECT (random() * 2 - 1)::real
        FROM generate_series(1,17) j WHERE i > 0) FROM generate_series(1,3000) i;
});
my @queries = map { my $q=$_; '[' . join(',', map { sin($_ * 0.71 + $q) } (1 .. 17)) . ']' } (1..3);
my @metrics = (['<->','vector_l2_ops'], ['<=>','vector_cosine_ops'], ['<#>','vector_ip_ops']);
for my $metric (@metrics)
{
    my ($op,$ops)=@$metric;
    $node->safe_psql('postgres', "CREATE INDEX idx ON items USING hnsw (v $ops);");
    my @truth = map { $node->safe_psql('postgres',
        "SELECT id FROM items WHERE id % 31 = 0 ORDER BY (v $op '$_') + 0 LIMIT 10;") } @queries;
    for my $mode ('strict_order','relaxed_order')
    {
        for my $optimistic ('off','on')
        {
            for my $prefetch ('off','on')
            {
                my $hits=0;
                for my $i (0..$#queries)
                {
                    my $actual=$node->safe_psql('postgres', qq{
                        SET enable_seqscan=off;
                        SET hnsw.ef_search=40;
                        SET hnsw.max_scan_tuples=100000;
                        SET hnsw.iterative_scan=$mode;
                        SET calico.optimistic_reads=$optimistic;
                        SET hnsw.enable_prefetch=$prefetch;
                        WITH candidates AS MATERIALIZED (
                            SELECT id, v $op '$queries[$i]' AS d FROM items
                            WHERE id % 31=0 ORDER BY d LIMIT 50)
                        SELECT id FROM candidates ORDER BY d+0 LIMIT 10;
                    });
                    my @ids=split("\n",$actual);
                    my %unique=map {$_=>1} @ids;
                    is(scalar(@ids),10,"$op $mode opt=$optimistic prefetch=$prefetch complete");
                    is(scalar(keys %unique),10,'no duplicate results');
                    my %expected=map {$_=>1} split("\n",$truth[$i]);
                    $hits += scalar grep {exists $expected{$_}} @ids;
                }
                cmp_ok($hits/30,'>=',0.95,"$op $mode opt=$optimistic prefetch=$prefetch recall");
            }
        }
    }
    my $ordered=$node->safe_psql('postgres', qq{
        SET enable_seqscan=off; SET hnsw.ef_search=8;
        SET hnsw.iterative_scan=strict_order;
        SELECT v $op '$queries[0]' FROM items WHERE id % 31=0
        ORDER BY v $op '$queries[0]' LIMIT 30;
    });
    my @distances=split("\n",$ordered);
    ok(!scalar(grep {$distances[$_] < $distances[$_-1]} (1..$#distances)),"$op strict ordering across batches");
    my $counts=$node->safe_psql('postgres', qq{
        SET enable_seqscan=off; SET hnsw.ef_search=8;
        SET hnsw.iterative_scan=relaxed_order; SET hnsw.max_scan_tuples=1;
        SET work_mem='64kB';
        SELECT count(*)=count(DISTINCT id) FROM
          (SELECT id FROM items WHERE id % 31=0 ORDER BY v $op '$queries[0]' LIMIT 100) s;
    });
    is($counts,'t',"$op draining pending candidates under scan/memory limits");
    my $rescans=$node->safe_psql('postgres', qq{
        SET enable_seqscan=off; SET hnsw.iterative_scan=relaxed_order;
        SELECT string_agg(n::text,',' ORDER BY q) FROM (VALUES (1),(2),(3)) q(q)
        CROSS JOIN LATERAL (SELECT count(*) n FROM
          (SELECT id FROM items WHERE id % 31=q ORDER BY v $op '$queries[0]' LIMIT 10) s) t;
    });
    is($rescans,'10,10,10',"$op parameterized rescans");
    $node->pgbench('--no-vacuum --client=3 --transactions=100',0,
        [qr{actually processed}],[qr{^$}],"$op concurrent iterative search and vacuum",{
        "${ops}_search\@10"=>"SET enable_seqscan=off; SET hnsw.iterative_scan=relaxed_order; SET hnsw.ef_search=8; SELECT id FROM items WHERE id % 31=0 ORDER BY v $op '$queries[0]' LIMIT 10;",
        "${ops}_vacuum\@1"=>'VACUUM items;',
        "${ops}_update\@1"=>'UPDATE items SET v=v WHERE id % 101=0;'});
    $node->safe_psql('postgres','DROP INDEX idx;');
}
# Duplicate vectors can store several heap TIDs in one graph element. Check
# that the compact arena preserves all of them across result batches.
$node->safe_psql('postgres', q{
    CREATE TABLE duplicates AS SELECT i AS id, ARRAY[(i % 25)::real, 1, 2]::vector(3) AS v
      FROM generate_series(1,500) i;
    CREATE INDEX duplicates_idx ON duplicates USING hnsw (v vector_cosine_ops);
});
my $duplicate_truth = join(',', grep {$_ % 3 == 0} (1..500));
for my $mode ('strict_order','relaxed_order')
{
    for my $optimistic ('off','on')
    {
        my $ids=$node->safe_psql('postgres', qq{
            SET enable_seqscan=off; SET hnsw.ef_search=4;
            SET hnsw.iterative_scan=$mode; SET calico.optimistic_reads=$optimistic;
            SET hnsw.max_scan_tuples=10000;
            SELECT string_agg(id::text,',' ORDER BY id) FROM
              (SELECT id FROM duplicates WHERE id % 3=0
               ORDER BY v <=> '[1,1,2]' LIMIT 1000) s;
        });
        is($ids,$duplicate_truth,"$mode opt=$optimistic retains duplicate-vector heap TIDs");
    }
}
# Keep discarded candidates alive while VACUUM removes dead elements and
# subsequent inserts reuse their slots. The cursor must retain its snapshot.
$node->safe_psql('postgres', q{
    CREATE INDEX idx ON items USING hnsw (v vector_cosine_ops);
    DELETE FROM items WHERE id % 2=1;
});
my $cursor=$node->background_psql('postgres');
$cursor->query_safe(qq{
    SET enable_seqscan=off; SET hnsw.ef_search=8;
    SET hnsw.iterative_scan=relaxed_order; SET hnsw.max_scan_tuples=100000;
    BEGIN;
    DECLARE scan_cursor CURSOR FOR SELECT id FROM items WHERE id % 31=0
      ORDER BY v <=> '$queries[0]' LIMIT 100;
});
my $first=$cursor->query_safe('FETCH 1 FROM scan_cursor;');
$node->safe_psql('postgres', 'VACUUM items;');
$node->safe_psql('postgres', q{
    INSERT INTO items SELECT 10000+i, ARRAY(SELECT (random()*2-1)::real
      FROM generate_series(1,17) j WHERE i>0) FROM generate_series(1,3000) i;
});
my $rest=$cursor->query_safe('FETCH ALL FROM scan_cursor;');
my @visible=split("\n", "$first\n$rest");
cmp_ok(scalar(@visible),'>=',10,'cursor resumes after vacuum and slot reuse');
ok(!scalar(grep {$_ >= 10000 || $_ % 2 || $_ % 31} @visible),
   'resumed cursor returns only original visible qualifying tuples');
my %seen=map {$_=>1} @visible;
is(scalar(keys %seen),scalar(@visible),'no duplicates after vacuum and slot reuse');
$cursor->query_safe('COMMIT;');
$cursor->quit;
$node->stop;
done_testing();
