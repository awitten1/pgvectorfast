#!/usr/bin/env bash
# Run inside a disposable Debian container against its installed package/image.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PG_CONFIG="/usr/lib/postgresql/${PG_MAJOR:-18}/bin/pg_config"
export PGHOST="${PGHOST:-/var/run/postgresql}"
export PGUSER=postgres
export PATH="/usr/lib/postgresql/${PG_MAJOR:-18}/bin:$PATH"
export TEST_SOURCE="${TEST_SOURCE:-/src}"
export TEST_WORK
TEST_WORK=$(mktemp -d /tmp/pgvectorfast-package-test.XXXXXX)

# Docker and APT must ship the same extension upgrade paths as source/PGXN.
for script in "$TEST_SOURCE"/sql/*--*--*.sql; do
    cmp "$script" "$("$PG_CONFIG" --sharedir)/extension/$(basename "$script")"
done

# Test tooling is installed in this disposable container; make installcheck
# exercises the shipped library and SQL files without rebuilding or installing.
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    make gcc libcapstone-dev "postgresql-server-dev-${PG_MAJOR:-18}" >/dev/null
chown postgres:postgres "$TEST_WORK"
su -p postgres -s /bin/bash <<'EOF'
set -euo pipefail
cd "$TEST_WORK"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then cat regression.diffs 2>/dev/null || true; fi; exit "$rc"' EXIT

createdb packaging_single
psql -X -d packaging_single -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION vectorfast;'
psql -X -d packaging_single -v ON_ERROR_STOP=1 -f "$TEST_SOURCE/test/packaging.sql"

createdb packaging_manual
psql -X -d packaging_manual -v ON_ERROR_STOP=1 \
    -c 'CREATE EXTENSION vector;' \
    -c 'CREATE EXTENSION calico;' \
    -c 'CALL calico.activate();'
psql -X -d packaging_manual -v ON_ERROR_STOP=1 -f "$TEST_SOURCE/test/packaging.sql"

make -f "$TEST_SOURCE/Makefile" srcdir="$TEST_SOURCE" \
    PG_CONFIG="$PG_CONFIG" installcheck
EOF
