#!/usr/bin/env bash
set -euo pipefail

PG_MAJOR="${PG_MAJOR:-18}"
PGVECTORFAST_VERSION="${PGVECTORFAST_VERSION:-0.1.0}"
PG_CONFIG="${PG_CONFIG:-/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config}"
OUTDIR="${OUTDIR:-$(pwd)/dist}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="$(dpkg --print-architecture)"
PG_PKG="postgresql-${PG_MAJOR}"
PG_DBG_PKG="postgresql-${PG_MAJOR}-dbgsym"
PG_PKG_VER="$(dpkg-query -W -f='${Version}' "${PG_PKG}")"
PG_DBG_VER="$(dpkg-query -W -f='${Version}' "${PG_DBG_PKG}")"

PKG_NAME="postgresql-${PG_MAJOR}-pgvectorfast"
PKG_VER="${PGVECTORFAST_VERSION#v}+pg${PG_PKG_VER}"
if [[ "${PKG_VER}" != *-* ]]; then
    PKG_VER="${PKG_VER}-1"
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

SRC_COPY="${WORKDIR}/src"
STAGE_DIR="${WORKDIR}/stage"
SHLIBS_WORK="${WORKDIR}/shlibs"

cp -a "${ROOT_DIR}" "${SRC_COPY}"
make -C "${SRC_COPY}" clean >/dev/null 2>&1 || true
make -C "${SRC_COPY}" -j"$(nproc)" PG_CONFIG="${PG_CONFIG}" OPTFLAGS=""
make -C "${SRC_COPY}" PG_CONFIG="${PG_CONFIG}" DESTDIR="${STAGE_DIR}" install

DOC_DIR="${STAGE_DIR}/usr/share/doc/${PKG_NAME}"
mkdir -p "${DOC_DIR}" "${STAGE_DIR}/DEBIAN" "${SHLIBS_WORK}/debian" "${OUTDIR}"
cp "${ROOT_DIR}/packaging/apt/copyright" "${DOC_DIR}/copyright"
cp "${ROOT_DIR}/README.md" "${ROOT_DIR}/BENCHMARKS.md" "${ROOT_DIR}/PGVECTOR.md" "${DOC_DIR}/"
cp -r "${ROOT_DIR}/examples" "${DOC_DIR}/"

# Compute exact runtime shlib dependencies (e.g. libcapstone4 on bookworm/ubuntu,
# libcapstone5 on trixie) via dpkg-shlibdeps from a minimal synthetic debian/control.
cat > "${SHLIBS_WORK}/debian/control" <<EOF
Source: pgvectorfast
Section: database
Priority: optional
Maintainer: Xinjing Zhou <zxjcarrot@gmail.com>

Package: ${PKG_NAME}
Architecture: ${ARCH}
EOF

PKGLIBDIR="$("${PG_CONFIG}" --pkglibdir)"
SHLIBS_DEP="$(
    cd "${SHLIBS_WORK}" && \
    dpkg-shlibdeps -O "${STAGE_DIR}${PKGLIBDIR}/vectorfast.so" | sed -n 's/^shlibs:Depends=//p'
)"
DEPS="${PG_PKG} (= ${PG_PKG_VER}), ${PG_DBG_PKG} (= ${PG_DBG_VER})"
if [[ -n "${SHLIBS_DEP}" ]]; then
    DEPS="${SHLIBS_DEP}, ${DEPS}"
fi
INSTALLED_SIZE="$(du -ks "${STAGE_DIR}" | awk '{print $1}')"

cat > "${STAGE_DIR}/DEBIAN/control" <<EOF
Package: ${PKG_NAME}
Version: ${PKG_VER}
Architecture: ${ARCH}
Maintainer: Xinjing Zhou <zxjcarrot@gmail.com>
Installed-Size: ${INSTALLED_SIZE}
Section: database
Priority: optional
Homepage: https://github.com/mitdbg/pgvectorfast
Depends: ${DEPS}
Conflicts: postgresql-${PG_MAJOR}-pgvector
Replaces: postgresql-${PG_MAJOR}-pgvector
Provides: postgresql-${PG_MAJOR}-pgvector
Description: Fast pgvector + Calico shared buffer pool extension for PostgreSQL ${PG_MAJOR}
 Drop-in replacement for pgvector providing the vector, halfvec, sparsevec,
 and bit types alongside the Calico array-based shared buffer pool manager,
 optimistic lock-free HNSW reads, and batched group prefetching.
EOF

DEB_PATH="${OUTDIR}/${PKG_NAME}_${PKG_VER}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "${STAGE_DIR}" "${DEB_PATH}"
dpkg-deb -I "${DEB_PATH}"
echo "Built ${DEB_PATH}"
