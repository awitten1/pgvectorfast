srcdir ?= $(patsubst %/,%,$(dir $(firstword $(MAKEFILE_LIST))))

EXTENSION = vector calico vectorfast
EXTVERSION = 0.8.7

MODULE_big = vectorfast
DATA = $(patsubst $(srcdir)/%,%,$(wildcard $(srcdir)/sql/*--*--*.sql))
DATA_built = sql/vector--$(EXTVERSION).sql
DATA += sql/calico--0.1.sql sql/calico--0.2.sql sql/vectorfast--0.2.1.sql
OBJS = src/calico.o src/calico_pool.o src/calico_lifecycle.o src/calico_intercept.o src/calico_bufmgr.o src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard $(srcdir)/test/sql/*.sql)
REGRESS = $(notdir $(TESTS:.sql=))
REGRESS_OPTS = --inputdir=$(srcdir)/test --load-extension=vector

# To compile for portability, run: make OPTFLAGS=""
OPTFLAGS = -march=native

# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# no difference with -march=armv8.5-a
		OPTFLAGS =
	endif
endif

# PowerPC doesn't support -march=native
ifneq ($(filter ppc64%, $(shell uname -m)), )
	OPTFLAGS =
endif

# RISC-V64 doesn't support -march=native
ifeq ($(shell uname -m), riscv64)
	OPTFLAGS =
endif

# For auto-vectorization:
# - GCC (needs -ftree-vectorize OR -O3) - https://gcc.gnu.org/projects/tree-ssa/vectorization.html
# - Clang (could use pragma instead) - https://llvm.org/docs/Vectorizers.html
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math -ffp-contract=fast

# Debug GCC auto-vectorization
# PG_CFLAGS += -fopt-info-vec

# Debug Clang auto-vectorization
# PG_CFLAGS += -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize

all: sql/vector--$(EXTVERSION).sql

sql/vector--$(EXTVERSION).sql: sql/vector.sql
	cp $< $@

PG_CONFIG ?= pg_config
PG_MAJOR_CALICO := $(shell $(PG_CONFIG) --version | sed -E 's/.* ([0-9]+).*/\1/')
ifndef CAPSTONE_PREFIX
ifneq ($(wildcard /usr/local/include/capstone/capstone.h),)
CAPSTONE_PREFIX := /usr/local
else ifneq ($(wildcard /usr/include/capstone/capstone.h),)
CAPSTONE_PREFIX := /usr
else
CAPSTONE_DEPS_DIR := $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/pgvectorfast/python-deps
CAPSTONE_PREFIX := $(CAPSTONE_DEPS_DIR)/capstone
endif
endif
PG_CPPFLAGS += -I$(srcdir)/src
ifneq ($(CAPSTONE_PREFIX),/usr)
PG_CPPFLAGS += -I$(CAPSTONE_PREFIX)/include
SHLIB_LINK += -L$(CAPSTONE_PREFIX)/lib -Wl,-rpath,$(CAPSTONE_PREFIX)/lib
endif
PG_CFLAGS += -fcf-protection=branch -Werror=implicit-function-declaration
SHLIB_LINK += -lcapstone -ldl

ifneq ($(CAPSTONE_DEPS_DIR),)
CAPSTONE_HEADER := $(CAPSTONE_PREFIX)/include/capstone/capstone.h
CAPSTONE_SOLINK := $(CAPSTONE_PREFIX)/lib/libcapstone.so.5
ifeq ($(filter clean distclean maintainer-clean uninstall,$(MAKECMDGOALS)),)
ifeq ($(if $(and $(wildcard $(MODULE_big)$(DLSUFFIX)),$(filter install installdirs,$(MAKECMDGOALS)),$(if $(filter-out install installdirs,$(MAKECMDGOALS)),,skip)),skip),)
$(CAPSTONE_HEADER):
	mkdir -p "$(CAPSTONE_DEPS_DIR)"
	python3 -m pip install --only-binary=:all: --target "$(CAPSTONE_DEPS_DIR)" capstone==5.0.6

$(CAPSTONE_SOLINK): $(CAPSTONE_HEADER)
	@if [ ! -e "$@" ] && [ -e "$(CAPSTONE_PREFIX)/lib/libcapstone.so" ]; then \
		ln -sf libcapstone.so "$@"; \
	fi

$(OBJS): | $(CAPSTONE_HEADER) $(CAPSTONE_SOLINK)
endif
endif
endif

$(OBJS): $(srcdir)/src/calico_internal.h $(srcdir)/Makefile
src/calico_bufmgr.o src/calico_bufmgr.bc: $(srcdir)/src/calico_bufmgr$(PG_MAJOR_CALICO).inc

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Copied PostgreSQL routines keep the server's arithmetic/compiler semantics.
# Vector-specific SIMD/fast-math flags above apply to pgvector's own sources.
# PIC/visibility and CET are required by the extension's integration boundary.
CALICO_CORE_CFLAGS ?= $(shell $(PG_CONFIG) --cflags) $(OPTFLAGS)
src/calico.o src/calico_pool.o src/calico_lifecycle.o src/calico_intercept.o src/calico_bufmgr.o: override CFLAGS = $(CALICO_CORE_CFLAGS) $(CFLAGS_SL) $(CFLAGS_SL_MODULE) -fcf-protection=branch -Werror=implicit-function-declaration

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

# for Postgres < 15
PROVE_FLAGS += -I ./test/perl

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=pgvectorfast-$(EXTVERSION)/ --output dist/pgvectorfast-$(EXTVERSION).zip HEAD
