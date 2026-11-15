# syntax=docker/dockerfile:1

ARG PG_MAJOR=18
ARG DEBIAN_CODENAME=trixie
FROM postgres:$PG_MAJOR-$DEBIAN_CODENAME
ARG PG_MAJOR

COPY . /tmp/pgvectorfast

RUN apt-get update && \
		apt-mark hold locales && \
		apt-get install -y --no-install-recommends build-essential pkg-config libcapstone-dev postgresql-server-dev-$PG_MAJOR postgresql-$PG_MAJOR-dbgsym && \
		CAPSTONE_RUNTIME=$(dpkg-query -W -f='${Package}\n' 'libcapstone[0-9]*' | head -n 1) && \
		apt-mark manual "$CAPSTONE_RUNTIME" && \
		sed -i -E "s|^#?shared_preload_libraries = ''|shared_preload_libraries = 'vectorfast'|" "/usr/share/postgresql/$PG_MAJOR/postgresql.conf.sample" && \
		cd /tmp/pgvectorfast && \
		make clean && \
		make -j"$(nproc)" OPTFLAGS="" && \
		make install && \
		mkdir -p /usr/share/doc/pgvectorfast && \
		cp LICENSE README.md BENCHMARKS.md PGVECTOR.md /usr/share/doc/pgvectorfast && \
		cp -r examples /usr/share/doc/pgvectorfast && \
		cd / && \
		rm -r /tmp/pgvectorfast && \
		apt-get remove -y build-essential pkg-config libcapstone-dev postgresql-server-dev-$PG_MAJOR && \
		apt-get autoremove -y && \
		apt-mark unhold locales && \
		rm -rf /var/lib/apt/lists/*
