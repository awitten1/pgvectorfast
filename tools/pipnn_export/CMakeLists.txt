# pipnn_export: builds a PiPNN graph (Ginger, third_party/ginger) over vectors
# SELECTed out of Postgres, for CREATE INDEX with hnsw.build_graph_file.
#
#   git submodule update --init third_party/ginger
#   cmake -S tools/pipnn_export -B tools/pipnn_export/build -G Ninja \
#         -DCMAKE_BUILD_TYPE=Release -DPG_CONFIG=/path/to/pg_config
#   cmake --build tools/pipnn_export/build --target pipnn_export
#
# Ginger fetches its own dependencies (abseil, protobuf, highway, oneTBB,
# KaMinPar, ...) on the first configure, which needs network and takes a while.
# Ginger's default copts include -march=native, so build on the machine that
# runs the tool.
cmake_minimum_required(VERSION 3.24)
project(pipnn_export LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GINGER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ginger)
if(NOT EXISTS ${GINGER_SRC}/CMakeLists.txt)
  message(FATAL_ERROR "third_party/ginger is empty; run: git submodule update --init third_party/ginger")
endif()

# Only the library is needed.
set(GINGER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GINGER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GINGER_SRC} ginger EXCLUDE_FROM_ALL)

# Ginger's default copts (-march=native, -funsigned-char, -fno-exceptions) and
# linkopts are variables in its directory scope, invisible here. Read them out:
# -funsigned-char must match the library, since Ginger's headers inline code
# into this file.
get_directory_property(GINGER_DEFAULT_COPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_COPTS)
get_directory_property(GINGER_DEFAULT_LINKOPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_LINKOPTS)
if(NOT GINGER_DEFAULT_COPTS)
  message(FATAL_ERROR "could not read GINGER_DEFAULT_COPTS from ${GINGER_SRC}")
endif()
find_package(Threads REQUIRED)

# libpq, from the Postgres the extension is built against.
set(PG_CONFIG pg_config CACHE STRING "pg_config of the Postgres to read from")
execute_process(COMMAND ${PG_CONFIG} --includedir OUTPUT_VARIABLE PG_INCLUDEDIR
                OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE pg_rc)
execute_process(COMMAND ${PG_CONFIG} --libdir OUTPUT_VARIABLE PG_LIBDIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT pg_rc EQUAL 0)
  message(FATAL_ERROR "cannot run ${PG_CONFIG}; pass -DPG_CONFIG=/path/to/pg_config")
endif()
find_library(PQ_LIBRARY NAMES pq HINTS ${PG_LIBDIR} REQUIRED NO_DEFAULT_PATH)

add_executable(pipnn_export pipnn_export.cc)
target_compile_features(pipnn_export PRIVATE cxx_std_20)
target_compile_options(pipnn_export PRIVATE ${GINGER_DEFAULT_COPTS})
target_include_directories(pipnn_export PRIVATE ${PG_INCLUDEDIR})
target_link_libraries(pipnn_export PRIVATE
  ginger::ginger
  absl::status
  absl::statusor
  absl::span
  protobuf::libprotobuf
  ${PQ_LIBRARY}
  ${GINGER_DEFAULT_LINKOPTS})
set_target_properties(pipnn_export PROPERTIES BUILD_RPATH ${PG_LIBDIR} INSTALL_RPATH ${PG_LIBDIR})
