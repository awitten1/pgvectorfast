# tq_check: checks the C TurboQuant codec in src/hnswtq.c against Ginger's
# TurboQuant (third_party/ginger), on random vectors.
#
#   git submodule update --init third_party/ginger
#   cmake -S tools/tq_check -B tools/tq_check/build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build tools/tq_check/build --target tq_check
#   tools/tq_check/build/tq_check
#
# Pass -DGINGER_SRC=/path/to/ginger to use another checkout of the same commit.
# Ginger fetches its dependencies on the first configure; to reuse sources a
# Ginger build has already fetched, pass -DFETCHCONTENT_SOURCE_DIR_<DEP>=...
# (see the README).
cmake_minimum_required(VERSION 3.24)
project(tq_check LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GINGER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ginger CACHE PATH "Ginger source tree")
if(NOT EXISTS ${GINGER_SRC}/CMakeLists.txt)
  message(FATAL_ERROR "${GINGER_SRC} is empty; run: git submodule update --init third_party/ginger")
endif()

set(GINGER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GINGER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GINGER_SRC} ginger EXCLUDE_FROM_ALL)

get_directory_property(GINGER_DEFAULT_COPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_COPTS)
get_directory_property(GINGER_DEFAULT_LINKOPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_LINKOPTS)
if(NOT GINGER_DEFAULT_COPTS)
  message(FATAL_ERROR "could not read GINGER_DEFAULT_COPTS from ${GINGER_SRC}")
endif()
find_package(Threads REQUIRED)

set(PGVF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

# The codec, compiled as the extension's Makefile compiles it: plain C, no
# fast-math, no FP contraction (hnswtq.c uses fmaf() where Ginger fuses).
add_library(hnswtq STATIC ${PGVF_SRC}/hnswtq.c)
target_include_directories(hnswtq PUBLIC ${PGVF_SRC})
set_target_properties(hnswtq PROPERTIES C_STANDARD 11)
target_compile_options(hnswtq PRIVATE -O2 -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(hnswtq PRIVATE -march=native)
endif()

add_executable(tq_check tq_check.cc)
target_compile_features(tq_check PRIVATE cxx_std_20)
target_compile_options(tq_check PRIVATE ${GINGER_DEFAULT_COPTS})
target_link_libraries(tq_check PRIVATE
  hnswtq
  ginger::ginger
  absl::status
  absl::statusor
  absl::span
  Threads::Threads
  ${GINGER_DEFAULT_LINKOPTS})
