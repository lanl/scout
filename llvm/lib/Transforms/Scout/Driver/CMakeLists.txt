#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

project(scDriver)
include(scout-llvm-defs)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${SCOUT_INCLUDE_DIR}
)

file(GLOB driver_srcs *.cpp)

SET(CMAKE_CXX_FLAGS "-fPIC")

if (SCOUT_ENABLE_RT_THREADS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_library(scDriver STATIC ${driver_srcs})

