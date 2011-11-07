#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scStandard)
enable_testing()

set(CMAKE_C_COMPILER ${LLVM_BINARY_DIR}/bin/scc)
set(CMAKE_CXX_COMPILER ${LLVM_BINARY_DIR}/bin/scc)

include_directories(
  ${SCOUT_INCLUDE_DIR}
)

file(GLOB std_hdrs ${SCOUT_INCLUDE_DIR}/scout/*.h)
file(GLOB std_srcs *.cpp)


  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
if (SCOUT_ENABLE_RT_THREADS) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()


add_library(scStandard STATIC ${std_srcs})
add_dependencies(scStandard scc)

