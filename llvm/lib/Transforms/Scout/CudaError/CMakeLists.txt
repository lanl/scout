#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

include_directories(
  ${CUDA_INCLUDE_DIRS}
  )

set(CXXFLAGS "${CFLAGS} -std=c++98 -pedantic")

add_llvm_library(scCudaError
  CheckCudaError.cpp
)
