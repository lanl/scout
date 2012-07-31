#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

add_subdirectory(Driver)

if(${CUDA_FOUND})
  add_subdirectory(CudaError)
endif()

add_subdirectory(DoallToPTX)
add_subdirectory(DoallToAMDIL)
