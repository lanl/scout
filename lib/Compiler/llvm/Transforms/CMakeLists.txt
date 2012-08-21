#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#


# Is this all CUDA-dependent?
add_subdirectory(Driver)

if(${SC_ENABLE_CUDA})
  add_subdirectory(CudaError)
  add_subdirectory(DoallToPTX)
endif()

if (${SC_ENABLE_OPENCL})
  add_subdirectory(DoallToAMDIL)
endif()

