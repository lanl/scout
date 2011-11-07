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

#
# Try to work-around an "error" in cuda.h where there is a 
# trailing comma in an enumerated type.  This appears to 
# make things happy on Linux but the Mac still fails -- in
# this case we've been forced to edit cuda.h manually and 
# remove the offending comma...
#
SET(CMAKE_CXX_FLAGS "-std=c99")

add_llvm_library(scCudaError
  CheckCudaError.cpp
)

