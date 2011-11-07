#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

SET(CMAKE_CXX_FLAGS "-fPIC")
add_llvm_library(scDriver 
  CudaDriver.cpp
  Driver.cpp
  PTXDriver.cpp)

add_llvm_library_dependencies(scDriver
  LLVMAnalysis
  LLVMCore
  LLVMInstCombine
  LLVMSupport
  LLVMTarget
  LLVMTransformUtils
  )

