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
 ${CMAKE_BINARY_DIR}/include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_llvm_library(LLVMLegionTaskWrapper
      LegionTaskWrapper.cpp
    )
add_dependencies(LLVMLegionTaskWrapper intrinsics_gen)
