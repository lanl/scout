#
# ----- The Scout Programming Language
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details.
#
# See the file docs/build.html for instructions on how to build
# Scout with CMake.
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

if(SCOUT_ENABLE_LIB_NVVM)
  if(${APPLE})
    include_directories(
      ${SCOUT_SRC_DIR}/lib/scc/Mac/include
    )
  else()
    include_directories(
      ${SCOUT_SRC_DIR}/lib/scc/Linux/include
    )
  endif()
endif()

add_llvm_library(LLVMdoallToPTX
  DoallToPTX.cpp
)
