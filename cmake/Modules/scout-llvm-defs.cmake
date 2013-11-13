#
# -----  Scout Programming Language -----
#
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file License.txt (located in the
# top level of the source distribution) for details.
#
#-----
#
#

include(AddLLVM)
include(${Scout_BINARY_DIR}/llvm/share/llvm/cmake/LLVMConfig.cmake)

# Scout-centric paths for LLVM.  Note that some header files are
# auto-generated in the LLVM build process and live within the binary
# build directories.
set(LLVM_SOURCE_DIR   ${Scout_SOURCE_DIR}/llvm)
set(LLVM_BINARY_DIR   ${Scout_BINARY_DIR}/llvm)
set(LLVM_INCLUDE_DIRS ${LLVM_SOURCE_DIR}/include ${LLVM_BINARY_DIR}/include)
set(LLVM_TOOLS_DIR    ${LLVM_BINARY_DIR}/bin)
set(LLVM_LIBRARY_DIR  ${LLVM_BINARY_DIR}/lib)


# Scout-centric paths for Clang.  Note that some header files are
# auto-generated in the Clang build process and live within the binary
# build directories.
set(CLANG_SOURCE_DIR   ${LLVM_SOURCE_DIR}/tools/clang)
set(CLANG_BINARY_DIR   ${LLVM_BINARY_DIR}/tools/clang)
set(CLANG_INCLUDE_DIRS ${CLANG_SOURCE_DIR}/include ${CLANG_BINARY_DIR}/include)
set(CLANG_LIBRARY_DIR  ${LLVM_LIBRARY_DIR})
set(CLANG_TOOLS_DIR    ${LLVM_TOOLS_DIR})

# We need these to play well with the LLVM source base -- you should
# use caution to make sure you only include this file once within a
# multi-directory project structure (or you will end up with multiple
# definitions).
ADD_DEFINITIONS(-D__STDC_LIMIT_MACROS=1)
ADD_DEFINITIONS(-D__STDC_CONSTANT_MACROS=1)

# We use the llvm-config utility to get a list of the LLVM libraries.
# This will likely produce a list of too many libraries for some of
# our needs but it is simple approach.  If compile times get long
# we might want to specialize a bit...
find_program(LLVM_CONFIG_BIN
  llvm-config
  PATHS
  ${LLVM_TOOLS_DIR}
  NO_DEFAULT_PATH
)

# Typically we would like to use llvm-config to get a list of the LLVM
# libraries we need to link with...  Unfortunatley, with the current
# dependency on LLVM and Clang within our approach we need to take a
# different approach (as CMake runs before building is complete).
set(LLVM_LIBS )

if (${SCOUT_ENABLE_CUDA})
  set(SCOUT_LLVM_LINK_LIBS LLVMdoallToPTX)
endif()

if (${SCOUT_ENABLE_OPENCL})
  set(SCOUT_LLVM_LINK_LIBS ${SCOUT_LLVM_LINK_LIBS} ${SCOUT_LLVMdoallToAMDIL)
endif()

set(SCOUT_LLVM_LINK_LIBS ${SCOUT_LLVM_LINK_LIBS} scDriver)
