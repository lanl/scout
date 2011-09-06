#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
# 
project(scRuntime)
enable_testing()

include_directories(
  ${SCOUT_INCLUDE_DIR}
  ${PNG_INCLUDE_DIR}
  ${HWLOC_INCLUDE_DIR}
  ${SCOUT_SRC_DIR}/packages/hwloc/include
)

file(GLOB rt_hdrs ${SCOUT_INCLUDE_DIR}/runtime/*.h)
file(GLOB rt_srcs *.cpp)

if(SCOUT_ENABLE_RT_OPENGL) 
  file(GLOB gl_hdrs ${SCOUT_INCLUDE_DIR}/runtime/opengl/*.h)
  file(GLOB gl_srcs opengl/*.cpp)
endif()

if(SCOUT_ENABLE_PNG_SUPPORT) 
  file(GLOB png_srcs png/*.cpp)
endif()

if (SCOUT_ENABLE_RT_THREADS) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# We probably want the runtime to eventually be a dynamic library...
add_library(scRuntime STATIC ${rt_srcs} ${gl_srcs} ${png_srcs})

