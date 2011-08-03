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
)

file(GLOB rt_hdrs ${SCOUT_INCLUDE_DIR}/runtime/*.h)

if(SCOUT_ENABLE_RT_OPENGL) 
  file(GLOB rt_hdrs ${SCOUT_INCLUDE_DIR}/runtime/*.h ${SCOUT_INCLUDE_DIR}/runtime/opengl/*.h)
  file(GLOB rt_srcs *.cpp opengl/*.cpp)
else()
  file(GLOB rt_hdrs ${SCOUT_INCLUDE_DIR}/runtime/*.h)
  file(GLOB rt_srcs *.cpp)
endif()

if (SCOUT_ENABLE_RT_THREADS) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# We probably want the runtime to dynamic...
add_library(scRuntime STATIC ${rt_srcs})
