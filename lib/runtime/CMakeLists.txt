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
  ${SCOUT_SRC_DIR}/packages/hwloc/include
)

set(hw_loc_lib ${SCOUT_SRC_DIR}/packages/hwloc/src/libhwloc.la)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

file(GLOB hwloc_srcs
     ${SCOUT_SRC_DIR}/packages/hwloc/src/misc.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology-synthetic.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/distances.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/cpuset.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/traversal.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology-darwin.c)

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

file(GLOB hwloc_srcs
     ${SCOUT_SRC_DIR}/packages/hwloc/src/misc.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology-synthetic.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/distances.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/cpuset.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/traversal.c
     ${SCOUT_SRC_DIR}/packages/hwloc/src/topology-linux.c)

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

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

# We probably want the runtime to dynamic...
add_library(scRuntime STATIC ${rt_srcs} ${gl_srcs} ${png_srcs} ${hwloc_srcs})

-----

#	modified:   Makefile
#	modified:   include/runtime/range.h
#	modified:   include/runtime/scout.h
#	modified:   include/runtime/tbq.h
#	modified:   lib/compiler/CMakeLists.txt
#	modified:   lib/runtime/CMakeLists.txt
#	modified:   lib/runtime/scout.cpp
#	modified:   lib/runtime/tbq.cpp
#	modified:   llvm/tools/clang/clang.xcodeproj/project.xcworkspace/xcuserdata/nickm.xcuserdatad/UserInterfaceState.xcuserstate
#	modified:   llvm/tools/clang/include/clang/AST/Stmt.h
#	modified:   llvm/tools/clang/include/clang/Parse/Parser.h
#	modified:   llvm/tools/clang/lib/CodeGen/CGDebugInfo.cpp
#	modified:   llvm/tools/clang/lib/Parse/ParseStmt.cpp
#	modified:   llvm/tools/clang/lib/Parse/Parser.cpp
#	modified:   sandbox/runtime/tbq/multi-queue.cpp
#	modified:   sandbox/runtime/tbq/multi-queue2.cpp
#	modified:   test/forall.sc

