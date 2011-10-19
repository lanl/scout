#
# ----- The Scout Programming Language
# 
# This file is distributed under an open source license by Los Alamos
# National Security, LCC.  See the file LICENSE.txt for details. 
#
# See the file docs/build.html for instructions on how to build 
# Scout with CMake. 
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-rtti") 
set(CLANG_ARGS -g -S -emit-ast -x c++ -fno-rtti) 

set( LLVM_USED_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangIndex
  clangLex
  clangParse
  clangARCMigrate
  clangRewrite
  clangSema
  clangSerialization
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  instrumentation
  ipo
  selectiondag
  )

add_clang_executable(scc
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})

if(${APPLE})
  find_library(FOUNDATION_FRAMEWORK NAMES Foundation)
  target_link_libraries(scc ${FOUNDATION_FRAMEWORK})
endif()

file(COPY scc-ast DESTINATION ${SCOUT_BINARY_DIR}/tools/scc)

install(TARGETS scc RUNTIME DESTINATION bin)
