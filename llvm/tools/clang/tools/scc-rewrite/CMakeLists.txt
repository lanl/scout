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

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  instrumentation
  ipo
  linker
  selectiondag
  )

add_clang_executable(scc-rewrite
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  )

target_link_libraries(scc-rewrite
  clangFrontendTool
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangEdit
  clangARCMigrate
  clangRewrite
  clangSema
  clangSerialization
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  LLVMdoallToPTX
  scDriver
  )

if(${APPLE})
  find_library(FOUNDATION_FRAMEWORK NAMES Foundation)
  if(SCOUT_ENABLE_LIB_NVVM)
    find_library(LIB_NVVM NAMES nvvm 
                 HINTS ${SCOUT_SRC_DIR}/lib/scc/Mac/lib)
    target_link_libraries(scc-rewrite ${FOUNDATION_FRAMEWORK} ${LIB_NVVM})
  endif()
else()
  if(SCOUT_ENABLE_LIB_NVVM)
    find_library(LIB_NVVM NAMES nvvm 
                 HINTS ${SCOUT_SRC_DIR}/lib/scc/Linux/lib)
    target_link_libraries(scc-rewrite ${FOUNDATION_FRAMEWORK} ${LIB_NVVM})
  endif()
endif()

install(TARGETS scc-rewrite RUNTIME DESTINATION bin)
