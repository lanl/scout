project(scc)

include(sc-llvm-link-libs) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fno-rtti")

add_llvm_definitions( -D__STDC_CONSTANT_MACROS )
add_llvm_definitions( -D__STDC_FORMAT_MACROS )
add_llvm_definitions( -D__STDC_LIMIT_MACROS )

include_directories(
  ${SC_INC_PATH}
  ${SC_LLVM_INC_PATH}
  ${SC_CLANG_INC_PATH}
  ${CMAKE_INSTALL_PREFIX}/include 
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  irreader
  codegen
  instrumentation
  ipo
  linker
  selectiondag
  )

add_clang_executable(scc
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  ScoutVisitor.cpp
  )

target_link_libraries(scc
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
  clangRewriteCore
  clangRewriteFrontend
  clangSema
  clangSerialization
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  ${SC_LLVM_LINK_LIBS}
  scConfig
  )

set_target_properties(scc PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
set_target_properties(scc PROPERTIES ENABLE_EXPORTS 1)

add_dependencies(scc
  clang-headers
  scConfig
  clang
  )

install(TARGETS scc
  RUNTIME DESTINATION bin)
