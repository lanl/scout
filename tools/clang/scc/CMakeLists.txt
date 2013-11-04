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

# Create scc++ symlink at installation time.
install(SCRIPT scc_symlink.cmake -DCMAKE_INSTALL_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")

# Configure plist creation for OS X.
set (TOOL_INFO_PLIST "Info.plist" CACHE STRING "Plist name")
if (APPLE)
  if (SCOUT_VENDOR)
    set(TOOL_INFO_NAME "${SCOUT_VENDOR} scc")
  else()
    set(TOOL_INFO_NAME "scc")
  endif()

  set(TOOL_INFO_UTI "${SCC_VENDOR_UTI}")
  set(TOOL_INFO_VERSION "${SCC_VERSION}")
  if (LLVM_SUBMIT_VERSION)
    set(TOOL_INFO_BUILD_VERSION
      "${LLVM_SUBMIT_VERSION}.${LLVM_SUBMIT_SUBVERSION}")
  endif()

  set(TOOL_INFO_PLIST_OUT "${CMAKE_CURRENT_BINARY_DIR}/${TOOL_INFO_PLIST}")
  target_link_libraries(scc
    "-Wl,-sectcreate,__TEXT,__info_plist,${TOOL_INFO_PLIST_OUT}")
  configure_file("${TOOL_INFO_PLIST}.in" "${TOOL_INFO_PLIST_OUT}" @ONLY)

  set(TOOL_INFO_UTI)
  set(TOOL_INFO_NAME)
  set(TOOL_INFO_VERSION)
  set(TOOL_INFO_BUILD_VERSION)
endif()

if(SCC_ORDER_FILE)
  target_link_libraries(scc "-Wl,-order_file,${SCC_ORDER_FILE}")
endif()

