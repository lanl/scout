project(scc)
include(sc-llvm-link-libs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility-inlines-hidden -fno-rtti")

add_llvm_definitions( -D__STDC_CONSTANT_MACROS )
add_llvm_definitions( -D__STDC_FORMAT_MACROS )
add_llvm_definitions( -D__STDC_LIMIT_MACROS )

include_directories(
  ${SCOUT_INCLUDE_PATH}
  ${SCOUT_LLVM_INCLUDE_PATH}
  ${SCOUT_CLANG_INCLUDE_PATH}
  )

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  Core
  IPA
  IPO
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
  )

set(LLVM_RUNTIME_OUTPUT_INTDIR ${SCOUT_BUILD_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${SCOUT_BUILD_DIR}/lib)

add_clang_executable(scc
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  ScoutVisitor.cpp
  )

target_link_libraries(scc
  clangBasic
  clangDriver
  clangFrontend
  clangFrontendTool
  ${SCOUT_LLVM_LINK_LIBS}
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

set(SCXX_LINK_OR_COPY create_symlink)
set(scc_binary  "${SCOUT_BUILD_DIR}/bin/scc")
set(scpp_binary "${SCOUT_BUILD_DIR}/bin/sc++")

# Create the sc++ symlink in the build directory.

add_custom_command(TARGET scc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E ${SCXX_LINK_OR_COPY} "${scc_binary}" "${scpp_binary}")

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

