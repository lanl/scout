# +===== Scout ============================================================+
add_subdirectory(Scout)
# +========================================================================+

set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangDriver
  Action.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  Multilib.cpp
  MSVCToolChain.cpp
  Phases.cpp
  SanitizerArgs.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  Tools.cpp
  Types.cpp

  DEPENDS
  ClangDriverOptions
  # +===== Scout ============================================================+
  scoutDriver
  ClangAttrList
  ClangDiagnosticCommon
  ClangDiagnosticDriver
  # +========================================================================+

  LINK_LIBS
  # +===== Scout ============================================================+
  scoutDriver
  # +========================================================================+
  clangBasic
  )
