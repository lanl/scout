include_directories(../)

set(LLVM_LINK_COMPONENTS
  Option
  Support
  TransformUtils
  )

add_clang_library(scoutDriver
  Tools.cpp

  DEPENDS
  ClangCC1AsOptions
  ClangDriverOptions
  ClangAttrList
  ClangDiagnosticCommon
  ClangDiagnosticDriver

  LINK_LIBS
  clangBasic
  )
