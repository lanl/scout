include_directories(../)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(scoutLex
  Lexer.cpp

  LINK_LIBS
  clangBasic
  )
