include_directories(../)

set(LLVM_LINK_COMPONENTS
  MCParser
  MC
  Support
  )

add_clang_library(scoutParse
  ParseDecl.cpp
  ParseExpr.cpp
  ParseStmt.cpp
  Parser.cpp
  
  LINK_LIBS
  clangBasic
  clangAST
  clangLex
  clangSema
  )
