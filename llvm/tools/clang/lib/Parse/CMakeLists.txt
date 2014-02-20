set(LLVM_LINK_COMPONENTS
  MCParser
  MC
  Support
  )

add_clang_library(clangParse
  ParseAST.cpp
  ParseCXXInlineMethods.cpp
  ParseDecl.cpp
  ParseDeclCXX.cpp
  ParseExpr.cpp
  ParseExprCXX.cpp
  ParseInit.cpp
  ParseObjc.cpp
  ParseOpenMP.cpp
  ParsePragma.cpp
  ParseStmt.cpp
  ParseTemplate.cpp
  ParseTentative.cpp
  Parser.cpp
  # +===== Scout ============================================================+
  Scout/ParseDecl.cpp
  Scout/ParseDeclCXX.cpp
  Scout/ParseExpr.cpp
  Scout/ParseStmt.cpp
  Scout/Parser.cpp
  # +========================================================================+
  )

add_dependencies(clangParse
  ClangAttrClasses
  ClangAttrParserStringSwitches
  ClangAttrList
  ClangAttrParsedAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangDiagnosticParse
  ClangStmtNodes
  )

target_link_libraries(clangParse
  clangBasic
  clangAST
  clangLex
  clangSema
  )
