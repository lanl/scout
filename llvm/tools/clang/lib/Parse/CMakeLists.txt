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
  # ===== Scout ===========================================
  scout/ParseDecl.cpp
  scout/ParseDeclCXX.cpp
  scout/ParseExpr.cpp
  scout/ParseStmt.cpp
  scout/Parser.cpp
  scout/ASTViewScout.cpp
  # =======================================================
  )

add_dependencies(clangParse
  ClangAttrClasses
  ClangAttrExprArgs
  ClangAttrLateParsed
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
