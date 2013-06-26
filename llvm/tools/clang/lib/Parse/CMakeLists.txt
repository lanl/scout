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
  Scout/ParseDecl.cpp
  Scout/ParseDeclCXX.cpp
  Scout/ParseExpr.cpp
  Scout/ParseStmt.cpp
  Scout/Parser.cpp
  Scout/ASTViewScout.cpp
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
