set(MODULE TRUE)

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(PrintMeshDecls PrintMeshDecls.cpp)

add_dependencies(PrintMeshDecls
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(PrintMeshDecls
  clangFrontend
  clangAST
  )

set_target_properties(PrintMeshDecls
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")

