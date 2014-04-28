include_directories(../)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(scoutAST
  ASTContext.cpp
  ForallStmt.cpp
  MeshDecl.cpp
  MeshType.cpp
  MeshLayout.cpp
  RectilinearMeshDecl.cpp
  RenderallStmt.cpp
  RenderTargetType.cpp
  StructuredMeshDecl.cpp
  UniformMeshDecl.cpp
  UnstructuredMeshDecl.cpp
  VolumeRenderAllStmt.cpp
  Type.cpp
  
  LINK_LIBS
  clangBasic
  clangLex
  )
