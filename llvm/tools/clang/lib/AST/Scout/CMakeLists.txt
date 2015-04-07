include_directories(../)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(scoutAST
  ASTContext.cpp
  ScoutStmt.cpp
  ForallStmt.cpp
  MeshDecl.cpp
  FrameDecl.cpp
  MeshType.cpp
  FrameType.cpp
  MeshLayout.cpp
  RectilinearMeshDecl.cpp
  RenderallStmt.cpp
  RenderTargetType.cpp
  StructuredMeshDecl.cpp
  UniformMeshDecl.cpp
  ALEMeshDecl.cpp
  UnstructuredMeshDecl.cpp
  VolumeRenderAllStmt.cpp
  Type.cpp
  Expr.cpp
  
  LINK_LIBS
  clangBasic
  clangLex
  )
