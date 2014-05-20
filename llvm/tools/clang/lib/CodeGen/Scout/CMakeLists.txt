# ===== Scout =================================================================
include_directories(
 ${CMAKE_BINARY_DIR}/config/include
 ../
)
# =============================================================================

set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  IPO
  IRReader
  InstCombine
  Instrumentation
  Linker
  MC
  ObjCARCOpts
  ProfileData
  ScalarOpts
  Support
  Target
  TransformUtils
  )

add_clang_library(scoutCodeGen
  CGDebugInfo.cpp
  CGDecl.cpp
  CGExpr.cpp
  CGStmt.cpp
  CodeGenFunction.cpp
  CodeGenTypes.cpp
  CGMeshLayoutBuilder.cpp
  CGScoutRuntime.cpp
  CGBuiltin.cpp
  CodeGenTypes.cpp
  CodeGenModule.cpp
  
  DEPENDS
  intrinsics_gen

  LINK_LIBS
  clangBasic
  clangAST
  clangFrontend
)