#include "CGDebugInfo.h"
#include "CGBlocks.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace clang::CodeGen;

llvm::DIType CGDebugInfo::CreateType(const MeshType *Ty) {

  if (MeshDecl* MD = Ty->getDecl()) {
    if (MD->isUniformMesh()) {
      UniformMeshType *MT;
      MT = cast<UniformMeshType>(CGM.getContext.getUniformMeshType(MD).getTypePtr());
      return CreateType(MT);
    } else if (MD->isStructuredMesh()) {
      StructuredMeshType *MT;
      MT = cast<StructuredMeshType>(CGM.getContext.getStructuredMeshType(MD).getTypePtr());
      return CreateType(MT);    
    } else if (MD->isRectilinearMesh()) {
      RectilinearMeshType *MT;
      MT = cast<RectilinearMeshType>(CGM.getContext.getRectilinearMeshType(MD).getTypePtr());
      return CreateType(MT);
    } else if (MD->isUnstructuredMesh()) {
      UniformMeshType *MT;
      MT = cast<UnstructuredMeshType>(CGM.getContext.getUnstructuredMeshType(MD).getTypePtr());
      return CreateType(MT);
    } else {
      assert(false && "unknown mesh type!");
      return CreateType(0);
    }
  } else {
    assert(false && "unable to get get mesh decl from type!");
    return CreateType(0);
  }
}
