#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace clang::CodeGen;

/// Emit field annotations for the given mesh field & value. Returns the
/// annotation result.
llvm::Value *CodeGenFunction::EmitFieldAnnotations(const MeshFieldDecl *D,
                                                   llvm::Value *V) {
  assert(D->hasAttr<AnnotateAttr>() && "no annotate attribute");
  llvm::Type *VTy = V->getType();
  llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::ptr_annotation,
                                    CGM.Int8PtrTy);

  for (specific_attr_iterator<AnnotateAttr>
       ai = D->specific_attr_begin<AnnotateAttr>(),
       ae = D->specific_attr_end<AnnotateAttr>(); ai != ae; ++ai) {
    // FIXME Always emit the cast inst so we can differentiate between
    // annotation on the first field of a struct and annotation on the struct
    // itself.
    if (VTy != CGM.Int8PtrTy)
      V = Builder.Insert(new llvm::BitCastInst(V, CGM.Int8PtrTy));
    V = EmitAnnotationCall(F, V, (*ai)->getAnnotation(), D->getLocation());
    V = Builder.CreateBitCast(V, VTy);
  }

  return V;
}

// Special case for mesh types, because we cannot check
// for type pointer equality  because each mesh has its own type
bool CodeGenFunction::CheckMeshPtrTypes(QualType &ArgType, QualType &ActualArgType) {

  const Type* argType =
      getContext().getCanonicalType(ArgType.getNonReferenceType()).getTypePtr();
  const Type* actualType = getContext().getCanonicalType(ActualArgType).getTypePtr();

  //Check that mesh dimensions match
  if(argType->isMeshType() && actualType->isMeshType()) {
    const MeshType* SMTy = dyn_cast<MeshType>(argType);
    const MeshType* DMTy = dyn_cast<MeshType>(actualType);
    if(SMTy->rankOf() != DMTy->rankOf()) return false;

    if ((argType->isUniformMeshType() && actualType->isUniformMeshType())  ||
        (argType->isRectilinearMeshType() && actualType->isRectilinearMeshType()) ||
        (argType->isStructuredMeshType() && actualType->isStructuredMeshType()) ||
        (argType->isUnstructuredMeshType() && actualType->isUnstructuredMeshType())) {
      llvm::errs() << "codegen mesh pointer compare ok\n";
      return true;
    }
  }
  return false;
}


#if 0
bool CodeGen::CodeGenFunction::isMeshMember(llvm::Argument *arg, 
                                            bool& isSigned, 
                                            std::string& typeStr) {
     
  isSigned = false;

  if(arg->getName().endswith("height")) return false;
  if(arg->getName().endswith("width"))  return false;
  if(arg->getName().endswith("depth"))  return false;
  if(arg->getName().endswith("ptr"))    return false;
  if(arg->getName().endswith("dim_x"))  return false;
  if(arg->getName().endswith("dim_y"))  return false;
  if(arg->getName().endswith("dim_z"))  return false;
    
  typedef MemberMap::iterator MemberIterator;
  for(MemberIterator it = MeshMembers.begin(), end = MeshMembers.end(); it != end; ++it) {

    std::string name = it->first;
    std::string argName = arg->getName();

    size_t pos = argName.find(name);
    size_t len = name.length();
    if (pos == 0 && (argName.length() <= len || std::isdigit(argName[len]))) {
      QualType qt = it->second.second;
      isSigned = qt.getTypePtr()->isSignedIntegerType();
      typeStr = qt.getAsString() + "*";
      //llvm::outs() << "mesh: " << name << " " << typeStr << "\n";
      return true;
    }  
  }
  return false;
}
#endif
