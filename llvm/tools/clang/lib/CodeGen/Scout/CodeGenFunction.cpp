#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Intrinsics.h"
#include "Scout/ASTVisitors.h"

using namespace clang;
using namespace clang::CodeGen;

static char IRNameStr[160];
static const char *IndexNames[] = { "x", "y", "z", "w"};
static const char *DimNames[]   = { "width", "height", "depth" };


void CodeGenFunction::EmitMeshFieldsUsedMD(MeshFieldMap HS,
    const char *str, llvm::BranchInst *BI) {
  SmallVector<llvm::Value*, 16> MDL;
  llvm::MDString *MDName = llvm::MDString::get(getLLVMContext(), str);
  MDL.push_back(MDName);

  for( MeshFieldMap::const_iterator it = HS.begin(); it != HS.end(); ++it)
  {
    MDName = llvm::MDString::get(getLLVMContext(), it->first);
    MDL.push_back(MDName);

  }
  BI->setMetadata(StringRef(str),
      llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(MDL)));

}


void CodeGenFunction::EmitStencilMDBlock(const FunctionDecl *FD) {
  llvm::BasicBlock *entry = createBasicBlock("stencil.md");
  llvm::BranchInst *BI = Builder.CreateBr(entry);
  //SC_TODO: add stencil metadata
  (void)BI;
  EmitBlock(entry);
}

void CodeGenFunction::EmitTaskMDBlock(const FunctionDecl *FD) {
  TaskDeclVisitor v(FD);
  v.VisitStmt(FD->getBody());

  llvm::BasicBlock *entry = createBasicBlock("task.md");
  llvm::BranchInst *BI = Builder.CreateBr(entry);

  llvm::NamedMDNode *MeshMD = CGM.getModule().getNamedMetadata("scout.meshmd");
  MeshNameMap MNM = v.getMeshNamemap();

  for(MeshNameMap::const_iterator it =
      MNM.begin(); it != MNM.end(); ++it) {
    // find meta data for mesh used in this forall
    const std::string MeshName = it->first;
    const std::string MeshTypeName =  it->second;
    for (llvm::NamedMDNode::op_iterator II = MeshMD->op_begin(), IE = MeshMD->op_end();
        II != IE; ++II) {
      if((*II)->getOperand(0)->getName().str() == MeshTypeName) {
        BI->setMetadata(MeshName, *II);
      }
    }
  }

  //find fields used on LHS and add to metadata
  MeshFieldMap LHS = v.getLHSmap();
  EmitMeshFieldsUsedMD(LHS, "LHS", BI);

  //find fields used on RHS and add to metadata
  MeshFieldMap RHS = v.getRHSmap();
  EmitMeshFieldsUsedMD(RHS, "RHS", BI);

  EmitBlock(entry);

}

// If in Stencil then lookup and load InductionVar, otherwize return it directly
llvm::Value *CodeGenFunction::LookupInductionVar(unsigned int index) {
  llvm::Value *V = LocalDeclMap.lookup(ScoutABIInductionVarDecl[index]);
  if(V) {
    if (index == 3) sprintf(IRNameStr, "stencil.linearidx.ptr");
    else sprintf(IRNameStr, "stencil.induct.%s.ptr", IndexNames[index]);
    return Builder.CreateLoad(V, IRNameStr);
  }
  return InductionVar[index];
}

// If in Stencil then lookup and load LoopBound, otherwise return it directly
llvm::Value *CodeGenFunction::LookupLoopBound(unsigned int index) {
  llvm::Value *V = LocalDeclMap.lookup(ScoutABILoopBoundDecl[index]);
  if(V) {
    sprintf(IRNameStr, "stencil.%s.ptr", DimNames[index]);
    return Builder.CreateLoad(V, IRNameStr);
  }
  return LoopBounds[index];
}

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
  const Type* actualType =
      getContext().getCanonicalType(ActualArgType).getTypePtr();

  if(CGM.getContext().CompareMeshTypes(argType, actualType)) return true;
  return false;
}

