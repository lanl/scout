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
#include "CGScoutRuntime.h"

using namespace clang;
using namespace clang::CodeGen;

static char IRNameStr[160];
static const char *IndexNames[] = { "x", "y", "z", "w"};
static const char *DimNames[]   = { "width", "height", "depth" };


void CodeGenFunction::EmitMeshFieldsUsedMD(MeshFieldMap HS,
    const char *str, llvm::BranchInst *BI) {
  SmallVector<llvm::Metadata*, 16> MDL;
  llvm::MDString *MDName = llvm::MDString::get(getLLVMContext(), str);
  MDL.push_back(MDName);

  for( MeshFieldMap::const_iterator it = HS.begin(); it != HS.end(); ++it)
  {
    MDName = llvm::MDString::get(getLLVMContext(), it->first);
    MDL.push_back(MDName);

  }
  BI->setMetadata(StringRef(str),
      llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Metadata*>(MDL)));

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
      if(cast<llvm::MDString>((*II)->getOperand(0))->getString() == MeshTypeName) {
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


llvm::Value *CodeGenFunction::LinearIdx2InductionVar(llvm::Value *linearidx,
    MeshElementType elementType, unsigned int dindex, unsigned int ndims) {

  llvm::Value* induct;

  llvm::Value* width = Builder.CreateLoad(scoutPtr(MeshDims[0]), "width");
  if(elementType == Vertices) {
    width = Builder.CreateAdd(width, llvm::ConstantInt::get(Int64Ty, 1));
  }

  sprintf(IRNameStr, "induct.%s", IndexNames[dindex]);

  switch (dindex) {
    case 0:
      induct = Builder.CreateURem(linearidx, width, IRNameStr);
      break;
    case 1:
      if (ndims == 1) {
        induct = llvm::ConstantInt::get(Int64Ty, 0);
        break;
      }

      if(ndims == 3) {
        llvm::Value* height = Builder.CreateLoad(scoutPtr(MeshDims[1]), "height");
        if (elementType == Vertices) {
          height = Builder.CreateAdd(height, llvm::ConstantInt::get(Int64Ty, 1));
        }

        linearidx = Builder.CreateURem(linearidx, Builder.CreateMul(width, height));
      }

      induct = Builder.CreateUDiv(linearidx, width, IRNameStr);

      break;
    case 2: {
      if (ndims < 3) {
        induct = llvm::ConstantInt::get(Int64Ty, 0);
        break;
      }

      llvm::Value* height = Builder.CreateLoad(scoutPtr(MeshDims[1]), "height");
      if (elementType == Vertices) {
        height = Builder.CreateAdd(height, llvm::ConstantInt::get(Int64Ty, 1));
      }

      induct = Builder.CreateUDiv(linearidx, Builder.CreateMul(width, height), IRNameStr);
      break;
    }
    case 3:
      induct = linearidx;
  }
  return induct;

}


// If in Stencil then lookup and load InductionVar, otherwise return it directly
llvm::Value *CodeGenFunction::LookupInductionVar(unsigned int index) {
  auto itr = LocalDeclMap.find(ScoutABIInductionVarDecl[index]);
  
  if(itr != LocalDeclMap.end()) {
    llvm::Value *V = itr->second.getPointer();
    llvm::errs() << "stencil in LookupInductionVar\n";
    if (index == 3) sprintf(IRNameStr, "stencil.linearidx.ptr");
    else sprintf(IRNameStr, "stencil.induct.%s.ptr", IndexNames[index]);
    return Builder.CreateLoad(scoutPtr(V), IRNameStr);
  }
  
  if(ForallStack.empty()){
    return InductionVar[index];
  }
  
  ForallData* data = GetForallData(ForallStackIndex);
  
  if(index == 3) {
    return data->indexPtr;
  }
  
  if(!data->hasInductionVar[index]){
    const MeshType* mt = data->getMeshType();
    auto& dims = mt->dimensions();
    
    sprintf(IRNameStr, "induct.%s.ptr", IndexNames[index]);
    
    llvm::Value* idx = Builder.CreateLoad(scoutPtr(data->indexPtr), "index");
    llvm::Value* induct = LinearIdx2InductionVar(idx, data->elementType, index, dims.size());
    
    Builder.CreateStore(induct, scoutPtr(data->inductionVar[index]));
    data->hasInductionVar[index] = true;
  }
  
  return data->inductionVar[index];
}

llvm::Value *CodeGenFunction::LookupMeshStart(unsigned int index) {
  auto itr = LocalDeclMap.find(ScoutABIInductionVarDecl[index]);
  
  if(itr != LocalDeclMap.end()) {
    llvm::Value *V = itr->second.getPointer();
    sprintf(IRNameStr, "stencil.induct.%s.ptr", IndexNames[index]);
    return Builder.CreateLoad(scoutPtr(V), IRNameStr);
  }
  return MeshStart[index];
}

// If in Stencil then lookup and load Mesh Dimension, otherwise return it directly
llvm::Value *CodeGenFunction::LookupMeshDim(unsigned int index) {
  auto itr = LocalDeclMap.find(ScoutABIInductionVarDecl[index]);
  
  if(itr != LocalDeclMap.end()) {
    llvm::Value *V = itr->second.getPointer();
    sprintf(IRNameStr, "stencil.%s.ptr", DimNames[index]);
    return Builder.CreateLoad(scoutPtr(V), IRNameStr);
  }
  return MeshDims[index];
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

void CodeGenFunction::dumpValue(const char* label, llvm::Value* value){
	CGM.getScoutRuntime().DumpValue(*this, label, value);
}

void CodeGenFunction::dumpUnsignedValue(const char* label, llvm::Value* value){
  CGM.getScoutRuntime().DumpUnsignedValue(*this, label, value);
}
