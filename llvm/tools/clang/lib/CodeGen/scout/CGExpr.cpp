
#include "CodeGenFunction.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CGRecordLayout.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/ConvertUTF.h"

using namespace clang;
using namespace CodeGen;

// SC_TODO - we need to replace Scout's vector types with Clang's
// "builtin" type.  This has been done in the "refactor" branch
// but it still needs to be merged with "devel". 
LValue
CodeGenFunction::EmitScoutVectorMemberExpr(const ScoutVectorMemberExpr *E) {

  if (isa<MemberExpr>(E->getBase())) {
    ValueDecl *VD = cast<MemberExpr>(E->getBase())->getMemberDecl();
    if (VD->getName() == "position") {
      return MakeAddrLValue(ScoutIdxVars[E->getIdx()], getContext().IntTy);
    }
    assert(false && "Attempt to translate Scout 'position' to LLVM IR failed");
  } else {
    LValue LHS = EmitLValue(E->getBase());
    llvm::Value *Idx = llvm::ConstantInt::get(Int32Ty, E->getIdx());
    return LValue::MakeVectorElt(LHS.getAddress(), Idx,
                                 E->getBase()->getType(),
                                 LHS.getAlignment());
  }
}


LValue
CodeGenFunction::EmitScoutColorDeclRefLValue(const NamedDecl *ND) {
  const ValueDecl *VD = cast<ValueDecl>(ND);
  CharUnits Alignment = getContext().getDeclAlign(ND);
  llvm::Value *idx = getGlobalIdx();
  llvm::Value* ep = Builder.CreateInBoundsGEP(Colors, idx);
  return MakeAddrLValue(ep, VD->getType(), Alignment);
}

LValue
CodeGenFunction::EmitScoutForAllArrayDeclRefLValue(const NamedDecl *ND) {
  CharUnits Alignment = getContext().getDeclAlign(ND);  
  for(unsigned i = 0; i < 3; ++i) {
    const IdentifierInfo* ii = CurrentForAllArrayStmt->getInductionVar(i);
    if (!ii) 
      break;
    
    if (ii->getName().equals(ND->getName())) {
      const ValueDecl *VD = cast<ValueDecl>(ND);
      return MakeAddrLValue(ScoutIdxVars[i], VD->getType(), Alignment);
    }

  }
  // SC_TODO -- what happens if we fall through here?  For now we'll
  // bail with an assertion.  Overall, the logic seems a bit screwy
  // to me here...
  assert(false && "unhandled conditional in emiting forall array lval.");
}

LValue
CodeGenFunction::EmitScoutMemberExpr(const MemberExpr *E,
                                     const VarDecl *VD) {

  llvm::Value* baseAddr;

  if (VD->hasGlobalStorage()) {
    baseAddr = Builder.CreateLoad(CGM.GetAddrOfGlobalVar(VD));
  } else {
    baseAddr = LocalDeclMap[VD];
    if (VD->getType().getTypePtr()->isReferenceType()) {
      baseAddr = Builder.CreateLoad(baseAddr);
    }
  }

  llvm::StringRef memberName = E->getMemberDecl()->getName();

  if (memberName == "width")
    return MakeAddrLValue(Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, 1),
                          getContext().IntTy);
  else if (memberName == "height")
    return MakeAddrLValue(Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, 2),
                          getContext().IntTy);
  else if (memberName == "depth")
    return MakeAddrLValue(Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, 3),
                          getContext().IntTy);
  else if (memberName == "ptr") {
    llvm::Value* mp = Builder.CreateBitCast(baseAddr, VoidPtrTy, "mesh.ptr");
    llvm::Value* tempAddr = CreateMemTemp(getContext().VoidPtrTy, "ref.temp");
    Builder.CreateStore(mp, tempAddr);
    return MakeAddrLValue(tempAddr, getContext().VoidPtrTy);
  }

  return EmitMeshMemberExpr(VD, memberName);
}

RValue CodeGenFunction::EmitCShiftExpr(ArgIterator ArgBeg, ArgIterator ArgEnd) {
  DEBUG_OUT("EmitCShiftExpr");

  if(const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(*(ArgBeg))) {
    if(const MemberExpr *ME = dyn_cast<MemberExpr>(CE->getSubExpr())) {
      Expr *BaseExpr = ME->getBase();
      const NamedDecl *ND = cast< DeclRefExpr >(BaseExpr)->getDecl();
      const VarDecl *VD = dyn_cast<VarDecl>(ND);
      llvm::StringRef memberName = ME->getMemberDecl()->getName();

      SmallVector< llvm::Value *, 3 > vals;
      while(++ArgBeg != ArgEnd) {
        RValue RV = EmitAnyExpr(*(ArgBeg));
        if(RV.isAggregate()) {
          vals.push_back(RV.getAggregateAddr());
        } else {
          vals.push_back(RV.getScalarVal());
        }
      }

      LValue LV = EmitMeshMemberExpr(VD, memberName, vals);
      return RValue::get(Builder.CreateLoad(LV.getAddress()));
    }
  }
  assert(false && "Failed to translate Scout cshift expression to LLVM IR!");
}



LValue 
CodeGenFunction::EmitMeshMemberExpr(const VarDecl *VD, 
                                    llvm::StringRef memberName,
                                    SmallVector< llvm::Value *, 3 > vals) {
  DEBUG_OUT("EmitMeshMemberExpr");

  const MeshType *MT = cast<MeshType>(VD->getType().getCanonicalType());

  // If it is not a mesh member, assume we want the pointer to storage
  // for all mesh members of that name.  In that case, figure out the index 
  // to the member and access that.
  if (!isa<ImplicitParamDecl>(VD) )  {

    MeshDecl* MD = MT->getDecl();
    MeshDecl::mesh_field_iterator itr = MD->mesh_field_begin();
    MeshDecl::mesh_field_iterator itr_end = MD->mesh_field_end();

    for(unsigned int i = 4; itr != itr_end; ++itr, ++i) {
      if(dyn_cast<NamedDecl>(*itr)->getName() == memberName) {
        if ((*itr)->isExternAlloc()) {
          QualType memberTy = dyn_cast< FieldDecl >(*itr)->getType();
          QualType memberPtrTy = getContext().getPointerType(memberTy);
          llvm::Value* baseAddr = LocalDeclMap[VD];
          llvm::Value *memberAddr = Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, i);
          return MakeAddrLValue(memberAddr, memberPtrTy);
        } else {
          // SC_TODO - is this an error?
        }
       }
    }
    // if got here, there was no member of that name, so issue an error
  }  

  // Now we deal with the case of an individual mesh member value
  MeshType::MeshDimensionVec exprDims = MT->dimensions();
  llvm::Value *arg = getGlobalIdx();

  if (!vals.empty()) {
    SmallVector< llvm::Value *, 3 > dims;
    for(unsigned i = 0, e = exprDims.size(); i < e; ++i) {
      dims.push_back(Builder.CreateLoad(ScoutMeshSizes[i]));
    }

    for(unsigned i = dims.size(); i < 3; ++i) {
      dims.push_back(llvm::ConstantInt::get(Int32Ty, 1));
    }

    for(unsigned i = vals.size(); i < 3; ++i) {
      vals.push_back(llvm::ConstantInt::get(Int32Ty, 0));
    }

    llvm::Value *idx   = getGlobalIdx();
    llvm::Value *add   = Builder.CreateAdd(idx, vals[0]);
    llvm::Value *rem   = Builder.CreateURem(add, dims[0]);
    llvm::Value *div   = Builder.CreateUDiv(idx, dims[0]);
    llvm::Value *rem1  = Builder.CreateURem(div, dims[1]);
    llvm::Value *add2  = Builder.CreateAdd(rem1, vals[1]);
    llvm::Value *rem3  = Builder.CreateURem(add2, dims[1]);
    llvm::Value *mul   = Builder.CreateMul(dims[0], dims[1]);
    llvm::Value *div4  = Builder.CreateUDiv(idx, mul);
    llvm::Value *rem5  = Builder.CreateURem(div4, dims[2]);
    llvm::Value *add7  = Builder.CreateAdd(vals[2], rem5);
    llvm::Value *rem8  = Builder.CreateURem(add7, dims[2]);
    llvm::Value *mul12 = Builder.CreateMul(rem8, dims[1]);
    llvm::Value *tmp   = Builder.CreateAdd(mul12, rem3);
    llvm::Value *tmp1  = Builder.CreateMul(tmp, dims[0]);
    arg = Builder.CreateAdd(tmp1, rem);
  }
  
  llvm::Value *var = MeshMembers[memberName].first;
  QualType Ty = MeshMembers[memberName].second;
  llvm::Value *addr = Builder.CreateInBoundsGEP(var, arg, "arrayidx");
  return MakeAddrLValue(addr, Ty);
}

