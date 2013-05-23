
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
