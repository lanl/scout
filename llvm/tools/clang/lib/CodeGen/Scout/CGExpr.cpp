#include <stdio.h>

#include "CodeGenFunction.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CGRecordLayout.h"
#include "CGMeshLayout.h"
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
#include "Scout/CGMeshLayout.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include <stdio.h>

using namespace clang;
using namespace CodeGen;

// We use 'IRNameStr' to hold the generated names we use for
// various values in the IR building.  We've added a static
// buffer to avoid the need for a lot of fine-grained new and
// delete calls...  We're likely safe with 160 character long
// strings.
static char IRNameStr[160];


/*
LValue
CodeGenFunction::EmitScoutColorDeclRefLValue(const NamedDecl *ND) {
  const ValueDecl *VD = cast<ValueDecl>(ND);
  CharUnits Alignment = getContext().getDeclAlign(ND);
  llvm::Value *idx = getGlobalIdx();
  llvm::Value* ep = Builder.CreateInBoundsGEP(Colors, idx);
  return MakeAddrLValue(ep, VD->getType(), Alignment);
}
*/

/*
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
  assert(false && "missed conditional case in emiting forall array lval.");
}
*/

bool
CodeGenFunction::EmitScoutMemberExpr(const MemberExpr *E, LValue *LV) {
  Expr *BaseExpr = E->getBase();
  NamedDecl *MND = E->getMemberDecl(); //this memberDecl is for the "Implicit" mesh

  if (MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(MND)) {
    DeclRefExpr *D = dyn_cast<DeclRefExpr>(BaseExpr);
    VarDecl *VD = dyn_cast<VarDecl>(D->getDecl());

    if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(VD)) {

      // lookup underlying mesh instead of implicit mesh
      llvm::Value *V = LocalDeclMap.lookup(IMPD->getMeshVarDecl());
      // need underlying mesh to make LValue
      LValue BaseLV  = MakeAddrLValue(V, E->getType());

      *LV = EmitLValueForMeshField(BaseLV, MFD, Builder.CreateLoad(getGlobalIdx(), "forall.linearidx"));
      return true;
    } else {
      llvm_unreachable("Cannot lookup underlying mesh");
    }
  } else {
    return false;
  }
}

LValue
CodeGenFunction::EmitLValueForMeshField(LValue base,
                                     const MeshFieldDecl *field, llvm::Value *Index) {

  // This follows very closely with the details used to
  // emit a record member from the clang code. EmitLValueForField()
  // We have removed details having to do with unions as we know
  // we are struct-like in behavior. A few questions remain
  // here:
  //
  //   SC_TODO - we need to address alignment details better.
  //   SC_TODO - we need to make sure we can ditch code for
  //             TBAA (type-based aliases analysis).
  // fields are before mesh dimensions so we can do things the same a struct
  // this is setup in Codegentypes.h ConvertScoutMeshType()
  const MeshDecl *mesh = field->getParent();

  if (field->isBitField()) {
    const CGMeshLayout &ML = CGM.getTypes().getCGMeshLayout(mesh);
    const CGBitFieldInfo &Info = ML.getBitFieldInfo(field);
    llvm::Value *Addr = base.getAddress();
    unsigned Idx = ML.getLLVMFieldNo(field);
    if (Idx != 0)
      // For structs, we GEP to the field that the record layout suggests.
      sprintf(IRNameStr, "%s.%s.ptr", mesh->getName().str().c_str(),field->getName().str().c_str());
      Addr = Builder.CreateStructGEP(Addr, Idx, IRNameStr);
    // Get the access type.
    llvm::Type *PtrTy = llvm::Type::getIntNPtrTy(
      getLLVMContext(), Info.StorageSize,
      CGM.getContext().getTargetAddressSpace(base.getType()));
    if (Addr->getType() != PtrTy)
      Addr = Builder.CreateBitCast(Addr, PtrTy);

    QualType fieldType =
      field->getType().withCVRQualifiers(base.getVRQualifiers());
    return LValue::MakeBitfield(Addr, Info, fieldType, base.getAlignment());
  }

  QualType type = field->getType();
  CharUnits alignment = getContext().getDeclAlign(field);

 // FIXME: It should be impossible to have an LValue without alignment for a
  // complete type.
  if (!base.getAlignment().isZero())
    alignment = std::min(alignment, base.getAlignment());

  bool mayAlias = mesh->hasAttr<MayAliasAttr>();

  llvm::Value *addr = base.getAddress();
  unsigned cvr = base.getVRQualifiers();
  bool TBAAPath = CGM.getCodeGenOpts().StructPathTBAA;

  // We GEP to the field that the record layout suggests.
  unsigned idx = CGM.getTypes().getCGMeshLayout(mesh).getLLVMFieldNo(field);
  sprintf(IRNameStr, "%s.%s.ptr", mesh->getName().str().c_str(),field->getName().str().c_str());
  addr = Builder.CreateStructGEP(addr, idx, IRNameStr);

  // If this is a reference field, load the reference right now.
  if (const ReferenceType *refType = type->getAs<ReferenceType>()) {
    llvm::LoadInst *load = Builder.CreateLoad(addr, "ref");
    if (cvr & Qualifiers::Volatile) load->setVolatile(true);
    load->setAlignment(alignment.getQuantity());

    // Loading the reference will disable path-aware TBAA.
    TBAAPath = false;
    if (CGM.shouldUseTBAA()) {
      llvm::MDNode *tbaa;
      if (mayAlias)
        tbaa = CGM.getTBAAInfo(getContext().CharTy);
      else
        tbaa = CGM.getTBAAInfo(type);
      CGM.DecorateInstruction(load, tbaa);
    }

    addr = load;
    mayAlias = false;
    type = refType->getPointeeType();
    if (type->isIncompleteType())
      alignment = CharUnits();
    else
      alignment = getContext().getTypeAlignInChars(type);
    cvr = 0; // qualifiers don't recursively apply to referencee
  }

  sprintf(IRNameStr, "%s.%s", mesh->getName().str().c_str(),field->getName().str().c_str());
  addr = Builder.CreateLoad(addr, IRNameStr);

  if (field->hasAttr<AnnotateAttr>())
    addr = EmitFieldAnnotations(field, addr);

  // get the correct element of the field depending on the index
  //llvm::Value *index = Builder.CreateLoad(Index, "forall.linearidx");
  sprintf(IRNameStr, "%s.%s.element", mesh->getName().str().c_str(),field->getName().str().c_str());
  addr = Builder.CreateInBoundsGEP(addr, Index, IRNameStr);

  LValue LV = MakeAddrLValue(addr, type, alignment);
  LV.getQuals().addCVRQualifiers(cvr);
  /*if (TBAAPath) {
    const ASTRecordLayout &Layout =
        getContext().getASTMeshLayout(field->getParent());
    // Set the base type to be the base type of the base LValue and
    // update offset to be relative to the base type.
    LV.setTBAABaseType(mayAlias ? getContext().CharTy : base.getTBAABaseType());
    LV.setTBAAOffset(mayAlias ? 0 : base.getTBAAOffset() +
                     Layout.getFieldOffset(field->getFieldIndex()) /
                                           getContext().getCharWidth());
  }*/

  // __weak attribute on a field is ignored.
  if (LV.getQuals().getObjCGCAttr() == Qualifiers::Weak)
    LV.getQuals().removeObjCGCAttr();

  // Fields of may_alias structs act like 'char' for TBAA purposes.
  // FIXME: this should get propagated down through anonymous structs
  // and unions.
  if (mayAlias && LV.getTBAAInfo())
    LV.setTBAAInfo(CGM.getTBAAInfo(getContext().CharTy));

  return LV;

}

RValue CodeGenFunction::EmitCShiftExpr(ArgIterator ArgBeg, ArgIterator ArgEnd) {
  DEBUG_OUT("EmitCShiftExpr");

  if(const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(*(ArgBeg))) {
    if(const MemberExpr *E = dyn_cast<MemberExpr>(CE->getSubExpr())) {
      Expr *BaseExpr = E->getBase();
      NamedDecl *MND = E->getMemberDecl(); //this memberDecl is for the "Implicit" mesh

     if (MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(MND)) {
       DeclRefExpr *D = dyn_cast<DeclRefExpr>(BaseExpr);
       VarDecl *VD = dyn_cast<VarDecl>(D->getDecl());

        if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(VD)) {

          // lookup underlying mesh instead of implicit mesh
          llvm::Value *V = LocalDeclMap.lookup(IMPD->getMeshVarDecl());
          // need underlying mesh to make LValue
          LValue BaseLV  = MakeAddrLValue(V, E->getType());

          SmallVector< llvm::Value *, 3 > args;
          while(++ArgBeg != ArgEnd) {
            RValue RV = EmitAnyExpr(*(ArgBeg));
            if(RV.isAggregate()) {
              args.push_back(RV.getAggregateAddr());
            } else {
              args.push_back(RV.getScalarVal());
            }
          }
          llvm::errs() << "nargs " << args.size() << "\n";
          for(unsigned i = 0; i < args.size(); i++) {
            llvm::errs() << "arg " << i << " " << *args[i] << "\n";
          }

          SmallVector< llvm::Value *, 3 > dims;
          llvm::errs() << "ndims " << LoopBounds.size() << "\n";
          for(unsigned i = 0; i < LoopBounds.size(); ++i) {
            if (LoopBounds[i]) dims.push_back(Builder.CreateLoad(LoopBounds[i])); //SC_TODO IR naming
            else dims.push_back(llvm::ConstantInt::get(Int32Ty, 1));
            llvm::errs() << "dims " << *dims[i] << "\n";
          }

          // zero out remaining args
          for(unsigned i = args.size(); i < 3; ++i) {
             args.push_back(llvm::ConstantInt::get(Int32Ty, 0));
          }

          // offset = x + Height * (y + Width * z)
          llvm::Value *Wz     = Builder.CreateMul(dims[0], args[2], "WidthxZ");
          llvm::Value *yWz    = Builder.CreateAdd(args[1], Wz, "ypWidthxZ");
          llvm::Value *HyWz   = Builder.CreateMul(dims[2], yWz, "HxypWidthxZ");
          llvm::Value *offset = Builder.CreateAdd(args[0], HyWz, "offset");
          llvm::Value *index  = Builder.CreateAdd(Builder.CreateLoad(getGlobalIdx(),
              "forall.linearidx"), offset, "cshift.linearidx");
          LValue LV = EmitLValueForMeshField(BaseLV, MFD,  index);
          return RValue::get(Builder.CreateLoad(LV.getAddress()));
        }
      }
    }
  }
  assert(false && "Failed to translate Scout cshift expression to LLVM IR!");
}

#if 0
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

//SC_TODO: remove this
LValue CodeGenFunction::EmitMeshMemberExpr(const VarDecl *VD,
                                           llvm::StringRef memberName,
                                           SmallVector< llvm::Value *, 3 > vals) {


  // SC_TODO - 'vals' appears to be predicated loop bounds. ???
  // We should really call it something else so this is clear.
  // If 'vals' is empty we use the global index.

  const MeshType *MT = cast<MeshType>(VD->getType().getCanonicalType());

  // If it is not a mesh member, assume we want the pointer to storage
  // for all mesh members of that name.  In that case, figure out the index
  // to the member and access that.
  if (!isa<ImplicitParamDecl>(VD) )  {
    llvm::errs() << "EmitMeshMemberExpr -- not an implicit parameter '"
                 << memberName << "'.\n";

    MeshDecl* MD = MT->getDecl();
    MeshDecl::field_iterator itr = MD->field_begin();
    MeshDecl::field_iterator itr_end = MD->field_end();

    for(unsigned int i = 4; itr != itr_end; ++itr, ++i) {
      NamedDecl* ND = dyn_cast<NamedDecl>(*itr);
      if (ND && ND->getName() == memberName) {
        // SC_TODO - Does this introduce a bug?  Fix me???  -PM
        if (ND->hasExternalFormalLinkage()) {
          QualType memberTy = dyn_cast< FieldDecl >(*itr)->getType();
          QualType memberPtrTy = getContext().getPointerType(memberTy);
          llvm::Value* baseAddr = LocalDeclMap[VD];
          assert(baseAddr && "vardecl not found in map");
          llvm::Value *memberAddr = Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, i);
          return MakeAddrLValue(memberAddr, memberPtrTy);
        } else {
          // SC_TODO - is this an error?
        }
       }
    }
    // if got here, there was no member of that name, so issue an error
  }

//  RecordDecl *RD = VD->getParent();
//  if (RD)


  llvm::errs() << "EmitMeshMemberExpr -- implicit parameter (or some odd fall-through)\n";
  llvm::errs() << "\tmember name: " << memberName << "\n";
  // Now we deal with the case of an individual mesh member value
  MeshType::MeshDimensions exprDims = MT->dimensions();
  llvm::Value *arg = getGlobalIdx(); // SC_TODO -- we never use this????

  if (!vals.empty()) {
    llvm::errs() << "vals is non-empty.\n";
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

  // getFieldIndex can be used on a mesh decl to lookup the
  // field's index in the mesh.

  llvm::Value *var = MeshMembers[memberName].first;
  assert(var && "unable to find mesh member in map.");
  QualType Ty = MeshMembers[memberName].second;

  char *IRNameStr = new char[memberName.size() + 16];
  sprintf(IRNameStr, "%s.idx.", memberName.str().c_str());
  llvm::Value *addr = Builder.CreateInBoundsGEP(var, arg, IRNameStr);
  delete []IRNameStr; // SC_TODO: we're assuming this is safe after
                      // creating the GEP instruction...
  return MakeAddrLValue(addr, Ty);
}
#endif

