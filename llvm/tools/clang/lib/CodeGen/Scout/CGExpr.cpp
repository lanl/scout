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
#include "clang/AST/scout/ImplicitMeshParamDecl.h"

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
  assert(false && "missed conditional case in emiting forall array lval.");
}

static llvm::Value *
EmitBitCastOfLValueToProperType(CodeGenFunction &CGF,
                                llvm::Value *V, llvm::Type *IRType,
                                StringRef Name = StringRef()) {
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  return CGF.Builder.CreateBitCast(V, IRType->getPointerTo(AS), Name);
}


bool
CodeGenFunction::EmitScoutMemberExpr(const MemberExpr *E, LValue *LV) {
  unsigned rank = 0;
  Expr *BaseExpr = E->getBase();
  NamedDecl *MND = E->getMemberDecl(); //this memberDecl is for the Implicit mesh, maybe needs to be for underlying mesh?

  if (MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(MND)) {
    DeclRefExpr *D = dyn_cast<DeclRefExpr>(BaseExpr);
    VarDecl *VD = dyn_cast<VarDecl>(D->getDecl());

    const Type* T = VD->getType().getCanonicalType().getTypePtr();
    if(const MeshType *MT = dyn_cast<MeshType>(T)){
      rank = MT->dimensions().size();
      llvm::errs() << "mesh rank " << rank << "\n";
    } else {
      llvm_unreachable("Cannot determine mesh rank");
    }

    llvm::errs() << "mesh name " << VD->getName() << "\n";
    llvm::errs() << "member name " << MND->getName() << "\n";

    if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(VD)) {
      llvm::errs() << "underlying mesh is " << IMPD->getMeshVarDecl()->getName() << "\n";

      // lookup underlying mesh instead of implicit mesh
      llvm::Value *V = LocalDeclMap.lookup(IMPD->getMeshVarDecl());
      LValue BaseLV  = MakeAddrLValue(V, E->getType());

      *LV = EmitMeshMemberExpr(BaseLV, MFD, rank);
      return true;
    } else {
      llvm_unreachable("Cannot lookup underlying mesh");

    }
  } else {
    return false;
  }
}

LValue
CodeGenFunction::EmitMeshMemberExpr(LValue base,
                                     const MeshFieldDecl *field, unsigned rank) {

  // This follows very closely with the details used to 
  // emit a record member from the clang code.  We have 
  // removed details having to do with unions as we know
  // we are struct-like in behavior. A few questions remain
  // here:
  // 
  //   SC_TODO - we need to address alignment details better.
  //   SC_TODO - we need to make sure we can ditch code for 
  //             TBAA (type-based aliases analysis). 
  //
  if (field->isBitField()) {
    const CGMeshLayout &ML = CGM.getTypes().getCGMeshLayout(field->getParentMesh());
    const CGBitFieldInfo &Info = ML.getBitFieldInfo(field);
    llvm::Value *Addr = base.getAddress();
    unsigned Idx = ML.getLLVMFieldNo(field) + rank + 1; //SC_TODO: is +rank+1 correct here?
    if (Idx != 0)
      // For structs, we GEP to the field that the record layout suggests.
      Addr = Builder.CreateStructGEP(Addr, Idx, field->getName());
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

  const MeshDecl *mesh = field->getParentMesh();
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
  unsigned idx = CGM.getTypes().getCGMeshLayout(mesh).getLLVMFieldNo(field) + rank + 1;
  addr = Builder.CreateStructGEP(addr, idx, field->getName());   //GEP of the field

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

  addr = Builder.CreateLoad(addr); 

  if (field->hasAttr<AnnotateAttr>())
    addr = EmitFieldAnnotations(field, addr);

  // get the field element for this index
  llvm::Value *index = Builder.CreateAlignedLoad(getGlobalIdx(), 4, "idx");
  addr = Builder.CreateInBoundsGEP(addr, index, "meshidx"); 

  LValue LV = MakeAddrLValue(addr, type, alignment);
  LV.getQuals().addCVRQualifiers(cvr);
  if (TBAAPath) {
    const ASTRecordLayout &Layout =
        getContext().getASTRecordLayout(field->getParent());
    // Set the base type to be the base type of the base LValue and
    // update offset to be relative to the base type.
    LV.setTBAABaseType(mayAlias ? getContext().CharTy : base.getTBAABaseType());
    LV.setTBAAOffset(mayAlias ? 0 : base.getTBAAOffset() +
                     Layout.getFieldOffset(field->getFieldIndex()) /
                                           getContext().getCharWidth());
  }

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

