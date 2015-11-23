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
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/ConvertUTF.h"
#include "Scout/CGMeshLayout.h"
#include "Scout/CGScoutRuntime.h"
#include "Scout/CGPlotRuntime.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include <stdio.h>

using namespace clang;
using namespace CodeGen;

static const char *DimNames[]   = { "width", "height", "depth" };
static const char *IndexNames[] = { "x", "y", "z"};

// We use 'IRNameStr' to hold the generated names we use for
// various values in the IR building.  We've added a static
// buffer to avoid the need for a lot of fine-grained new and
// delete calls...  We're likely safe with 160 character long
// strings.
static char IRNameStr[160];

#if 0 // not currently used
namespace{

// note: these functions were copied from Clang CGExpr.cpp

llvm::Value *
EmitBitCastOfLValueToProperType(CodeGenFunction &CGF,
                                llvm::Value *V, llvm::Type *IRType,
                                StringRef Name = StringRef()) {
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  return CGF.Builder.CreateBitCast(V, IRType->getPointerTo(AS), Name);
}

LValue EmitGlobalVarDeclLValue(CodeGenFunction &CGF,
    const Expr *E, const VarDecl *VD) {
  llvm::Value *V = CGF.CGM.GetAddrOfGlobalVar(VD);
  llvm::Type *RealVarTy = CGF.getTypes().ConvertTypeForMem(VD->getType());
  V = EmitBitCastOfLValueToProperType(CGF, V, RealVarTy);
  CharUnits Alignment = CGF.getContext().getDeclAlign(VD);
  QualType T = E->getType();
  LValue LV;
  if (VD->getType()->isReferenceType()) {
    llvm::LoadInst *LI = CGF.Builder.CreateLoad(V);
    LI->setAlignment(Alignment.getQuantity());
    V = LI;
    LV = CGF.MakeNaturalAlignAddrLValue(V, T);
  } else {
    LV = CGF.MakeAddrLValue(V, E->getType(), Alignment);
  }
  return LV;
}

} // end namespace
#endif

LValue
CodeGenFunction::EmitColorDeclRefLValue(const NamedDecl *ND) {
  CharUnits Alignment = getContext().getDeclAlign(ND);
  const ValueDecl *VD = cast<ValueDecl>(ND);
  
  if(CurrentVolumeRenderallMeshPtr.isValid()){
    CurrentVolumeRenderallColor =
    Builder.CreateAlloca(llvm::VectorType::get(FloatTy, 4));
    
    return MakeAddrLValue(CurrentVolumeRenderallColor, VD->getType(), Alignment);
  }
  
  llvm::Value *idx = getLinearIdx();
  llvm::Value* ep = Builder.CreateInBoundsGEP(Color, idx);
  return MakeAddrLValue(ep, VD->getType(), Alignment);
}

LValue CodeGenFunction::EmitFrameVarDeclRefLValue(const VarDecl* VD){
  using namespace std;
  using namespace llvm;
  
  llvm::Function* func = Builder.GetInsertBlock()->getParent();
  
  auto aitr = func->arg_begin();
  
  Value* plotPtr = &*aitr;
  aitr++;
  Value* index = &*aitr;
  aitr++;
  
  typedef vector<Value*> ValueVec;
  
  auto R = CGM.getPlotRuntime();
  
  assert(CurrentPlotStmt);
  
  uint32_t varId = CurrentPlotStmt->getVarId(VD);
  
  if(varId == 0){
    varId = CurrentPlotStmt->getExtVarId(VD);
  }
  
  if(varId == 0){
    varId = CurrentPlotStmt->getFrameDecl()->getVarId(VD);
  }
  
  assert(varId != 0);
  
  ValueVec args = {plotPtr, ConstantInt::get(R.Int32Ty, varId), index};
  
  llvm::Type* rt = ConvertType(VD->getType());
  
  Value* ret;
  
  if(rt->isIntegerTy(32)){
    ret = Builder.CreateCall(R.PlotGetI32Func(), args);
  }
  else if(rt->isIntegerTy(64)){
    ret = Builder.CreateCall(R.PlotGetI64Func(), args);
  }
  else if(rt->isFloatTy()){
    ret = Builder.CreateCall(R.PlotGetFloatFunc(), args);
  }
  else if(rt->isDoubleTy()){
    ret = Builder.CreateCall(R.PlotGetDoubleFunc(), args);
  }
  else{
    assert(false && "invalid frame var type");
  }
  
  CharUnits Alignment = getContext().getDeclAlign(VD);
  
  Address addr = scoutPtr(Builder.CreateAlloca(ret->getType()));
  Builder.CreateStore(ret, addr);
  
  return MakeAddrLValue(addr.getPointer(), VD->getType(), Alignment);
}

Address CodeGenFunction::GetForallIndex(const MemberExpr* E){
  DeclRefExpr* base;
  if(ImplicitCastExpr* ce = dyn_cast<ImplicitCastExpr>(E->getBase())){
    base = cast<DeclRefExpr>(ce->getSubExpr());
  }
  else{
    base = cast<DeclRefExpr>(E->getBase());
  }

  if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(base->getDecl())) {
  
  #if 0
    const ValueDecl* mvd;

    if(mp){
      mvd = mp->getMeshVarDecl();
    }
    else{
      mvd = base->getDecl();
    }
  #endif

    int i = FindForallData(IMPD->getElementType());
    assert(i >= 0 && "error finding forall data");

    ForallData* data = ForallStack[i];
    return data->indexPtr;
  } else if (isa<ParmVarDecl>(base->getDecl())) {
    llvm::errs() << "ParmVarDecl in GetForallIndex must be stencil\n";
    
    if (ForallStack.empty()) {
      return LookupInductionVar(3);
    } else {
      assert(false && "stencil in nested forall not supported");
    }
  }
  assert(false && "failed to getforallIndex");
}

LValue
CodeGenFunction::EmitMeshMemberExpr(const MemberExpr* E,
                                    llvm::Value* IndexPtr){
  DeclRefExpr* base;
  if(ImplicitCastExpr* ce = dyn_cast<ImplicitCastExpr>(E->getBase())){
    base = cast<DeclRefExpr>(ce->getSubExpr());
  }
  else{
    base = cast<DeclRefExpr>(E->getBase());
  }
  
  Address Addr = Address::invalid();
  
  // inside forall we are referencing the implicit mesh e.g. 'c' in forall cells c in mesh
   if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(base->getDecl())) {
     // lookup underlying mesh instead of implicit mesh
     Addr = GetMeshBaseAddr(IMPD->getMeshVarDecl());
   } else if (ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(base->getDecl())) {
     llvm::errs() << "ParmVarDecl in EmitMeshMemberExpr must be stencil\n";

     Addr = GetAddrOfLocalVar(PVD);
     const Type *T = PVD->getType().getTypePtr();
     if(T->isAnyPointerType() || T->isReferenceType()) {
      Addr = scoutPtr(Builder.CreateLoad(Addr));
     }
   } else {
     llvm_unreachable("Cannot lookup underlying mesh");
   }
  
  LValue BaseLV  = MakeAddrLValue(Addr, E->getType());
  // assume we have already checked that we are working w/ a mesh and cast to MeshField Decl
  MeshFieldDecl* MFD = cast<MeshFieldDecl>(E->getMemberDecl());
  return EmitLValueForMeshField(BaseLV,  cast<MeshFieldDecl>(MFD), IndexPtr);
}

LValue
CodeGenFunction::EmitVolumeRenderMeshMemberExpr(const MemberExpr *E) {  
  MeshFieldDecl* fd = dyn_cast<MeshFieldDecl>(E->getMemberDecl());
  assert(fd);
  
  auto itr = CurrentVolumeRenderallFieldMap.find(fd);
  assert(itr != CurrentVolumeRenderallFieldMap.end());
  
  std::string fieldName = fd->getName().str();
  fieldName += ".ptr";
  
  Address fieldPtr =
  Builder.CreateStructGEP(CurrentVolumeRenderallMeshPtr,
                          itr->second, getPointerAlign(), fieldName);
  
  llvm::Value *fieldV = Builder.CreateLoad(fieldPtr);
  
  llvm::Value* addr =
  Builder.CreateGEP(nullptr, fieldV,
                    CurrentVolumeRenderallIndex, fd->getName().str());
  
  return MakeAddrLValue(scoutPtr(addr), E->getType());
}

LValue CodeGenFunction::EmitLValueForMeshField(LValue base,
                                               const MeshFieldDecl *field,
                                               llvm::Value *Index) {
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
  StringRef MeshName = base.getAddress().getPointer()->getName();
  StringRef FieldName = field->getName();

  if(isGPU()){
    const CGMeshLayout &ML = CGM.getTypes().getCGMeshLayout(field->getParent());
    unsigned Idx = ML.getLLVMFieldNo(field);

    llvm::StructType* s = ML.getLLVMType();
    llvm::Type* et = s->getElementType(Idx);
    sprintf(IRNameStr, "TheMesh.%s.ptr",
            field->getName().str().c_str());
    llvm::Value* addr = Builder.CreateAlloca(et, 0, IRNameStr);

    sprintf(IRNameStr, "TheMesh.%s.element.ptr",
            field->getName().str().c_str());
    addr = Builder.CreateInBoundsGEP(addr, Index, IRNameStr);
    LValue LV = MakeAddrLValue(addr, field->getType(), getPointerAlign());
    return LV;
  }

  if (field->isBitField()) {
    const CGMeshLayout &ML = CGM.getTypes().getCGMeshLayout(mesh);
    const CGBitFieldInfo &Info = ML.getBitFieldInfo(field);
     Address Addr = base.getAddress();
    unsigned Idx = ML.getLLVMFieldNo(field);
    if (Idx != 0)
      // For structs, we GEP to the field that the record layout suggests.
      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(),FieldName.str().c_str());
      Addr = Builder.CreateStructGEP(Addr, Idx, getPointerAlign(), IRNameStr);
    // Get the access type.
    llvm::Type *PtrTy = llvm::Type::getIntNPtrTy(
      getLLVMContext(), Info.StorageSize,
      CGM.getContext().getTargetAddressSpace(base.getType()));
    if (Addr.getPointer()->getType() != PtrTy)
      Addr = Builder.CreateBitCast(Addr, PtrTy);

    QualType fieldType =
      field->getType().withCVRQualifiers(base.getVRQualifiers());
    return LValue::MakeBitfield(Addr, Info, fieldType, base.getAlignmentSource());
  }

  QualType type = field->getType();
  CharUnits alignment = getContext().getDeclAlign(field);

  // FIXME: It should be impossible to have an LValue without alignment for a
  // complete type.
  if (!base.getAlignment().isZero())
    alignment = std::min(alignment, base.getAlignment());

  bool mayAlias = mesh->hasAttr<MayAliasAttr>();

  Address addr = base.getAddress();
  unsigned cvr = base.getVRQualifiers();

  // We GEP to the field that the record layout suggests.
  unsigned idx = CGM.getTypes().getCGMeshLayout(mesh).getLLVMFieldNo(field);

  sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(),FieldName.str().c_str());
  addr = Builder.CreateStructGEP(addr, idx, getPointerAlign(), IRNameStr);

  // If this is a reference field, load the reference right now.
  if (const ReferenceType *refType = type->getAs<ReferenceType>()) {
    llvm::LoadInst *load = Builder.CreateLoad(addr, "ref");
    if (cvr & Qualifiers::Volatile) load->setVolatile(true);
    load->setAlignment(alignment.getQuantity());

    // Loading the reference will disable path-aware TBAA.
    if (CGM.shouldUseTBAA()) {
      llvm::MDNode *tbaa;
      if (mayAlias)
        tbaa = CGM.getTBAAInfo(getContext().CharTy);
      else
        tbaa = CGM.getTBAAInfo(type);
      CGM.DecorateInstructionWithTBAA(load, tbaa);
    }

    addr = scoutPtr(load);
    mayAlias = false;
    type = refType->getPointeeType();
    if (type->isIncompleteType())
      alignment = CharUnits();
    else
      alignment = getContext().getTypeAlignInChars(type);
    cvr = 0; // qualifiers don't recursively apply to referencee
  }

  sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(),FieldName.str().c_str());
  addr = scoutPtr(Builder.CreateLoad(addr, IRNameStr));

  if (field->hasAttr<AnnotateAttr>())
    addr = EmitFieldAnnotations(field, addr);

  // work around bug in llvm, this is similar to what a for loop appears to do
  // see EmitArraySubscriptExpr()
  //llvm::Value *Idx = Builder.CreateSExt(Index, IntPtrTy, "Xall.linearidx"); //forall or renderall
  //llvm::Value* Idx = Index;
  
  llvm::Value* Idx;
  if(Index->getType()->isPointerTy()){
    Idx = Builder.CreateLoad(scoutPtr(Index));
  }
  else{
    Idx = Index;
  }
  
  // get the correct element of the field depending on the index
  sprintf(IRNameStr, "%s.%s.element.ptr", MeshName.str().c_str(),FieldName.str().c_str());
  addr = scoutPtr(Builder.CreateInBoundsGEP(addr.getPointer(), Idx, IRNameStr));

  LValue LV = MakeAddrLValue(addr, type);

  LV.getQuals().addCVRQualifiers(cvr);


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

llvm::Value *CodeGenFunction::getMeshIndex(const MeshFieldDecl* MFD) {
  assert(!ForallStack.empty() && "empty forall stack");

  //const MeshType* MT = ForallStack[0].getMeshType();
  //auto& dims = MT->dimensions();
  
  llvm::Value* Index;


  if(MFD->isVertexLocated()) {
    int i = FindForallData(Vertices);
    assert(i >= 0 && "null vertex index while referencing vertex field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(ForallStack[i]->indexPtr);
  } else if(MFD->isEdgeLocated()) {
    int i = FindForallData(Edges);
    assert(i >= 0 && "null edge index while referencing edge field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(ForallStack[i]->indexPtr);
  } else if(MFD->isFaceLocated()) {
    int i = FindForallData(Faces);
    assert(i >= 0 && "null face index while referencing face field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(ForallStack[i]->indexPtr);
  } else if(MFD->isCellLocated()) {
    int i = FindForallData(Cells);
    assert(i >= 0 && "null cell index while referencing vertex field");
    Index = Builder.CreateLoad(ForallStack[i]->indexPtr);
  } else {
    Index = getLinearIdx();
  }
  return Index;
}

// compute the linear index based on cshift parameters
// with circular boundary conditions
llvm::Value *
CodeGenFunction::getCShiftLinearIdx(const MeshFieldDecl *MFD, SmallVector< llvm::Value *, 3 > args) {

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);

  //get the dimensions (Width, Height, Depth)
  SmallVector< llvm::Value *, 3 > dims;
  for(unsigned i = 0; i < args.size(); ++i) {
    sprintf(IRNameStr, "%s", DimNames[i]);
    if (MFD->isCellLocated()) {
      dims.push_back(Builder.CreateSExt(Builder.CreateLoad(LookupMeshDim(i), IRNameStr), Int64Ty));
    } else if (MFD->isVertexLocated()) {
      llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);
      dims.push_back(Builder.CreateAdd(
        Builder.CreateSExt(Builder.CreateLoad(LookupMeshDim(i), IRNameStr), Int64Ty), One));
    } else {
      assert(false && "Non Cell/Vertex in CShift");
    }
  }
  
  SmallVector< llvm::Value *, 3 > start;
  if(CGM.getCodeGenOpts().ScoutLegionSupport) {
    for(unsigned i = 0; i < args.size(); ++i) {
      sprintf(IRNameStr, "start.%s", DimNames[i]);
      start.push_back(Builder.CreateLoad(LookupMeshStart(i), IRNameStr));
    }
  }
  
  SmallVector< llvm::Value *, 3 > indices;
  for(unsigned i = 0; i < args.size(); ++i) {
    llvm::Value* ai = Builder.CreateSExt(args[i], Int64Ty);
    
    sprintf(IRNameStr, "forall.induct.%s", IndexNames[i]);

    llvm::Value *iv   = Builder.CreateLoad(LookupInductionVar(i), IRNameStr);
    
    // take index and add offset from cshift
    sprintf(IRNameStr, "cshift.rawindex.%s", IndexNames[i]);

    llvm::Value *Index;
    if(CGM.getCodeGenOpts().ScoutLegionSupport) {
      // add starting offset (for legion mode)
      Index = Builder.CreateAdd(Builder.CreateAdd(iv, ai), start[i], IRNameStr);
    } else {
      Index = Builder.CreateAdd(Builder.CreateSExt(iv, Int64Ty), ai, IRNameStr);
    }
    
    // make sure it is in range or wrap
    sprintf(IRNameStr, "cshift.index.%s", IndexNames[i]);
    llvm::Value *y = Builder.CreateSRem(Index, dims[i], IRNameStr);
    llvm::Value *Check = Builder.CreateICmpSLT(y, ConstantZero);
    llvm::Value *x = Builder.CreateSelect(Check,
        Builder.CreateAdd(dims[i], y), y);

    if(CGM.getCodeGenOpts().ScoutLegionSupport) {
      //remove starting offset (legion mode)
      llvm::Value *x2 = Builder.CreateSub(x, start[i]);
      indices.push_back(x2);
    } else {
      indices.push_back(x);
    }
  }

  switch(args.size()) {
    case 1:
      return indices[0];
    case 2: {
      // linearIdx = x + Width * y;
      llvm::Value *Hy    = Builder.CreateMul(dims[0], indices[1], "WidthxY");
      return Builder.CreateAdd(indices[0], Hy, "cshift.linearidx");
    }
    case 3: {
      // linearIdx = x + Height * (y + Width * z)
      llvm::Value *Wz     = Builder.CreateMul(dims[0], indices[2], "WidthxZ");
      llvm::Value *yWz    = Builder.CreateAdd(indices[1], Wz, "ypWidthxZ");
      llvm::Value *HyWz   = Builder.CreateMul(dims[1], yWz, "HxypWidthxZ");
      return Builder.CreateAdd(indices[0], HyWz, "cshift.linearidx");
    }
    default:
      assert(false && "bad number of args in cshift");
  }
  return indices[0]; // suppress warning.
}

// compute the linear index based on vfield parameters
llvm::Value*
CodeGenFunction::getVFieldLinearIdx(SmallVector<llvm::Value*, 3> args){
  using namespace llvm;
  
  auto& B = Builder;
  
  Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  
  Value* x = B.CreateLoad(LookupInductionVar(0), "x");
  Value* xv = B.CreateAdd(x, args[0], "xv");
  
  if(args.size() == 1){
    return xv;
  }
  
  Value* y = B.CreateLoad(LookupInductionVar(1), "y");
  Value* yc = B.CreateAdd(y, args[1], "yc");
  Value* width = B.CreateLoad(LookupMeshDim(0), "width");
  Value* width1 = B.CreateAdd(One, width, "width1");
  Value* yv = B.CreateAdd(xv, B.CreateMul(width1, yc), "yv");
  
  if(args.size() == 2){
    return yv;
  }

  Value* z = B.CreateLoad(LookupInductionVar(2), "z");
  Value* zc = B.CreateAdd(z, args[2], "zc");
  Value* height = B.CreateLoad(LookupMeshDim(1), "height");
  Value* height1 = B.CreateAdd(One, height, "height1");
  Value* zv = B.CreateMul(B.CreateMul(width1, height1), zc, "zv");
  
  return B.CreateAdd(yv, zv);
}

#if 0
//currently unused
static llvm::Value *
EmitBitCastOfLValueToProperType(CodeGenFunction &CGF,
                                llvm::Value *V, llvm::Type *IRType,
                                StringRef Name = StringRef()) {
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  return CGF.Builder.CreateBitCast(V, IRType->getPointerTo(AS), Name);
}
#endif

RValue CodeGenFunction::EmitCShiftExpr(ArgIterator ArgBeg, ArgIterator ArgEnd) {

  const Expr *A1E;
  //turn first arg into Expr
  if(const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(*(ArgBeg))) {
    // CShiftI, CShiftF, CshiftD get you here.
    A1E = CE->getSubExpr();
  } else if (const Expr *EE = dyn_cast<Expr>(*(ArgBeg))) {
    // "generic" CShift gets you here
    A1E = EE;
  } else {
    assert(false && "cshift first arg not expr");
  }

  // extract the cshift args
  SmallVector< llvm::Value *, 3 > args;
  while(++ArgBeg != ArgEnd) {
    RValue RV = EmitAnyExpr(*(ArgBeg));
    if(RV.isAggregate()) {
      args.push_back(RV.getAggregateAddress().getPointer());
    } else {
      args.push_back(RV.getScalarVal());
    }
  }

  // get the member expr for first arg.
  if(const MemberExpr *E = dyn_cast<MemberExpr>(A1E)) {
    // make sure this is a mesh

    if(const MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(E->getMemberDecl())) {
      // get the correct mesh member
      llvm::Value* mi = getCShiftLinearIdx(MFD, args);
      LValue LV = EmitMeshMemberExpr(E, mi);
      
      return RValue::get(Builder.CreateLoad(LV.getAddress(), "cshift.element"));
    }
  }
  assert(false && "Failed to translate Scout cshift expression to LLVM IR!");
}

RValue CodeGenFunction::EmitVFieldExpr(ArgIterator argsBegin,
                                       ArgIterator argsEnd) {
  const Expr* fieldExpr = *argsBegin;
  
  // extract the offset args
  SmallVector< llvm::Value *, 3 > args;
  while(++argsBegin != argsEnd) {
    RValue RV = EmitAnyExpr(*(argsBegin));
    if(RV.isAggregate()) {
      args.push_back(RV.getAggregateAddress().getPointer());
    } else {
      args.push_back(RV.getScalarVal());
    }
  }
  
  // get the member expr for first arg.
  if(const MemberExpr *E = dyn_cast<MemberExpr>(fieldExpr)) {
    // make sure this is a mesh
    if(isa<MeshFieldDecl>(E->getMemberDecl())) {
      // get the correct mesh member
      
      assert(false && "vfield unimplemented");
      
      //LValue LV = EmitMeshMemberExpr(E, getVFieldLinearIdx(args));
      
      //return RValue::get(Builder.CreateLoad(LV.getAddress(), "vfield.element"));
    }
  }
  assert(false && "Failed to translate Scout vfield expression to LLVM IR!");
}

// end-off shift
RValue CodeGenFunction::EmitEOShiftExpr(ArgIterator ArgBeg, ArgIterator ArgEnd) {


	llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
	llvm::Value *ConstantOne  = llvm::ConstantInt::get(Int32Ty, 1);

  const Expr *A1E;
  //turn first arg into Expr
  if(const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(*(ArgBeg))) {
    // eoshifti, eoshiftf, eoshiftd get you here.
    A1E = CE->getSubExpr();
  } else if (const Expr *EE = dyn_cast<Expr>(*(ArgBeg))) {
    // "generic" eoshift gets you here
    A1E = EE;
  } else {
    assert(false && "eoshift first arg not expr");
  }

  // extract 2nd arg which is the boundary value
  ++ArgBeg;
  RValue Boundary = EmitAnyExpr(*(ArgBeg));

  // extract the remaining eoshift args
  SmallVector< llvm::Value *, 3 > args;
  while(++ArgBeg != ArgEnd) {
    RValue RV = EmitAnyExpr(*(ArgBeg));
    if(RV.isAggregate()) {
      args.push_back(RV.getAggregateAddress().getPointer());
    } else {
      args.push_back(RV.getScalarVal());
    }
  }

  // get the member expr for first arg.
  if(const MemberExpr *E = dyn_cast<MemberExpr>(A1E)) {
    // make sure this is a mesh
    if(MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(E->getMemberDecl())) {

      //get the dimensions (Width, Height, Depth)
      SmallVector< llvm::Value *, 3 > dims;
      for(unsigned i = 0; i < args.size(); ++i) {
        sprintf(IRNameStr, "%s", DimNames[i]);
        if (MFD->isCellLocated()) {
          dims.push_back(Builder.CreateLoad(LookupMeshDim(i), IRNameStr));
        } else if (MFD->isVertexLocated()) {
          llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);
          dims.push_back(Builder.CreateAdd(
            Builder.CreateLoad(LookupMeshDim(i), IRNameStr), One));
        } else {
          assert(false && "Non Cell/Vertex in EOShift");
        }
      }

       //get the eoshift indices
       SmallVector< llvm::Value *, 3 > rawindices, indices;
       for(unsigned i = 0; i < args.size(); ++i) {
	       sprintf(IRNameStr, "forall.induct.%s", IndexNames[i]);
	       llvm::Value *iv = Builder.CreateLoad(LookupInductionVar(i), IRNameStr);

         llvm::Value* ai = Builder.CreateSExt(args[i], Int64Ty);
         
				 // take index and add offset from eoshift
				 sprintf(IRNameStr, "eoshift.rawindex.%s", IndexNames[i]);
				 rawindices.push_back(Builder.CreateAdd(iv, ai, IRNameStr));

	       // find if index will wrap
	       sprintf(IRNameStr, "eoshift.index.%s", IndexNames[i]);
	       indices.push_back(Builder.CreateURem(rawindices[i], dims[i], IRNameStr));
       }

       // setup flag

       Address flag = scoutPtr(Builder.CreateAlloca(Int32Ty, 0, "flag"));
       Builder.CreateStore(ConstantZero, flag);

       // setup basic blocks
       SmallVector< llvm::BasicBlock *, 3 > Start, Then, Else;
       llvm::BasicBlock *Done = createBasicBlock("done");
       for(unsigned i = 0; i < args.size(); ++i) {
    	   sprintf(IRNameStr, "start.%s", DimNames[i]);
    	   Start.push_back(createBasicBlock(IRNameStr));
           sprintf(IRNameStr, "then.%s", DimNames[i]);
           Then.push_back(createBasicBlock(IRNameStr));
           sprintf(IRNameStr, "else.%s", DimNames[i]);
           Else.push_back(createBasicBlock(IRNameStr));
       }

       // enter Start block
       Builder.CreateBr(Start[0]);

       // check to see if any of the indices are out of range
       // and if so set flag
       for(unsigned i = 0; i < args.size(); ++i) {

      	 // Start Block
         EmitBlock(Start[i]);
         // check if index is in range
         llvm::Value *Check = Builder.CreateICmpEQ(rawindices[i], indices[i]);
         Builder.CreateCondBr(Check, Then[i], Else[i]);

         // Then Block
         EmitBlock(Then[i]);
         //index is in range do nothing
         if(i == args.size()-1) {
        	 Builder.CreateBr(Done);
         } else {
        	 Builder.CreateBr(Start[i+1]);
         }

         // Else Block
         EmitBlock(Else[i]);
         //index is not in range increment flag and break
         llvm::Value *IncFlag = Builder.CreateAdd(Builder.CreateLoad(flag, "flag"),
        		 ConstantOne, "flaginc");
         Builder.CreateStore(IncFlag, flag);

         Builder.CreateBr(Done);
       }

       //setup basic blocks
       llvm::BasicBlock *FlagThen = createBasicBlock("FlagThen");
       llvm::BasicBlock *FlagElse = createBasicBlock("FlagElse");
       llvm::BasicBlock *Merge = createBasicBlock("Merge");

       // Done Block
       EmitBlock(Done);
       //check if flag is set
       llvm::Value *FlagCheck = Builder.CreateICmpNE(Builder.CreateLoad(flag, "flag"), ConstantZero);
       Builder.CreateCondBr(FlagCheck, FlagThen, FlagElse);

       // Then Block
       EmitBlock(FlagThen);
       //index is not in range
       llvm::Value *V1 = Boundary.getScalarVal();
       Builder.CreateBr(Merge);

       // Else Block
       EmitBlock(FlagElse);

       //all indices are in range, compute the linear index
       llvm::Value *idx = NULL;
       switch(args.size()) {
				 case 1:
					 idx = indices[0];
					 break;
				 case 2: {
					 // linearIdx = x + Width * y;
					 llvm::Value *Hy = Builder.CreateMul(dims[0], indices[1], "WidthxY");
					 idx = Builder.CreateAdd(indices[0], Hy, "eoshift.linearidx");
					 break;
				 }
				 case 3: {
					 // linearIdx = x + Height * (y + Width * z)
					 llvm::Value *Wz = Builder.CreateMul(dims[0], indices[2], "WidthxZ");
					 llvm::Value *yWz = Builder.CreateAdd(indices[1], Wz, "ypWidthxZ");
					 llvm::Value *HyWz = Builder.CreateMul(dims[1], yWz, "HxypWidthxZ");
					 idx = Builder.CreateAdd(indices[0], HyWz, "eoshift.linearidx");
					 break;
				 }
       }
       LValue LV = EmitMeshMemberExpr(E, idx);
       llvm::Value *V2 = Builder.CreateLoad(LV.getAddress(), "eoshift.element");
       Builder.CreateBr(Merge);


       //Merge Block
       EmitBlock(Merge);
       llvm::PHINode *PN = Builder.CreatePHI(Boundary.getScalarVal()->getType(), 2, "iftmp");
       PN->addIncoming(V1, FlagThen);
       PN->addIncoming(V2, FlagElse);
      
       return RValue::get(PN);

    }
  }
  assert(false && "Failed to translate Scout eoshift expression to LLVM IR!");
}


//emit width()/height()/depth()/rank() with mesh as argument
RValue CodeGenFunction::EmitMeshParameterExpr(const Expr *E, MeshParameterOffset offset) {
  llvm::Value* value = 0;
  unsigned int nfields = 0;

  static const char *names[]   = { "width", "height", "depth", "rank" };

  // Get the expr we are after, might have leading casts and star
  Expr *EE = const_cast<Expr *>(E);
  if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(EE)) {
    EE = ICE->getSubExpr();
    if(const UnaryOperator *UO = dyn_cast<UnaryOperator>(EE)) {
      EE = UO->getSubExpr();
      if (ImplicitCastExpr *ICE2 = dyn_cast<ImplicitCastExpr>(EE)) {
        EE = ICE2->getSubExpr();
      }
    }
  }

  if(const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(EE)) {

    //get number of fields
    if(const MeshType *MT = dyn_cast<MeshType>(DRE->getType())) {
      nfields = MT->getDecl()->fields();
    } else { // might have mesh ptr
      const Type *T = DRE->getType().getTypePtr()->getPointeeType().getTypePtr();
      if(T) {
        if(const MeshType *MT = dyn_cast<MeshType>(T)) {
          nfields = MT->getDecl()->fields();
        }
      }
    }

    if(const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      Address BaseAddr = GetMeshBaseAddr(VD);
      llvm::StringRef MeshName = BaseAddr.getPointer()->getName();


      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), names[offset]);
      Address addr = Builder.CreateMeshGEP(BaseAddr, 0, nfields+offset, IRNameStr);

      sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), names[offset]);
      value = Builder.CreateLoad(addr, IRNameStr);
      return RValue::get(Builder.CreateTrunc(value, Int32Ty));
    }
  }
  //sema should make sure we don't get here.
  assert(false && "Failed to emit Mesh Parameter");
}

RValue CodeGenFunction::EmitTailExpr(void) {
  int i = FindForallData(Edges);
  assert(i >= 0 && "failed to find edge index");
  
  Address EdgeIndex = ForallStack[i]->indexPtr;
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  Rank = Builder.CreateTrunc(Rank, Int32Ty);
  llvm::Value *Index = Builder.CreateLoad(EdgeIndex, "forall.edgeIndex");
  Index = Builder.CreateTrunc(Index, Int32Ty);
  llvm::Value *Dx = Builder.CreateLoad(MeshDims[0]);
  Dx = Builder.CreateTrunc(Dx, Int32Ty);
  llvm::Value *Dx1 = Builder.CreateAdd(Dx,One);
  llvm::Value *Dy = Builder.CreateLoad(MeshDims[1]);
  Dy = Builder.CreateTrunc(Dy, Int32Ty);
  llvm::Value *Dy1 = Builder.CreateAdd(Dy,One);

  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Merge = createBasicBlock("rank.merge");

  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, Three);
  Builder.CreateCondBr(Check3, Then3, Else3);

  // rank = 3
  EmitBlock(Then3);
  llvm::Value *Result3 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));
  //SC_TODO: add correct x,y,z
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(0), "tail.x");
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(1), "tail.y");
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(2), "tail.z");
  Result3 = Builder.CreateInsertElement(Result3, Index,
      Builder.getInt32(3), "tail.idx");

  Builder.CreateBr(Merge);

  // rank != 3
  EmitBlock(Else3);

  // rank = 2
  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);
  EmitBlock(Then2);
  llvm::Value *Result2 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));

  // number of edges in x direction
  llvm::Value *nx = Builder.CreateMul(Dx1, Dy);
  // what is direction for this Index
  llvm::Value *dir = Builder.CreateICmpSLT(Index, nx);
  llvm::Value *Index2 = Builder.CreateSub(Index, nx);

  // for edges in x direction
  llvm::Value *x1 = Builder.CreateSRem(Index, Dx1);
  llvm::Value *y1 = Builder.CreateSRem(Builder.CreateSDiv(Index, Dx1), Dy);
  // for edges in y direction
  llvm::Value *x2 = Builder.CreateSRem(Index2, Dx);
  llvm::Value *y2 = Builder.CreateSRem(Builder.CreateSDiv(Index2, Dx), Dy1);

  llvm::Value *x = Builder.CreateSelect(dir, x1, x2);
  llvm::Value *y = Builder.CreateSelect(dir, y1, y2);

  Result2 = Builder.CreateInsertElement(Result2, x,
      Builder.getInt32(0), "tail.x");
  Result2 = Builder.CreateInsertElement(Result2, y,
      Builder.getInt32(1), "tail.y");
  Result2 = Builder.CreateInsertElement(Result2, Zero,
      Builder.getInt32(2), "tail.z");
  Result2 = Builder.CreateInsertElement(Result2, Index,
      Builder.getInt32(3), "tail.idx");

  Builder.CreateBr(Merge);

  // rank !=2 (rank = 1)
  EmitBlock(Else2);
  llvm::Value *Result1 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));
  Result1 = Builder.CreateInsertElement(Result1, Index,
      Builder.getInt32(0), "tail.x");
  Result1 = Builder.CreateInsertElement(Result1, Zero,
      Builder.getInt32(1), "tail.y");
  Result1 = Builder.CreateInsertElement(Result1, Zero,
      Builder.getInt32(2), "tail.z");
  Result1 = Builder.CreateInsertElement(Result1, Index,
      Builder.getInt32(3), "tail.idx");

  Builder.CreateBr(Merge);

  // Merge Block
  EmitBlock(Merge);
  llvm::PHINode *PN = Builder.CreatePHI(llvm::VectorType::get(Int32Ty, 4), 3, "tail.phi");
  PN->addIncoming(Result3, Then3);
  PN->addIncoming(Result2, Then2);
  PN->addIncoming(Result1, Else2);

  return RValue::get(PN);
}

RValue CodeGenFunction::EmitHeadExpr(void) {
  int i = FindForallData(Edges);
  assert(i >= 0 && "failed to find edge index");
  
  Address EdgeIndex = ForallStack[i]->indexPtr;;

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  Rank = Builder.CreateTrunc(Rank, Int32Ty);
  llvm::Value *Index = Builder.CreateLoad(EdgeIndex, "forall.edgeIndex");
  Index = Builder.CreateTrunc(Index, Int32Ty);
  llvm::Value *Dx = Builder.CreateLoad(MeshDims[0]);
  Dx = Builder.CreateTrunc(Dx, Int32Ty);
  llvm::Value *Dx1 = Builder.CreateAdd(Dx,One);
  llvm::Value *Dy = Builder.CreateLoad(MeshDims[1]);
  Dy = Builder.CreateTrunc(Dy, Int32Ty);
  llvm::Value *Dy1 = Builder.CreateAdd(Dy,One);

  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Merge = createBasicBlock("rank.merge");

  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, Three);
  Builder.CreateCondBr(Check3, Then3, Else3);

  // rank = 3
  EmitBlock(Then3);
  llvm::Value *Result3 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));
  //SC_TODO: add correct x,y,z
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(0), "head.x");
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(1), "head.y");
  Result3 = Builder.CreateInsertElement(Result3, Zero,
      Builder.getInt32(2), "head.z");
  Result3 = Builder.CreateInsertElement(Result3, Index,
      Builder.getInt32(3), "head.idx");

  Builder.CreateBr(Merge);

  // rank != 3
  EmitBlock(Else3);

  // rank = 2
  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);
  EmitBlock(Then2);
  llvm::Value *Result2 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));

  // number of edges in x direction
  llvm::Value *nx = Builder.CreateMul(Dx1, Dy);
  // what is direction for this Index
  llvm::Value *dir = Builder.CreateICmpSLT(Index, nx);
  llvm::Value *Index2 = Builder.CreateSub(Index, nx);

  // for edges in x direction
  llvm::Value *x1 = Builder.CreateSRem(Index, Dx1);
  llvm::Value *y1 = Builder.CreateAdd(Builder.CreateSRem(Builder.CreateSDiv(Index, Dx1), Dy), One);
  // for edges in y direction
  llvm::Value *x2 = Builder.CreateAdd(Builder.CreateSRem(Index2, Dx), One);
  llvm::Value *y2 = Builder.CreateSRem(Builder.CreateSDiv(Index2, Dx), Dy1);

  llvm::Value *x = Builder.CreateSelect(dir, x1, x2);
  llvm::Value *y = Builder.CreateSelect(dir, y1, y2);

  Result2 = Builder.CreateInsertElement(Result2, x,
      Builder.getInt32(0), "head.x");
  Result2 = Builder.CreateInsertElement(Result2, y,
      Builder.getInt32(1), "head.y");
  Result2 = Builder.CreateInsertElement(Result2, Zero,
      Builder.getInt32(2), "head.z");
  Result2 = Builder.CreateInsertElement(Result2, Index,
      Builder.getInt32(3), "head.idx");

  Builder.CreateBr(Merge);

  // rank !=2 (rank = 1)
  EmitBlock(Else2);
  llvm::Value *Index1 = Builder.CreateAdd(Index, One);
  llvm::Value *Result1 =
      llvm::UndefValue::get(llvm::VectorType::get(Int32Ty, 4));
  Result1 = Builder.CreateInsertElement(Result1, Index1,
      Builder.getInt32(0), "head.x");
  Result1 = Builder.CreateInsertElement(Result1, Zero,
      Builder.getInt32(1), "head.y");
  Result1 = Builder.CreateInsertElement(Result1, Zero,
      Builder.getInt32(2), "head.z");
  Result1 = Builder.CreateInsertElement(Result1, Index1,
      Builder.getInt32(3), "head.idx");

  Builder.CreateBr(Merge);

  // Merge Block
  EmitBlock(Merge);
  llvm::PHINode *PN = Builder.CreatePHI(llvm::VectorType::get(Int32Ty, 4), 3, "head.phi");
  PN->addIncoming(Result3, Then3);
  PN->addIncoming(Result2, Then2);
  PN->addIncoming(Result1, Else2);

  return RValue::get(PN);
}

RValue
CodeGenFunction::EmitSaveMeshExpr(ArgIterator argsBegin, ArgIterator argsEnd){
  // proper checking is already done in Sema::CheckSaveMeshCall()
  
  auto R = CGM.getScoutRuntime();
  
  using ValueVec = std::vector<llvm::Value*>;
  
  const DeclRefExpr* base = cast<DeclRefExpr>(*argsBegin);
  const VarDecl* vd = cast<VarDecl>(base->getDecl());
  const MeshType* mt = cast<MeshType>(vd->getType().getTypePtr());

  auto& dims = mt->dimensions();
  
  MeshDecl* md = mt->getDecl();
  
  ++argsBegin;
  
  const StringLiteral* pathLiteral = dyn_cast<StringLiteral>(*argsBegin);
  std::string path = pathLiteral->getString();
  
  Address meshAddr = GetMeshBaseAddr(vd);
  Address topology = Builder.CreateStructGEP(meshAddr, md->fields(), getPointerAlign());
  topology = scoutPtr(Builder.CreateLoad(topology, "topology.ptr"));
  
  Address meshPtr = Builder.CreateBitCast(meshAddr, VoidPtrTy);
  
  llvm::StructType* structTy =
  cast<llvm::StructType>(meshAddr.getPointer()->getType()->getContainedType(0));
  
  CGScoutRuntime& r = CGM.getScoutRuntime();
  
  std::vector<llvm::Value*> args = {meshPtr.getPointer()};
  Builder.CreateCall(r.SaveMeshStartFunc(), args);
  
  for(auto itr = md->field_begin(), itrEnd = md->field_end();
      itr != itrEnd; ++itr){

    MeshFieldDecl* field = *itr;
    
    llvm::Value* fieldName = Builder.CreateGlobalStringPtr(field->getName());
    fieldName = Builder.CreateBitCast(fieldName, llvm::PointerType::get(CGM.Int8Ty, 0));
    
    unsigned idx = CGM.getTypes().getCGMeshLayout(md).getLLVMFieldNo(field);
    
    Address fieldAddr = Builder.CreateStructGEP(meshAddr, idx, getPointerAlign());
    fieldAddr = scoutPtr(Builder.CreateLoad(fieldAddr, "mesh.field"));
    
    llvm::Type* fieldTy = structTy->getContainedType(idx);
    llvm::PointerType* ptrTy = dyn_cast<llvm::PointerType>(fieldTy);
    assert(ptrTy && "expected a pointer");
    
    fieldTy = ptrTy->getElementType();
    
    llvm::Value* scalarKind;
    
    if(fieldTy->isIntegerTy(32)){
      scalarKind = r.Int32Val;
    }
    else if(fieldTy->isIntegerTy(64)){
      scalarKind = r.Int64Val;
    }
    else if(fieldTy->isFloatTy()){
      scalarKind = r.FloatVal;
    }
    else if(fieldTy->isDoubleTy()){
      scalarKind = r.DoubleVal;
    }
    else{
      assert(false && "invalid scalar kind");
    }
    
    llvm::Value* numItems;
    llvm::Value* elementKind;
    
    if (field->isCellLocated()) {
      ValueVec args = {topology.getPointer(), llvm::ConstantInt::get(Int32Ty, dims.size())};
      numItems = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      elementKind = r.CellVal;
    } else if (field->isVertexLocated()) {
      ValueVec args = {topology.getPointer(), llvm::ConstantInt::get(Int32Ty, 0)};
      numItems = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      elementKind = r.VertexVal;
    } else if(field->isEdgeLocated()) {
      ValueVec args = {topology.getPointer(), llvm::ConstantInt::get(Int32Ty, 1)};
      numItems = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      elementKind = r.EdgeVal;
    } else if(field->isFaceLocated()) {
      ValueVec args = {topology.getPointer(), llvm::ConstantInt::get(Int32Ty, dims.size() - 1)};
      numItems = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      elementKind = r.FaceVal;
    } else {
      assert(false && "invalid element kind");
    }
    
    fieldAddr = Builder.CreateBitCast(fieldAddr, VoidPtrTy);
    
    args = {meshPtr.getPointer(), fieldName, numItems, elementKind, scalarKind, fieldAddr.getPointer()};
    Builder.CreateCall(r.SaveMeshAddFieldFunc(), args);
  }
  
  llvm::Value* pathValue = Builder.CreateGlobalStringPtr(path);
  pathValue =
  Builder.CreateBitCast(pathValue, llvm::PointerType::get(CGM.Int8Ty, 0));
  
  args = {meshPtr.getPointer(), pathValue};
  Builder.CreateCall(r.SaveMeshEndFunc(), args);

  return RValue::get(llvm::ConstantInt::get(Int32Ty, 0));
}

RValue CodeGenFunction::EmitSwapFieldsExpr(ArgIterator argsBegin, ArgIterator argsEnd){
  // proper checking is already done in Sema::CheckSwapFieldsCall()
  
  Address fieldPtr[2] = {Address::invalid(),Address::invalid()};
  Address meshAddr = Address::invalid();
  
  for(size_t i = 0; i < 2; ++i){
    const MemberExpr* memberExpr = cast<MemberExpr>(*argsBegin);
    const DeclRefExpr* base = cast<DeclRefExpr>(memberExpr->getBase());
    const VarDecl* vd = cast<VarDecl>(base->getDecl());
    //const UniformMeshType* mt = cast<UniformMeshType>(vd->getType().getTypePtr());

    if(i == 0){
      meshAddr = GetMeshBaseAddr(vd);
    }
    
    MeshFieldDecl* field = cast<MeshFieldDecl>(memberExpr->getMemberDecl());
    
    const MeshDecl* mesh = field->getParent();
    unsigned idx = CGM.getTypes().getCGMeshLayout(mesh).getLLVMFieldNo(field);
    
    fieldPtr[i] = Builder.CreateStructGEP(meshAddr, idx, getPointerAlign());
    
    ++argsBegin;
  }
  
  llvm::Value* firstField = Builder.CreateLoad(fieldPtr[0]);
  Builder.CreateStore(Builder.CreateLoad(fieldPtr[1]), fieldPtr[0]);
  Builder.CreateStore(firstField, fieldPtr[1]);
  
  return RValue::get(llvm::ConstantInt::get(Int32Ty, 0));
}

void CodeGenFunction::EmitQueryExpr(const ValueDecl* VD,
                                    LValue LV,
                                    const QueryExpr* QE){
  using namespace std;
  using namespace llvm;
  
  //typedef vector<llvm::Value*> ValueVec;
  typedef vector<llvm::Type*> TypeVec;

  CGBuilderTy& B = Builder;
  LLVMContext& C = getLLVMContext();
  
  Value* One = ConstantInt::get(Int64Ty, 1);
  
  BasicBlock* prevBlock = B.GetInsertBlock();
  BasicBlock::iterator prevPoint = B.GetInsertPoint();
  
  const MemberExpr* memberExpr = QE->getField();
  const Expr* pred = QE->getPredicate();
  
  const DeclRefExpr* base = dyn_cast<DeclRefExpr>(memberExpr->getBase());
  assert(base && "expected a DeclRefExpr");

  const ImplicitMeshParamDecl* imp =
  dyn_cast<ImplicitMeshParamDecl>(base->getDecl());
  assert(base && "expected an ImplicitMeshParamDecl");

  MeshElementType et = imp->getElementType();
  
  const VarDecl* mvd = imp->getBaseVarDecl();
  const MeshType* mt;
  
  if(const PointerType* pt = dyn_cast<PointerType>(mvd->getType())){
    mt = dyn_cast<MeshType>(pt->getPointeeType());
  }
  else{
    mt = dyn_cast<MeshType>(mvd->getType());
  }
  
  const MeshDecl* md = mt->getDecl();
  
  //auto& dims = mt->dimensions();
  
  TypeVec params =
  {llvm::PointerType::get(ConvertType(mvd->getType()), 0),
    llvm::PointerType::get(Int8Ty, 0), Int64Ty, Int64Ty};

  llvm::FunctionType* ft =
  llvm::FunctionType::get(llvm::Type::getVoidTy(C), params, false);
  
  Function* queryFunc = Function::Create(ft,
                                         Function::ExternalLinkage,
                                         "MeshQueryFunction",
                                         &CGM.getModule());
  
  auto aitr = queryFunc->arg_begin();

  Value* meshPtr = &*aitr;
  aitr++;
  meshPtr->setName("mesh");

  Value* outPtr = &*aitr;
  aitr++;
  outPtr->setName("outMask");
  
  Value* start = &*aitr;
  aitr++;
  start->setName("start");
  
  Value* end = &*aitr;
  aitr++;
  end->setName("end");
  
  Address baseAddr = GetAddrOfLocalVar(mvd);
  
  LocalDeclMap.erase(mvd);
  setAddrOfLocalVar(mvd, scoutPtr(meshPtr));
  
  BasicBlock* entry = BasicBlock::Create(C, "entry", queryFunc);
  B.SetInsertPoint(entry);
  
  Address inductPtr = Address(B.CreateAlloca(Int64Ty, 0, "induct.ptr"), getPointerAlign());

  B.CreateStore(start, inductPtr);
  
  BasicBlock* loopBlock = BasicBlock::Create(C, "query.loop", queryFunc);
  B.CreateBr(loopBlock);
  B.SetInsertPoint(loopBlock);

  assert(ForallStack.empty());
  
  ForallData data;
  data.meshVarDecl = mvd;
  data.indexPtr = inductPtr;
  data.elementType = et;
  
  Address topology = B.CreateStructGEP(scoutPtr(meshPtr), md->fields(), getPointerAlign());
  llvm::Value *topov = B.CreateLoad(topology, "topology.ptr");
  data.topology = scoutPtr(topov);
  
  ForallStack.emplace_back(&data);
  
  Value* result = EmitAnyExprToTemp(pred).getScalarVal();

  size_t bits = result->getType()->getPrimitiveSizeInBits();
  
  if(bits < 8){
    result = B.CreateZExt(result, Int8Ty, "result");
  }
  else if(bits > 8){
    result = B.CreateTrunc(result, Int8Ty, "result");
  }
  
  ForallStack.pop_back();
  
  Value* induct = B.CreateLoad(inductPtr, "induct");

  Value* outPosPtr = B.CreateGEP(outPtr, induct, "outPos.ptr");
  B.CreateStore(result, scoutPtr(outPosPtr));

  Value* nextInduct = B.CreateAdd(induct, One, "nextInduct");
  B.CreateStore(nextInduct, inductPtr);

  BasicBlock* condBlock = BasicBlock::Create(C, "query.cond", queryFunc);
  
  B.CreateBr(condBlock);
  
  BasicBlock* mergeBlock = BasicBlock::Create(C, "query.merge", queryFunc);
  
  B.SetInsertPoint(condBlock);
  
  Value* cond = B.CreateICmpULT(induct, end, "cond");
  B.CreateCondBr(cond, loopBlock, mergeBlock);

  B.SetInsertPoint(mergeBlock);
  
  B.CreateRetVoid();
  
  B.SetInsertPoint(prevBlock, prevPoint);
 
  LocalDeclMap.erase(mvd);
  setAddrOfLocalVar(mvd, baseAddr);
  
  Address qp = LV.getAddress();

  Address funcField = B.CreateStructGEP(qp, 0, getPointerAlign(), "query.func.ptr");
  Address  meshPtrField = B.CreateStructGEP(qp, 1, getPointerAlign(), "query.mesh.ptr");
  
  B.CreateStore(B.CreateBitCast(queryFunc, CGM.VoidPtrTy), funcField);
  B.CreateStore(B.CreateBitCast(baseAddr.getPointer(), CGM.VoidPtrTy), meshPtrField);
  
  LocalDeclMap.insert({VD, qp});
}

RValue CodeGenFunction::EmitMPosition(const CallExpr *E , unsigned int index) {

  static const char *IndexNames[] = { "x", "y", "z"};

  // check if we are NOT in a forall or renderall
  if (!CurrentMeshVarDecl) {
    CGM.getDiags().Report(E->getExprLoc(), diag::warn_mesh_intrinsic_outside_scope);
    return RValue::get(llvm::ConstantFP::get(FloatTy, 0));
  }

  // check if this is an ALE mesh
  const ALEMeshType* aleMeshType = dyn_cast<ALEMeshType>(CurrentMeshVarDecl->getType().getTypePtr());
  if (!aleMeshType) {
    CGM.getDiags().Report(E->getExprLoc(), diag::err_invalid_mposition_call) << "must operate on ALE mesh vertices";
    return RValue::get(llvm::ConstantFP::get(FloatTy, 0));
  }

  // otherwise, we compute the index of the mpositionx
  sprintf(IRNameStr, "mposition.index.%s", IndexNames[index]);
  llvm::Value *indexVal = Builder.CreateAdd(
      Builder.CreateLoad(LookupInductionVar(index)),
      Builder.CreateLoad(LookupMeshStart(index)),
      IRNameStr);


  Address BaseAddr = GetMeshBaseAddr(CurrentMeshVarDecl);
  llvm::StringRef MeshName = BaseAddr.getPointer()->getName();

  // We GEP to the index-th field of the record 
  sprintf(IRNameStr, "%s.%s.ptr.%s", MeshName.str().c_str(), "mposition", IndexNames[index]);
  Address value = Builder.CreateMeshGEP(BaseAddr, 0, index, IRNameStr);

  // load that address value
  sprintf(IRNameStr, "%s.%s.%s", MeshName.str().c_str(), "mposition", IndexNames[index]);
  value = scoutPtr(Builder.CreateLoad(value, IRNameStr));

  // work around bug in llvm, this is similar to what a for loop appears to do
  // see EmitArraySubscriptExpr()
  sprintf(IRNameStr, "Xall.linearidx.mposition.%s", IndexNames[index]);
  llvm::Value *Idx = Builder.CreateSExt(indexVal, IntPtrTy, IRNameStr); //forall or renderall

  // get the correct element of the mposition field depending on the index
  sprintf(IRNameStr, "%s.%s.%s.element.ptr", MeshName.str().c_str(), "mposition", IndexNames[index]);
  llvm::Value *valueptr = Builder.CreateInBoundsGEP(value.getPointer(), Idx, IRNameStr);

  // number of args is already known to be 0 or 1 as it was checked in sema
  // return the value of the mposition if it has no arguments
  if(E->getNumArgs() == 0) { 
    sprintf(IRNameStr, "%s.%s.%s.element", MeshName.str().c_str(), "mposition", IndexNames[index]);
    llvm::Value *v = Builder.CreateLoad(scoutPtr(valueptr), IRNameStr);
    return RValue::get(v);

  } else { // otherwise, set the mposition[x,y,z] to the argument

    RValue argval = EmitAnyExpr(E->getArg(0));
    llvm::Value* newvalue;
    if(argval.isAggregate()) { 
      Address addr = argval.getAggregateAddress();
      sprintf(IRNameStr, "%s.%s.%s.newaggvalue", MeshName.str().c_str(), "mposition", IndexNames[index]);
      newvalue = Builder.CreateLoad(addr, IRNameStr);
      sprintf(IRNameStr, "%s.%s.%s.newcastedaggvalue", MeshName.str().c_str(), "mposition", IndexNames[index]);
      newvalue = Builder.CreateFPCast(newvalue, FloatTy, IRNameStr);
    } else {
      newvalue = argval.getScalarVal();
      sprintf(IRNameStr, "%s.%s.%s.newcastedscalarvalue", MeshName.str().c_str(), "mposition", IndexNames[index]);
      newvalue = Builder.CreateFPCast(newvalue, FloatTy, IRNameStr);
    }
    Builder.CreateStore(newvalue, scoutPtr(valueptr));
    return RValue::get(newvalue);
  }
}

RValue CodeGenFunction::EmitMPositionVector(const CallExpr *E) {
  static const char *IndexNames[] = { "x", "y", "z"};

  SmallVector<uint32_t, 3 > Elts;
  for (unsigned i = 0, e = 3; i != e; ++i) {
    Elts.push_back(0);
  }

  // check if we are NOT in a forall or renderall
  if (!CurrentMeshVarDecl) {
    CGM.getDiags().Report(E->getExprLoc(), diag::warn_mesh_intrinsic_outside_scope);
    return RValue::get(llvm::ConstantDataVector::getFP(getLLVMContext(), Elts));
    //return RValue::get(llvm::ConstantFP::get(FloatTy, 0));
  }

  // check if this is an ALE mesh
  const ALEMeshType* aleMeshType = dyn_cast<ALEMeshType>(CurrentMeshVarDecl->getType().getTypePtr());
  if (!aleMeshType) {
    CGM.getDiags().Report(E->getExprLoc(), diag::err_invalid_mposition_call) << "must operate on ALE mesh vertices";
    return RValue::get(llvm::ConstantDataVector::getFP(getLLVMContext(), Elts));
  }

  llvm::Value *Result;
  if (E->getNumArgs() == 0) { 
    Result = llvm::UndefValue::get(llvm::VectorType::get(FloatTy, 3));
  } else {
    RValue argval = EmitAnyExpr(E->getArg(0));
    llvm::Value* newvalue;
    if (argval.isAggregate()) {
      CGM.getDiags().Report(E->getExprLoc(), diag::err_invalid_mposition_call) << "arg must be a vector";
      return RValue::get(llvm::ConstantDataVector::getFP(getLLVMContext(), Elts));
    } else {
      newvalue = argval.getScalarVal();
      sprintf(IRNameStr, "newmpositionvec");
      // the result of this operation is actually the vector argument
      Result = Builder.CreateFPCast(newvalue, llvm::VectorType::get(FloatTy, 3), IRNameStr);
    }
  }


  for (unsigned i = 0; i <= 2; ++i) {
    // otherwise, we compute the index of the mpositionx
    llvm::Value *indexVal;
    sprintf(IRNameStr, "mposition.index.%s", IndexNames[i]);
    indexVal = Builder.CreateAdd(
      Builder.CreateLoad(LookupInductionVar(i)),
      Builder.CreateLoad(LookupMeshStart(i)),
      IRNameStr);
    Address BaseAddr = GetMeshBaseAddr(CurrentMeshVarDecl);
    llvm::StringRef MeshName = BaseAddr.getPointer()->getName();

    // We GEP to the ith field of the record 
    sprintf(IRNameStr, "%s.%s.ptr.%s", MeshName.str().c_str(), "mposition", IndexNames[i]);
    Address value = Builder.CreateMeshGEP(BaseAddr, 0, i, IRNameStr);

    // load that address value
    sprintf(IRNameStr, "%s.%s.%s", MeshName.str().c_str(), "mposition", IndexNames[i]);
    value = scoutPtr(Builder.CreateLoad(value, IRNameStr));

    // work around bug in llvm, this is similar to what a for loop appears to do
    // see EmitArraySubscriptExpr()
    sprintf(IRNameStr, "Xall.linearidx.mposition.%s", IndexNames[i]);
    llvm::Value *Idx = Builder.CreateSExt(indexVal, IntPtrTy, IRNameStr); //forall or renderall

    // get the correct element of the mposition field depending on the index
    sprintf(IRNameStr, "%s.%s.%s.element.ptr", MeshName.str().c_str(), "mposition", IndexNames[i]);
    llvm::Value *valueptr = Builder.CreateInBoundsGEP(value.getPointer(), Idx, IRNameStr);

    // number of args is already known to be 0 or 1 as it was checked in sema
    // return the value of the mposition if it has no arguments
    if(E->getNumArgs() == 0) {  // extract the vector values and return them
      sprintf(IRNameStr, "%s.%s.%s.element", MeshName.str().c_str(), "mposition", IndexNames[i]);
      llvm::Value *v = Builder.CreateLoad(scoutPtr(valueptr), IRNameStr);
      sprintf(IRNameStr, "%s.%s.%s.insertelement", MeshName.str().c_str(), "mposition", IndexNames[i]);
      Result = Builder.CreateInsertElement(Result, v, Builder.getInt32(i), IRNameStr);
    } else { // otherwise, set the mposition to the argument 
      sprintf(IRNameStr, "%s.%s.%s.extractelement", MeshName.str().c_str(), "mposition", IndexNames[i]);
      llvm::Value* newvalue = Builder.CreateExtractElement(Result, i, IRNameStr); 
      Builder.CreateStore(newvalue, scoutPtr(valueptr));
    }
  }
  return RValue::get(Result);
}

llvm::Value *CodeGenFunction::FindGIndex(unsigned int dim) {

   ForallData* data = ForallStack.back();
   const MeshType* mt = data->getMeshType();
   auto& dims = mt->dimensions();

   llvm::Value* idx = Builder.CreateLoad(data->indexPtr);

   // for partitioned legion case need to add start to induction var
   if (ForallStack.size() == 1 && dim < 3) {
      idx = Builder.CreateAdd(Builder.CreateLoad(LookupMeshStart(dim)), idx);
   }

   // get current iv
   return LinearIdx2InductionVar(idx, data->elementType, dim, dims.size());
}


llvm::Value *CodeGenFunction::EmitGIndex(unsigned int dim) {
  static const char *IndexNames[] = { "x", "y", "z", "w"};
  sprintf(IRNameStr, "gindex.%s", IndexNames[dim]);
  return Builder.CreateTrunc(FindGIndex(dim), Int32Ty, IRNameStr);
}

llvm::Value *CodeGenFunction::EmitLIndex(unsigned int dim) {  
  llvm::Value* idx;

  if (dim == 3) {
    idx = Builder.CreateLoad(ForallStack.back()->indexPtr);
    return Builder.CreateTrunc(idx, Int32Ty, "lindex.w");
  }

  // get current iv
  idx = FindGIndex(dim);

  if (ForallStack.size() > 1) {
    ForallData* data = ForallStack.back();
    const MeshType* mt = data->getMeshType();
    auto& dims = mt->dimensions();
    // get top iv
    llvm::Value *topiv = LinearIdx2InductionVar(
        Builder.CreateLoad(ForallStack[0]->indexPtr), ForallStack[0]->elementType,
        dim, dims.size());
    // SC_TODO: only works for cells/vert and vert/cells
    idx = Builder.CreateSub(idx, topiv);
    if (ForallStack[0]->elementType == Vertices) {
      llvm::Value* one = llvm::ConstantInt::get(Int64Ty, 1);
      idx = Builder.CreateAdd(idx, one);
    }
  }

  sprintf(IRNameStr, "lindex.%s", IndexNames[dim]);
  return Builder.CreateTrunc(idx, Int32Ty, IRNameStr);
}
