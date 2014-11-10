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
#include "Scout/CGLegionRuntime.h"
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
  const ValueDecl *VD = cast<ValueDecl>(ND);
  CharUnits Alignment = getContext().getDeclAlign(ND);
  llvm::Value *idx = getLinearIdx();
  llvm::Value* ep = Builder.CreateInBoundsGEP(Color, idx);
  return MakeAddrLValue(ep, VD->getType(), Alignment);
}

LValue
CodeGenFunction::EmitMeshMemberExpr(const MemberExpr *E, llvm::Value *Index) {

  DeclRefExpr* Base;
  if(ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E->getBase())) {
    Base = cast<DeclRefExpr>(CE->getSubExpr());
  } else {
    Base = cast<DeclRefExpr>(E->getBase());
  }

  llvm::Value *Addr;

  // inside forall we are referencing the implicit mesh e.g. 'c' in forall cells c in mesh
  if (ImplicitMeshParamDecl *IMPD = dyn_cast<ImplicitMeshParamDecl>(Base->getDecl())) {
    // lookup underlying mesh instead of implicit mesh
    GetMeshBaseAddr(IMPD->getMeshVarDecl(), Addr);
  } else if (ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(Base->getDecl())) {
    llvm::errs() << "ParmVarDecl in EmitMeshMemberExpr must be stencil?\n";
    Addr = LocalDeclMap.lookup(PVD);
    if(Addr) {
      // If Mesh ptr then load
      const Type *T = PVD->getType().getTypePtr();
      if(T->isAnyPointerType() || T->isReferenceType()) {
        Addr = Builder.CreateLoad(Addr);
      }
    }
  } else {
    llvm_unreachable("Cannot lookup underlying mesh");
  }

  LValue BaseLV  = MakeAddrLValue(Addr, E->getType());
  // assume we have already checked that we are working w/ a mesh and cast to MeshField Decl
  MeshFieldDecl* MFD = cast<MeshFieldDecl>(E->getMemberDecl());
  return EmitLValueForMeshField(BaseLV,  cast<MeshFieldDecl>(MFD), Index); //SC_TODO: why cast?
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
  StringRef MeshName = base.getAddress()->getName();
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
    LValue LV = MakeAddrLValue(addr, field->getType(), CharUnits::Zero());
    return LV;
  }

  if (field->isBitField()) {
    const CGMeshLayout &ML = CGM.getTypes().getCGMeshLayout(mesh);
    const CGBitFieldInfo &Info = ML.getBitFieldInfo(field);
    llvm::Value *Addr = base.getAddress();
    unsigned Idx = ML.getLLVMFieldNo(field);
    if (Idx != 0)
      // For structs, we GEP to the field that the record layout suggests.
      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(),FieldName.str().c_str());
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

  // We GEP to the field that the record layout suggests.
  unsigned idx = CGM.getTypes().getCGMeshLayout(mesh).getLLVMFieldNo(field);

  sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(),FieldName.str().c_str());
  addr = Builder.CreateStructGEP(addr, idx, IRNameStr);

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

  sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(),FieldName.str().c_str());
  addr = Builder.CreateLoad(addr, IRNameStr);

  if (field->hasAttr<AnnotateAttr>())
    addr = EmitFieldAnnotations(field, addr);

  // work around bug in llvm, this is similar to what a for loop appears to do
  // see EmitArraySubscriptExpr()
  llvm::Value *Idx = Builder.CreateSExt(Index, IntPtrTy, "Xall.linearidx"); //forall or renderall

  // get the correct element of the field depending on the index
  sprintf(IRNameStr, "%s.%s.element.ptr", MeshName.str().c_str(),FieldName.str().c_str());
  addr = Builder.CreateInBoundsGEP(addr, Idx, IRNameStr);

  LValue LV = MakeAddrLValue(addr, type, alignment);

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
  llvm::Value* Index;

  if(MFD->isVertexLocated()) {
    assert(VertexIndex && "null VertexIndex while referencing vertex field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(VertexIndex);
  } else if(MFD->isEdgeLocated()) {
    assert(EdgeIndex && "null EdgeIndex while referencing edge field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(EdgeIndex);
  } else if(MFD->isFaceLocated()) {
    assert(FaceIndex && "null FaceIndex while referencing face field");
    // use the vertex index if we are within a forall vertices
    Index = Builder.CreateLoad(FaceIndex);
  } else if(MFD->isCellLocated() && CellIndex) {
    Index = Builder.CreateLoad(CellIndex);
  } else {
    Index = getLinearIdx();
  }
  return Index;
}

// compute the linear index based on cshift parameters
// with circular boundary conditions
llvm::Value *
CodeGenFunction::getCShiftLinearIdx(SmallVector< llvm::Value *, 3 > args) {

  //get the dimensions (Width, Height, Depth)
  SmallVector< llvm::Value *, 3 > dims;
  for(unsigned i = 0; i < args.size(); ++i) {
    sprintf(IRNameStr, "%s", DimNames[i]);
    dims.push_back(Builder.CreateLoad(LookupMeshDim(i), IRNameStr));
  }

  SmallVector< llvm::Value *, 3 > indices;
  for(unsigned i = 0; i < args.size(); ++i) {
    sprintf(IRNameStr, "forall.induct.%s", IndexNames[i]);
    llvm::Value *iv   = Builder.CreateLoad(LookupInductionVar(i), IRNameStr);

    // take index and add offset from cshift
    sprintf(IRNameStr, "cshift.rawindex.%s", IndexNames[i]);
    llvm::Value *rawIndex = Builder.CreateAdd(iv, args[i], IRNameStr);

    // make sure it is in range or wrap
    sprintf(IRNameStr, "cshift.index.%s", IndexNames[i]);
    indices.push_back(Builder.CreateURem(rawIndex, dims[i], IRNameStr));
  }

  switch(args.size()) {
    case 1:
      return indices[0];
    case 2: {
      // linearIdx = x + Height * y;
      llvm::Value *Hy    = Builder.CreateMul(dims[1], indices[1], "HeightxY");
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
      args.push_back(RV.getAggregateAddr());
    } else {
      args.push_back(RV.getScalarVal());
    }
  }

  // get the member expr for first arg.
  if(const MemberExpr *E = dyn_cast<MemberExpr>(A1E)) {
    // make sure this is a mesh
    if(isa<MeshFieldDecl>(E->getMemberDecl())) {
      // get the correct mesh member
      LValue LV = EmitMeshMemberExpr(E, getCShiftLinearIdx(args));

      return RValue::get(Builder.CreateLoad(LV.getAddress(), "cshift.element"));
    }
  }
  assert(false && "Failed to translate Scout cshift expression to LLVM IR!");
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
      args.push_back(RV.getAggregateAddr());
    } else {
      args.push_back(RV.getScalarVal());
    }
  }

  // get the member expr for first arg.
  if(const MemberExpr *E = dyn_cast<MemberExpr>(A1E)) {
    // make sure this is a mesh
    if(isa<MeshFieldDecl>(E->getMemberDecl())) {

      //get the dimensions (Width, Height, Depth)
       SmallVector< llvm::Value *, 3 > dims;
       for(unsigned i = 0; i < args.size(); ++i) {
         sprintf(IRNameStr, "%s", DimNames[i]);
         dims.push_back(Builder.CreateLoad(LookupMeshDim(i), IRNameStr));
       }

       //get the eoshift indices
       SmallVector< llvm::Value *, 3 > rawindices, indices;
       for(unsigned i = 0; i < args.size(); ++i) {
				 sprintf(IRNameStr, "forall.induct.%s", IndexNames[i]);
				 llvm::Value *iv  = Builder.CreateLoad(LookupInductionVar(i), IRNameStr);

				 // take index and add offset from eoshift
				 sprintf(IRNameStr, "eoshift.rawindex.%s", IndexNames[i]);
				 rawindices.push_back(Builder.CreateAdd(iv, args[i], IRNameStr));

				 // module to find if index will wrap
				 sprintf(IRNameStr, "eoshift.index.%s", IndexNames[i]);
				 indices.push_back(Builder.CreateURem(rawindices[i], dims[i], IRNameStr));
       }

       // setup flag
       llvm::Value *flag;
       flag = Builder.CreateAlloca(Int32Ty, 0, "flag");
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
					 // linearIdx = x + Height * y;
					 llvm::Value *Hy = Builder.CreateMul(dims[1], indices[1], "HeightxY");
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
      llvm::Value *BaseAddr;
      GetMeshBaseAddr(VD, BaseAddr);
      llvm::StringRef MeshName = BaseAddr->getName();


      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), names[offset]);
      value = Builder.CreateConstInBoundsGEP2_32(BaseAddr, 0, nfields+offset, IRNameStr);

      sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), names[offset]);
      value = Builder.CreateLoad(value, IRNameStr);
      return RValue::get(value);
    }
  }
  //sema should make sure we don't get here.
  assert(false && "Failed to emit Mesh Parameter");
}

void CodeGenFunction::EmitQueryExpr(const ValueDecl* VD,
                                    LValue LV,
                                    const QueryExpr* QE){
  using namespace std;
  using namespace llvm;
  
  typedef vector<llvm::Value*> ValueVec;
  typedef vector<llvm::Type*> TypeVec;

  CGBuilderTy& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();
  LLVMContext& C = getLLVMContext();
  
  Value* One = ConstantInt::get(Int64Ty, 1);
  
  BasicBlock* prevBlock = B.GetInsertBlock();
  BasicBlock::iterator prevPoint = B.GetInsertPoint();
  
  const MemberExpr* memberExpr = QE->getField();
  const Expr* pred = QE->getPredicate();
  
  const DeclRefExpr* base = dyn_cast<DeclRefExpr>(memberExpr->getBase());
  assert(base && "expected a DeclRefExpr");

  const ImplicitMeshParamDecl* imp = dyn_cast<ImplicitMeshParamDecl>(base->getDecl());
  assert(base && "expected an ImplicitMeshParamDecl");

  ImplicitMeshParamDecl::MeshElementType et = imp->getElementType();
  
  const VarDecl* mvd = imp->getMeshVarDecl();
  
  TypeVec params =
  {R.PointerTy(ConvertType(mvd->getType())), R.PointerTy(Int8Ty), Int64Ty, Int64Ty};

  llvm::FunctionType* ft = llvm::FunctionType::get(R.VoidTy, params, false);
  
  Function* queryFunc = Function::Create(ft,
                                         Function::ExternalLinkage,
                                         "MeshQueryFunction",
                                         &CGM.getModule());
  
  auto aitr = queryFunc->arg_begin();

  Value* meshPtr = aitr++;
  meshPtr->setName("mesh");

  Value* outPtr = aitr++;
  outPtr->setName("outMask");
  
  Value* start = aitr++;
  start->setName("start");
  
  Value* end = aitr++;
  end->setName("end");
  
  Value* baseAddr = LocalDeclMap[mvd];
  
  LocalDeclMap[mvd] = meshPtr;
  
  BasicBlock* entry = BasicBlock::Create(C, "entry", queryFunc);
  B.SetInsertPoint(entry);
  
  Value* inductPtr = B.CreateAlloca(Int64Ty, 0, "induct.ptr");

  B.CreateStore(start, inductPtr);
  
  BasicBlock* loopBlock = BasicBlock::Create(C, "query.loop", queryFunc);
  B.CreateBr(loopBlock);
  B.SetInsertPoint(loopBlock);

  switch(et){
    case ImplicitMeshParamDecl::Cells:
      CellIndex = inductPtr;
      break;
    case ImplicitMeshParamDecl::Vertices:
      VertexIndex = inductPtr;
      break;
    case ImplicitMeshParamDecl::Edges:
      EdgeIndex = inductPtr;
      break;
    case ImplicitMeshParamDecl::Faces:
      FaceIndex = inductPtr;
      break;
    default:
      assert(false && "invalid mesh element type");
  }

  Value* result =
  B.CreateTrunc(EmitAnyExprToTemp(pred).getScalarVal(), Int8Ty, "result");
  
  switch(et){
    case ImplicitMeshParamDecl::Cells:
      CellIndex = 0;
      break;
    case ImplicitMeshParamDecl::Vertices:
      VertexIndex = 0;
      break;
    case ImplicitMeshParamDecl::Edges:
      EdgeIndex = 0;
      break;
    case ImplicitMeshParamDecl::Faces:
      FaceIndex = 0;
      break;
    default:
      break;
  }
  
  Value* induct = B.CreateLoad(inductPtr, "induct");

  Value* outPosPtr = B.CreateGEP(outPtr, induct, "outPos.ptr");
  B.CreateStore(result, outPosPtr);

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
  
  LocalDeclMap[mvd] = baseAddr;
  
  Value* qp = LV.getAddress();

  Value* funcField = B.CreateStructGEP(qp, 0, "query.func.ptr");
  Value* meshPtrField = B.CreateStructGEP(qp, 1, "query.mesh.ptr");
  
  B.CreateStore(B.CreateBitCast(queryFunc, CGM.VoidPtrTy), funcField);
  B.CreateStore(B.CreateBitCast(baseAddr, CGM.VoidPtrTy), meshPtrField);
  
  LocalDeclMap[VD] = qp;
}
