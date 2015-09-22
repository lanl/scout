/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ###########################################################################
 *
 * Notes
 *
 * #####
 */

#include "CodeGenFunction.h"
#include "CGDebugInfo.h"
#include "CGOpenCLRuntime.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "clang/AST/Type.h"
#include "Scout/CGPlotRuntime.h"
#include "Scout/CGScoutRuntime.h"
#include "Scout/CGLegionCRuntime.h"
#include <stdio.h>
#include <cassert>

using namespace std;
using namespace clang;
using namespace CodeGen;

static const char *DimNames[]   = { "width", "height", "depth" };
static const char *StartNames[]   = { "xstart", "ystart", "zstart" };
static const char *SizeNames[]   = { "xsize", "ysize", "zsize" };

// We use 'IRNameStr' to hold the generated names we use for
// various values in the IR building.  We've added a static
// buffer to avoid the need for a lot of fine-grained new and
// delete calls...  We're likely safe with 160 character long
// strings.
static char IRNameStr[160];

//if global mesh is not setup then set it up
//SC_TODO MeshAddr should be Address
void CodeGenFunction::EmitGlobalMeshAllocaIfMissing(llvm::Value* MeshAddr, const VarDecl &D) {

  const Type *Ty = D.getType().getTypePtr();

  // See if type is already setup
  llvm::DenseMap<const Type *, bool>::iterator TCI = getTypes().GlobalMeshInit.find(Ty);
  if (TCI != getTypes().GlobalMeshInit.end())
    return;

  // flag as setup
  getTypes().GlobalMeshInit[Ty] = true;

  const MeshType* MT = cast<MeshType>(Ty);
  llvm::StringRef MeshName  = MeshAddr->getName();
  MeshDecl* MD = MT->getDecl();
  unsigned int nfields = MD->fields();

  // If the rank has not been set it is ok to do the alloc and other setup
  // this is for the multifile case to make sure we don't double alloc.

  llvm::BasicBlock *Then = createBasicBlock("global.alloc");
  llvm::BasicBlock *Done = createBasicBlock("global.done");

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);

  // test if rank is not set.
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  Address Rank =
  Builder.CreateMeshGEP(Address(MeshAddr, getPointerAlign()), 0,
                                nfields + MeshParameterOffset::RankOffset, IRNameStr);
  sprintf(IRNameStr, "%s.rank", MeshName.str().c_str());
  llvm::Value *Check =
  Builder.CreateICmpEQ(Builder.CreateLoad(Rank, IRNameStr), ConstantZero);
  Builder.CreateCondBr(Check, Then, Done);

  //then block (do setup)
  EmitBlock(Then);
  EmitScoutAutoVarAlloca(MeshAddr, D);
  Builder.CreateBr(Done);

  // done block
  EmitBlock(Done);
  return;
}

void CodeGenFunction::EmitMeshParameters(llvm::Value* MeshAddr, const VarDecl &D) {

  QualType T = D.getType();
  const MeshType* MT = cast<MeshType>(T.getTypePtr());
  llvm::StringRef MeshName  = MeshAddr->getName();
  MeshDecl* MD = MT->getDecl();

  MeshType::MeshDimensions dims;
  dims = cast<MeshType>(T.getTypePtr())->dimensions();
  unsigned int rank = dims.size();

  unsigned int nfields = MD->fields();
  size_t start = nfields +  MeshParameterOffset::WidthOffset;
  size_t sizestart = nfields + MeshParameterOffset::XSizeOffset;

  for(size_t i = 0; i < rank; ++i) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    llvm::Value *x = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, start+i, IRNameStr);
    llvm::Value* intValue;

    Expr* E = dims[i];
    if (E->isGLValue()) {
      // Emit the expression as an lvalue.
      LValue LV = EmitLValue(E);
      // We have to load the lvalue.
      RValue RV = EmitLoadOfLValue(LV, E->getExprLoc());
      intValue = RV.getScalarVal();
    } else if (E->isConstantInitializer(getContext(), false)) {
      bool evalret;
      llvm::APSInt dimAPValue;
      evalret = E->EvaluateAsInt(dimAPValue, getContext());
      // SC_TODO: check the evalret
      (void)evalret; //suppress warning

      intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
    } else {
      // it is an Rvalue
      RValue RV = EmitAnyExpr(E);
      intValue = RV.getScalarVal();
    }

    llvm::Value* extIntValue = Builder.CreateZExt(intValue, Int64Ty);
    Builder.CreateStore(extIntValue, scoutPtr(x));

    // also store width/height/depth in sizes
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), SizeNames[i]);
    llvm::Value *size = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, sizestart+i, IRNameStr);
    Builder.CreateStore(extIntValue, scoutPtr(size));

  }
  // set unused dimensions/sizes to 0
  llvm::Value* ConstantZero =  llvm::ConstantInt::get(Int64Ty, 0);

  for(size_t i = rank; i< 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    llvm::Value *x = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, start+i, IRNameStr);
    Builder.CreateStore(ConstantZero, scoutPtr(x));

    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), SizeNames[i]);
    llvm::Value *size = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, sizestart+i, IRNameStr);
    Builder.CreateStore(ConstantZero, scoutPtr(size));
  }

  //set rank this makes Codegen easier for rank() builtin
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  llvm::Value *Rank = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, nfields +MeshParameterOffset::RankOffset, IRNameStr);
  Builder.CreateStore(llvm::ConstantInt::get(Int64Ty, rank), scoutPtr(Rank));

  // set xstart/ystart/zstart to 0
  size_t xstart = nfields + MeshParameterOffset::XStartOffset;
  for(size_t i = 0; i< 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), StartNames[i]);
    llvm::Value *x = Builder.CreateConstInBoundsGEP2_32(0, MeshAddr, 0, xstart+i, IRNameStr);
    llvm::Value* ConstantZero =  llvm::ConstantInt::get(Int64Ty, 0);
    Builder.CreateStore(ConstantZero, scoutPtr(x));
  }
}

// SC_TODO: this should not be a member function, as it calls dims.size()
// and should only be used in this file.
void CodeGenFunction::GetMeshDimensions(const MeshType* MT,
                                        SmallVector<llvm::Value*, 3>& DS){
  MeshType::MeshDimensions dims = MT->dimensions();
  unsigned int rank = dims.size();

  // Need to make this different for variable dims as
  // we want to evaluate each dim and if its a variable
  // then we want to make an expression multiplying
  // the dims to get numElts as a variable.
  for(unsigned i = 0; i < rank; ++i) {
    llvm::Value* intValue;
    Expr* E = dims[i];

    if (E->isConstantInitializer(getContext(), false)) {
      llvm::APSInt dimAPValue;
      bool success = E->EvaluateAsInt(dimAPValue, getContext());
      assert(success && "Failed to evaluate mesh dimension");

      intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
    } else if (E->isGLValue()) {
      // Emit the expression as an lvalue.
      LValue LV = EmitLValue(E);

      // We have to load the lvalue.
      RValue RV = EmitLoadOfLValue(LV, E->getExprLoc() );
      intValue  = RV.getScalarVal();
    } else {
      // it is an Rvalue
      RValue RV = EmitAnyExpr(E);
      intValue = RV.getScalarVal();
    }

    intValue = Builder.CreateZExt(intValue, Int64Ty);

    DS.push_back(intValue);
  }
}

//SC_TODO: this should go away completely as it uses rank.
void
CodeGenFunction::GetNumMeshItems(SmallVector<llvm::Value*, 3>& Dimensions,
                                 llvm::Value** numCells,
                                 llvm::Value** numVertices,
                                 llvm::Value** numEdges,
                                 llvm::Value** numFaces){
  size_t rank = Dimensions.size();

  llvm::Value* One = Builder.getInt64(1);

  llvm::Value* w1 = 0;
  llvm::Value* h1 = 0;
  llvm::Value* d1 = 0;

  if(numCells){
    switch(rank){
    case 1:
      *numCells = Dimensions[0];
      break;
    case 2:
    case 3:
      llvm::Value* wh = Builder.CreateMul(Dimensions[0], Dimensions[1]);
      if(rank == 2){
        *numCells = wh;
        break;
      }
      *numCells = Builder.CreateMul(wh, Dimensions[2]);
      break;
    }
  }

  if(numVertices){
    w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
    if(rank > 1){
      h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
      llvm::Value* wh1 = Builder.CreateMul(w1, h1);
      if(rank > 2){
        d1 = d1 ? d1 : Builder.CreateAdd(Dimensions[2], One);
        *numVertices = Builder.CreateMul(wh1, d1);
      }
      else{
        *numVertices = wh1;
      }
    }
    else{
      *numVertices = w1;
    }
  }

  if(numEdges){
    switch(rank){
    case 1:
      *numEdges = Dimensions[0];
      break;
    case 2:
    case 3:{
      w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
      h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
      llvm::Value* v3 = Builder.CreateMul(w1, Dimensions[1]);
      llvm::Value* v4 = Builder.CreateMul(h1, Dimensions[0]);
      llvm::Value* v5 = Builder.CreateAdd(v3, v4);

      if(rank == 2){
        *numEdges = v5;
        break;
      }

      d1 = d1 ? d1 : Builder.CreateAdd(Dimensions[2], One);
      llvm::Value* v7 = Builder.CreateMul(v5, d1);
      llvm::Value* v8 =
          Builder.CreateMul(Builder.CreateMul(w1, h1), Dimensions[2]);
      *numEdges = Builder.CreateAdd(v7, v8);
      break;
    }
    }
  }

  if(numFaces){
    switch(rank){
    case 1:
      *numFaces = Dimensions[0];
      break;
    case 2:{
      w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
      h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
      llvm::Value* v3 = Builder.CreateMul(w1, Dimensions[1]);
      llvm::Value* v4 = Builder.CreateMul(h1, Dimensions[0]);
      *numFaces = Builder.CreateAdd(v3, v4);
      break;
    }
    case 3:{
      w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
      h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
      d1 = d1 ? d1 : Builder.CreateAdd(Dimensions[2], One);

      llvm::Value* v1 =
          Builder.CreateMul(w1, Builder.CreateMul(Dimensions[1],
                                                  Dimensions[2]));
      llvm::Value* v2 =
          Builder.CreateMul(h1, Builder.CreateMul(Dimensions[0],
                                                  Dimensions[2]));
      llvm::Value* v3 =
          Builder.CreateMul(d1, Builder.CreateMul(Dimensions[0],
                                                  Dimensions[1]));

      *numFaces = Builder.CreateAdd(v1, Builder.CreateAdd(v2, v3));
      break;
    }
    }
  }
}

void CodeGenFunction::EmitScoutAutoVarAlloca(llvm::Value *Alloc,
                                             const VarDecl &D) {
  QualType T = D.getType();
  const clang::Type &Ty = *getContext().getCanonicalType(T).getTypePtr();

  using ValueVec = vector<llvm::Value*>;
  
  auto R = CGM.getScoutRuntime();
  
  // SC_TODO - we need to handle the other mesh types here...
  //
  if ((Ty.getTypeClass() == Type::UniformMesh) || (Ty.getTypeClass() == Type::ALEMesh)) {
    // For mesh types each mesh field becomes a pointer to the allocated
    // field.
    //
    // SC_TODO - We need to make sure we can support variable-sized
    //           allocations (vs. static).
    //
    // SC_TODO - We are only supporting one mesh type here...
    //

    const MeshType* MT = cast<MeshType>(T.getTypePtr());
    llvm::StringRef MeshName  = Alloc->getName();
    MeshDecl* MD = MT->getDecl();

    SmallVector<llvm::Value*, 3> Dimensions;
    GetMeshDimensions(MT, Dimensions);

    ValueVec args;
    
    llvm::Value* meshTopology;
    
    switch(Dimensions.size()){
      case 1:
        args = {Dimensions[0]};
        meshTopology =
        Builder.CreateCall(R.CreateUniformMesh1dFunc(), args, "topology");
        break;
      case 2:
        args = {Dimensions[0], Dimensions[1]};
        meshTopology =
        Builder.CreateCall(R.CreateUniformMesh2dFunc(), args, "topology");
        break;
      case 3:
        args = {Dimensions[0], Dimensions[1], Dimensions[2]};
        meshTopology =
        Builder.CreateCall(R.CreateUniformMesh3dFunc(), args, "topology");
        break;
      default:
        assert(false && "invalid mesh dimensions");
    }
    
    llvm::Value* Unimesh = 0;

    if(CGM.getCodeGenOpts().ScoutLegionSupport) {
      /*
      args = {meshTopology};
      Builder.CreateCall(R.MeshTopologyComputeAllFunc(), args);
      */
       
      llvm::SmallVector< llvm::Value *, 6 > Args;

      llvm::Value* runtime =
      Builder.CreateLoad(scoutPtr(CGM.getLegionCRuntime().GetLegionRuntimeGlobal()));

      Args.push_back(runtime);
      
      llvm::Value* context =
      Builder.CreateLoad(scoutPtr(CGM.getLegionCRuntime().GetLegionContextGlobal()));
      
      Args.push_back(context);

      Args.push_back(llvm::ConstantInt::get(Int64Ty, Dimensions.size()));
      
      for(unsigned int i = 0; i < Dimensions.size(); i++) {
        Args.push_back(Builder.CreateZExt(Dimensions[i], Int64Ty));
      }
      
      // if doesn't have 3 dims, add 0's for the dims
      for(unsigned int i = Dimensions.size(); i < 3; i++) {
        Args.push_back(Builder.getInt64(0));
      }
      
      Args.push_back(meshTopology);

      llvm::Function *F = CGM.getLegionCRuntime().ScUniformMeshCreateFunc();

      Unimesh = Builder.CreateCall(F, ArrayRef<llvm::Value *>(Args), "mesh");
      
      // Create metadata
      llvm::NamedMDNode *MeshMD;
      MeshMD = CGM.getModule().getOrInsertNamedMetadata("scout.legion.meshmd");
      SmallVector<llvm::Metadata*, 2> MeshInfoMD;
      
      llvm::MDString *MDMeshName = llvm::MDString::get(getLLVMContext(), MeshName);
      MeshInfoMD.push_back(MDMeshName);
      
      llvm::MDString *MDLegionMeshName = llvm::MDString::get(getLLVMContext(), Unimesh->getName());
      MeshInfoMD.push_back(MDLegionMeshName);
      
      MeshMD->addOperand(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Metadata*>(MeshInfoMD)));
    }

    llvm::Value* numCells = nullptr;
    llvm::Value* numVertices = nullptr;
    llvm::Value* numEdges = nullptr;
    llvm::Value* numFaces = nullptr;
    
    for(MeshDecl::field_iterator itr = MD->field_begin(),
        itr_end = MD->field_end(); itr != itr_end; ++itr){
      if(!numCells && itr->isCellLocated()){
        args =
        {meshTopology, llvm::ConstantInt::get(Int32Ty, Dimensions.size())};
        numCells = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      }
      else if(!numVertices && itr->isVertexLocated()) {
        args =
        {meshTopology, llvm::ConstantInt::get(Int32Ty, 0)};
        numVertices = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      }
      else if(!numEdges && itr->isEdgeLocated()){
        args =
        {meshTopology, llvm::ConstantInt::get(Int32Ty, 1)};
        numEdges = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      }
      else if(!numFaces && itr->isFaceLocated()){
        args =
        {meshTopology, llvm::ConstantInt::get(Int32Ty, Dimensions.size() - 1)};
        numFaces = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
      }
    }

    // need access to these field decls so we
    // can determine if we will dynamically allocate
    // memory for each field
    // fields are first and then mesh dimensions
    // this is setup in Codegentypes.h ConvertScoutMeshType()
    MeshDecl::field_iterator itr = MD->field_begin();
    MeshDecl::field_iterator itr_end = MD->field_end();
    unsigned int nfields = MD->fields();

    llvm::StructType *structTy = cast<llvm::StructType>(Alloc->getType()->getContainedType(0));
    
    for(unsigned i = 0; i < nfields; ++i) {

      // Compute size of needed field memory in bytes
      llvm::Type *fieldTy = structTy->getContainedType(i);
      llvm::PointerType* ptrTy = dyn_cast<llvm::PointerType>(fieldTy);
      assert(ptrTy && "Expected a pointer");
      fieldTy = ptrTy->getElementType();

      // If this is a externally allocated field, go on
      MeshFieldDecl* FD = *itr;

      if (itr != itr_end)
        ++itr;

      // SC_TODO - does this introduce a bug?  Fix me???  -PM
      if (FD->hasExternalFormalLinkage())
        continue;

      llvm::StringRef MeshFieldName = FD->getName();
      uint64_t fieldTyBytes;
      fieldTyBytes = CGM.getDataLayout().getTypeAllocSize(fieldTy);

      llvm::Value *fieldTotalBytes = 0;
      llvm::Value *fieldTyBytesValue = Builder.getInt64(fieldTyBytes);

      llvm::Value* numElements = 0;

      if (FD->isCellLocated()) {
        numElements = numCells;
      } else if(FD->isVertexLocated()) {
        numElements = numVertices;
      } else if(FD->isEdgeLocated()) {
        numElements = numEdges;
      } else if(FD->isFaceLocated()) {
        numElements = numFaces;
      }

      assert(numElements && "invalid numElements");

      fieldTotalBytes = Builder.CreateNUWMul(numElements, fieldTyBytesValue);

      if (!(CGM.getCodeGenOpts().ScoutLegionSupport)) {
        // Dynamically allocate memory.
        llvm::SmallVector< llvm::Value *, 3 > Args;
        Args.push_back(fieldTotalBytes);
        llvm::Function *MallocFunc = CGM.getScoutRuntime().MemAllocFunction();
        llvm::Value *val = Builder.CreateCall(MallocFunc, ArrayRef<llvm::Value *>(Args));
        val = Builder.CreateBitCast(val, structTy->getContainedType(i));

        sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), MeshFieldName.str().c_str());
        llvm::Value *field = Builder.CreateConstInBoundsGEP2_32(0, Alloc, 0, i, IRNameStr);
        Builder.CreateStore(val, scoutPtr(field));
      }

      if(CGM.getCodeGenOpts().ScoutLegionSupport) {
        llvm::SmallVector< llvm::Value *, 4 > Args;
        llvm::Function *F = CGM.getLegionCRuntime().ScUniformMeshAddFieldFunc();

        assert(Unimesh && "should have created this earlier");

        Args.push_back(Unimesh);

        Args.push_back(Builder.CreateGlobalStringPtr(MeshFieldName));
        
        llvm::Value* elementType;
        
        if (FD->isCellLocated()) {
          elementType = CGM.getLegionCRuntime().ScCellVal;
        } else if(FD->isVertexLocated()) {
          elementType = CGM.getLegionCRuntime().ScVertexVal;
        } else if(FD->isEdgeLocated()) {
          elementType = CGM.getLegionCRuntime().ScEdgeVal;
        } else if(FD->isFaceLocated()) {
          elementType = CGM.getLegionCRuntime().ScFaceVal;
        } else{
          assert(false && "invalid element type");
        }
        
        Args.push_back(elementType);
        
        llvm::PointerType* pointerType =
        cast<llvm::PointerType>(structTy->getElementType(i));
        
        llvm::Type* type = pointerType->getElementType();
        
        llvm::Value* fieldType;
        
        if(type->isIntegerTy(32)){
          fieldType = CGM.getLegionCRuntime().ScInt32Val;
        }
        else if(type->isIntegerTy(64)){
          fieldType = CGM.getLegionCRuntime().ScInt64Val;
        }
        else if(type->isFloatTy()){
          fieldType = CGM.getLegionCRuntime().ScFloatVal;
        }
        else if(type->isDoubleTy()){
          fieldType = CGM.getLegionCRuntime().ScDoubleVal;
        }
        else{
          assert(false && "invalid field type");
        }
        
        Args.push_back(fieldType);
        
        Builder.CreateCall(F, ArrayRef<llvm::Value *>(Args));
      }
    }

    // store ptr to mesh topology
    llvm::Value* mtField =
    Builder.CreateConstInBoundsGEP2_32(0, Alloc, 0, nfields);
    
    Builder.CreateStore(meshTopology, scoutPtr(mtField));
    
    if(CGM.getCodeGenOpts().ScoutLegionSupport) {
      llvm::Function *F = CGM.getLegionCRuntime().ScUniformMeshInitFunc();
    
      llvm::SmallVector< llvm::Value *, 1 > Args;
      Args.push_back(Unimesh);
      Builder.CreateCall(F, ArrayRef<llvm::Value *>(Args));
    }
    
    // mesh dimensions after the fields
    // this is setup in Codegentypes.cpp ConvertScoutMeshType()

    // Should we still do this if using Legion?  The declared mesh doesn't really get used here. 
    // The mesh really used by forall gets made later.
    EmitMeshParameters(Alloc, D);

  } else if (Ty.getTypeClass() == Type::Window) {
    //llvm::Type *voidPtrTy = llvm::PointerType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()),0);
    //llvm::Value *voidPtr  = Builder.CreateAlloca(voidPtrTy, 0, "void.ptr");

    const WindowType* windowTy = cast<WindowType>(&Ty);

    // params to __scrt_create_window()
    Expr* widthExpr = windowTy->getWidthExpr();
    Expr* heightExpr = windowTy->getHeightExpr();

    std::vector<llvm::Value*> ptr_call_params;

    llvm::Value* intValue;
    Expr* E;
    E = widthExpr;

    if (E->isGLValue()) {
      // Emit the expression as an lvalue.
      LValue LV = EmitLValue(E);

      // We have to load the lvalue.
      RValue RV = EmitLoadOfLValue(LV, E->getExprLoc() );
      intValue  = RV.getScalarVal();

    } else if (E->isConstantInitializer(getContext(), false)) {

      bool evalret;
      llvm::APSInt dimAPValue;
      evalret = E->EvaluateAsInt(dimAPValue, getContext());
      // SC_TODO: check the evalret
      (void)evalret; //supress warning

      intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
    } else {
      // it is an Rvalue
      RValue RV = EmitAnyExpr(E);
      intValue = RV.getScalarVal();
    }

    intValue = Builder.CreateTruncOrBitCast(intValue, Int16Ty);
    ptr_call_params.push_back(intValue);

    E = heightExpr;

    if (E->isGLValue()) {
      // Emit the expression as an lvalue.
      LValue LV = EmitLValue(E);

      // We have to load the lvalue.
      RValue RV = EmitLoadOfLValue(LV, E->getExprLoc() );
      intValue  = RV.getScalarVal();

    } else if (E->isConstantInitializer(getContext(), false)) {

      bool evalret;
      llvm::APSInt dimAPValue;
      evalret = E->EvaluateAsInt(dimAPValue, getContext());
      // SC_TODO: check the evalret
      (void)evalret; //supress warning

      intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
    } else {
      // it is an Rvalue
      RValue RV = EmitAnyExpr(E);
      intValue = RV.getScalarVal();
    }

    intValue = Builder.CreateTruncOrBitCast(intValue, Int16Ty);
    ptr_call_params.push_back(intValue);

    // Should do a runtime check on the window parameters to make sure within range
    
    // call __scrt_create_window()
    llvm::CallInst* ptr_call = 
      Builder.CreateCall(CGM.getScoutRuntime().CreateWindowFunction(), ptr_call_params, "");

    // make type for ptr to scout.window_t
    //llvm::StructType *StructTy_scout_window_t = CGM.getModule().getTypeByName("scout.window_t");
    
    llvm::PointerType* pt = dyn_cast<llvm::PointerType>(Alloc->getType());
    
    // cast call result to scout.window_t
    llvm::Value* ptr_cast = Builder.CreateBitCast(ptr_call, pt->getElementType(), "");
    
    // store call result into previously allocated ptr for window
    llvm::Value* void_store = Builder.CreateStore(ptr_cast, scoutPtr(Alloc), false);
    (void)void_store; // suppress warning

  }
  else if (Ty.getTypeClass() == Type::Frame) {
    using namespace std;
    using namespace llvm;
    
    auto R = CGM.getPlotRuntime();
    
    typedef vector<Value*> ValueVec;
    
    QualType T = D.getType();
    const FrameType* FT = cast<FrameType>(T.getTypePtr());
    FrameDecl* FD = FT->getDecl();
  
    ValueVec args;
    Value* fp = Builder.CreateCall(R.CreateFrameFunc(), args, "frame.ptr");
    Builder.CreateStore(fp, scoutPtr(Alloc));
    
    auto m = FD->getVarMap();
    for(auto& itr : m){
      const FrameDecl::Var& vi = itr.second;
      VarDecl* vd = vi.varDecl;
    
      llvm::Type* lt = ConvertType(QualType(vd->getType().getTypePtr(), 0));
      
      llvm::Value* fieldType;
      
      if(lt->isIntegerTy(32)){
        fieldType = R.ElementInt32Val;
      }
      else if(lt->isIntegerTy(64)){
        fieldType = R.ElementInt64Val;
      }
      else if(lt->isFloatTy()){
        fieldType = R.ElementFloatVal;
      }
      else if(lt->isDoubleTy()){
        fieldType = R.ElementDoubleVal;
      }
      else if(lt->isPointerTy()){
        if(lt->getPointerElementType()->isIntegerTy(8)){
          fieldType = R.ElementStringVal;
        }
        else{
          assert(false && "invalid field type");
        }
      }
      else{
        assert(false && "invalid field type");
      }
      
      args = {fp, Builder.getInt32(vi.varId), fieldType};
      Builder.CreateCall(R.FrameAddVarFunc(), args);
    }
  }
}
