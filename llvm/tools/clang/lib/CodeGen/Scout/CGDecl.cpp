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
#include "clang/AST/Type.h"
#include <stdio.h>
#include <cassert>

using namespace std;
using namespace clang;
using namespace CodeGen;

static const char *DimNames[]   = { "width", "height", "depth" };

// We use 'IRNameStr' to hold the generated names we use for
// various values in the IR building.  We've added a static
// buffer to avoid the need for a lot of fine-grained new and
// delete calls...  We're likely safe with 160 character long
// strings.
static char IRNameStr[160];

//if global mesh is not setup then set it up
void CodeGenFunction::EmitGlobalMeshAllocaIfMissing(llvm::Value* MeshAddr, const VarDecl &D) {

  const Type *Ty = D.getType().getTypePtr();

  // See if type is already setup
  llvm::DenseMap<const Type *, bool>::iterator TCI = getTypes().GlobalMeshInit.find(Ty);
  if (TCI != getTypes().GlobalMeshInit.end())
    return;

  // flag as setup
  getTypes().GlobalMeshInit[Ty] = true;

  const MeshType* MT = cast<MeshType>(Ty);
  llvm::StringRef MeshName  = MT->getName();
  MeshDecl* MD = MT->getDecl();
  unsigned int nfields = MD->fields();

  // If the rank has not been set it is ok to do the alloc and other setup
  // this is for the multifile case to make sure we don't double alloc.

  // get function
  llvm::Function *TheFunction;
  TheFunction = Builder.GetInsertBlock()->getParent();

  llvm::BasicBlock *Then = createBasicBlock("global.alloc");
  llvm::BasicBlock *Done = createBasicBlock("global.done");

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);

  // test if rank is not set.
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  llvm::Value *Rank = Builder.CreateConstInBoundsGEP2_32(MeshAddr, 0, nfields+3, IRNameStr);
  sprintf(IRNameStr, "%s.rank", MeshName.str().c_str());
  llvm::Value *Check = Builder.CreateICmpEQ(Builder.CreateLoad(Rank, IRNameStr), ConstantZero);
  Builder.CreateCondBr(Check, Then, Done);

  //then block (do setup)
  TheFunction->getBasicBlockList().push_back(Then);
  Builder.SetInsertPoint(Then);
  EmitScoutAutoVarAlloca(MeshAddr, D);
  Builder.CreateBr(Done);
  Then = Builder.GetInsertBlock();

  // done block
  TheFunction->getBasicBlockList().push_back(Done);
  Builder.SetInsertPoint(Done);
  Done = Builder.GetInsertBlock();
  return;
}

void CodeGenFunction::EmitMeshParameters(llvm::Value* MeshAddr, const VarDecl &D) {

  QualType T = D.getType();
  const MeshType* MT = cast<MeshType>(T.getTypePtr());
  llvm::StringRef MeshName  = MT->getName();
  MeshDecl* MD = MT->getDecl();
  unsigned int nfields = MD->fields();

  MeshType::MeshDimensions dims;
  dims = cast<MeshType>(T.getTypePtr())->dimensions();
  unsigned int rank = dims.size();

  for(size_t i = 0; i < rank; ++i) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    llvm::Value *field = Builder.CreateConstInBoundsGEP2_32(MeshAddr, 0, nfields+i, IRNameStr);
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

    Builder.CreateStore(intValue, field);
  }
  // set unused dimensions to size 1 this makes the codegen for forall/renderall easier
  for(size_t i = rank; i< 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    llvm::Value *field = Builder.CreateConstInBoundsGEP2_32(MeshAddr, 0, nfields+i, IRNameStr);
    llvm::Value* ConstantOne =  llvm::ConstantInt::get(Int32Ty, 1);
    Builder.CreateStore(ConstantOne, field);
  }
  //set rank this makes Codegen easier for rank() builtin
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  llvm::Value *Rank = Builder.CreateConstInBoundsGEP2_32(MeshAddr, 0, nfields+3, IRNameStr);
  Builder.CreateStore(llvm::ConstantInt::get(Int32Ty, rank), Rank);
}

void CodeGenFunction::EmitScoutAutoVarAlloca(llvm::Value *Alloc,
                                             const VarDecl &D) {
  QualType T = D.getType();
  const clang::Type &Ty = *getContext().getCanonicalType(T).getTypePtr();

  // SC_TODO - we need to handle the other mesh types here...
  //
  if (Ty.getTypeClass() == Type::UniformMesh) {
    // For mesh types each mesh field becomes a pointer to the allocated
    // field.
    //
    // SC_TODO - We need to make sure we can support variable-sized
    //           allocations (vs. static).
    //
    // SC_TODO - We are only supporting one mesh type here...
    //
    const MeshType* MT = cast<MeshType>(T.getTypePtr());
    llvm::StringRef MeshName  = MT->getName();
    MeshDecl* MD = MT->getDecl();

    MeshType::MeshDimensions dims;
    dims = cast<MeshType>(T.getTypePtr())->dimensions();
    unsigned int rank = dims.size();

    SmallVector<llvm::Value*, 3> Dimensions;

    // Need to make this different for variable dims as
    // we want to evaluate each dim and if its a variable
    // then we want to make an expression multiplying
    // the dims to get numElts as a variable.
    for(unsigned i = 0; i < rank; ++i) {
      llvm::Value* intValue;
      Expr* E = dims[i];

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

      intValue = Builder.CreateZExt(intValue, Int64Ty);

      Dimensions.push_back(intValue);
    }

    bool hasCells = false;
    bool hasVertices = false;
    bool hasEdges = false;
    bool hasFaces = false;

    for(MeshDecl::field_iterator itr = MD->field_begin(),
        itr_end = MD->field_end(); itr != itr_end; ++itr){
      if(itr->isCellLocated()){
        hasCells = true;
      }
      else if(itr->isVertexLocated()){
        hasVertices = true;
      }
      else if(itr->isEdgeLocated()){
        hasEdges = true;
      }
      else if(itr->isFaceLocated()){
        hasFaces = true;
      }
    }

    llvm::Value* numCells = 0;
    llvm::Value* numVertices = 0;
    llvm::Value* numEdges = 0;
    llvm::Value* numFaces = 0;

    llvm::Value* One = Builder.getInt64(1);

    llvm::Value* w1 = 0;
    llvm::Value* h1 = 0;
    llvm::Value* d1 = 0;

    if(hasCells){
      switch(rank){
      case 1:
        numCells = Dimensions[0];
        break;
      case 2:
      case 3:
        llvm::Value* wh = Builder.CreateMul(Dimensions[0], Dimensions[1]);
        if(rank == 2){
          numCells = wh;
          break;
        }
        numCells = Builder.CreateMul(wh, Dimensions[2]);
        break;
      }
    }

    if(hasVertices){
      w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
      if(rank > 1){
        h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
        llvm::Value* wh1 = Builder.CreateMul(w1, h1);
        if(rank > 2){
          d1 = d1 ? d1 : Builder.CreateAdd(Dimensions[2], One);
          numVertices = Builder.CreateMul(wh1, d1);
        }
        else{
          numVertices = wh1;
        }
      }
      else{
        numVertices = w1;
      }
    }

    if(hasEdges){
      switch(rank){
      case 1:
        numEdges = Dimensions[0];
        break;
      case 2:
      case 3:{
        w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
        h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
        llvm::Value* v3 = Builder.CreateMul(w1, Dimensions[1]);
        llvm::Value* v4 = Builder.CreateMul(h1, Dimensions[0]);
        llvm::Value* v5 = Builder.CreateAdd(v3, v4);

        if(rank == 2){
          numEdges = v5;
          break;
        }

        d1 = d1 ? d1 : Builder.CreateAdd(Dimensions[2], One);
        llvm::Value* v7 = Builder.CreateMul(v5, d1);
        llvm::Value* v8 =
            Builder.CreateMul(Builder.CreateMul(w1, h1), Dimensions[2]);
        numEdges = Builder.CreateAdd(v7, v8);
        break;
      }
      }
    }

    if(hasFaces){
      switch(rank){
      case 1:
        numFaces = Dimensions[0];
        break;
      case 2:{
        w1 = w1 ? w1 : Builder.CreateAdd(Dimensions[0], One);
        h1 = h1 ? h1 : Builder.CreateAdd(Dimensions[1], One);
        llvm::Value* v3 = Builder.CreateMul(w1, Dimensions[1]);
        llvm::Value* v4 = Builder.CreateMul(h1, Dimensions[0]);
        numFaces = Builder.CreateAdd(v3, v4);
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

        numFaces = Builder.CreateAdd(v1, Builder.CreateAdd(v2, v3));
        break;
      }
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

    llvm::Type *structTy = Alloc->getType()->getContainedType(0);

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

      if(FD->isCellLocated()){
        numElements = numCells;
      }
      else if(FD->isVertexLocated()){
        numElements = numVertices;
      }
      else if(FD->isEdgeLocated()){
        numElements = numEdges;
      }
      else if(FD->isFaceLocated()){
        numElements = numFaces;
      }

      assert(numElements && "invalid numElements");

      fieldTotalBytes = Builder.CreateNUWMul(numElements, fieldTyBytesValue);

      // Dynamically allocate memory.
      llvm::Value *val = CreateMemAllocForValue(fieldTotalBytes);
      val = Builder.CreateBitCast(val, structTy->getContainedType(i));

      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), MeshFieldName.str().c_str());
      llvm::Value *field = Builder.CreateConstInBoundsGEP2_32(Alloc, 0, i, IRNameStr);
      Builder.CreateStore(val, field);
    }

    // mesh dimensions after the fields
    // this is setup in Codegentypes.cpp ConvertScoutMeshType()
    EmitMeshParameters(Alloc, D);
  }
}
