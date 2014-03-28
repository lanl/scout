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

void CodeGenFunction::EmitScoutAutoVarAlloca(llvm::AllocaInst *Alloc,
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

    llvm::Value* numCells = 0;
    llvm::Value* numVertices = 0;
    llvm::Value* numFaces = 0;
    llvm::Value* numEdges = 0;
    
    for(MeshDecl::field_iterator itr = MD->field_begin(),
        itr_end = MD->field_end(); itr != itr_end; ++itr){
      if(itr->isCellLocated()){
        numCells = numCells ? numCells : Builder.getInt64(1);
      }
      else if(itr->isVertexLocated()){
        numVertices = numVertices ? numVertices : Builder.getInt64(1);
      }
      else if(itr->isFaceLocated()){
        numFaces = numFaces ? numFaces : Builder.getInt64(1);
      }
      else if(itr->isEdgeLocated()){
        numEdges = numEdges ? numEdges : Builder.getInt64(1);
      }
    }
    
    // Maybe dimensions needs to hold values???

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
      // SC_TODO: check the evalret
      if(numCells){
        numCells = Builder.CreateMul(intValue, numCells);
      }
      
      if(numVertices){
        llvm::Value* intValue2 = Builder.CreateAdd(intValue, Builder.getInt64(1));
        numVertices = Builder.CreateMul(intValue2, numVertices);
      }

      // SC_TODO: finish faces and edges - the following is not correct
      if(numFaces){
        numFaces = Builder.CreateMul(intValue, numFaces);
      }
      
      if(numEdges){
        numEdges = Builder.CreateMul(intValue, numEdges);
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
      else if(FD->isFaceLocated()){
        numElements = numFaces;
      }
      else if(FD->isEdgeLocated()){
        numElements = numEdges;
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



