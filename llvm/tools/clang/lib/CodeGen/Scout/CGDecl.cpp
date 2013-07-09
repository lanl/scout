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
using namespace clang;
using namespace CodeGen;

void CodeGenFunction::EmitScoutAutoVarAlloca(llvm::AllocaInst *Alloc,
                                             const VarDecl &D) {

  QualType T = D.getType();
  const clang::Type &Ty = *getContext().getCanonicalType(T).getTypePtr();
  
  // SC_TODO - we need to handle the other mesh types here...
  // 
  // SC_TODO - this conditional is redundant with the one preceding
  //           the call to this function (in EmitAutoVarAlloca()).
  //           Logic should be cleaned up...
  if (Ty.getTypeClass() == Type::UniformMesh) {
    // For mesh types each mesh field becomes a pointer to the allocated
    // field.
    //
    // SC_TODO - We need to make sure we can support variable-sized
    //           allocations (vs. static).
    //
    // SC_TODO - We are only supporting one mesh type here...
    //
    MeshType::MeshDimensions dims;
    dims = cast<MeshType>(T.getTypePtr())->dimensions();

    unsigned int rank = 0;
    for(unsigned int i = 0; i < dims.size(); ++i) {
      if (dims[i] != 0) 
        rank++;
    }
    
    // Maybe dimensions needs to hold values???

    // Need to make this different for variable dims as
    // we want to evaluate each dim and if its a variable
    // then we want to make an expression multiplying
    // the dims to get numElts as a variable.
    llvm::Value *numElements = Builder.getInt64(1);
    for(unsigned i = 0; i < rank; ++i) {
      llvm::Value* intValue;
      Expr* E = dims[i];
      
      if (E->isGLValue()) {
        // Emit the expression as an lvalue.
        LValue LV = EmitLValue(E);

        // We have to load the lvalue.
        RValue RV = EmitLoadOfLValue(LV);
        intValue  = RV.getScalarVal();

      } else if (E->isConstantInitializer(getContext(), false)) {

        bool evalret;
        llvm::APSInt dimAPValue;
        evalret = E->EvaluateAsInt(dimAPValue, getContext());
        // SC_TODO: check the evalret

        intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
      } else {
        // it is an Rvalue
        RValue RV = EmitAnyExpr(E);
        intValue = RV.getScalarVal();
      }

      intValue = Builder.CreateZExt(intValue, Int64Ty);
      // SC_TODO: check the evalret
      numElements = Builder.CreateMul(intValue, numElements);
    }

    // store the mesh dimensions
    // SC_TODO - can we save ourselves some complexity in the code gen
    // and drop generating zero value mesh dimensions in the code?
    for(size_t i = 0; i < rank; ++i) {
      
      llvm::Value *field = Builder.CreateConstInBoundsGEP2_32(Alloc, 0, i+1);      
      llvm::Value* intValue;
      
      Expr* E = dims[i];
      if (E->isGLValue()) {
        // Emit the expression as an lvalue.
        LValue LV = EmitLValue(E);
        // We have to load the lvalue.
        RValue RV = EmitLoadOfLValue(LV);
        intValue = RV.getScalarVal();
      } else if (E->isConstantInitializer(getContext(), false)) {
        bool evalret;
        llvm::APSInt dimAPValue;
        evalret = E->EvaluateAsInt(dimAPValue, getContext());
        // SC_TODO: check the evalret
        intValue = llvm::ConstantInt::get(getLLVMContext(), dimAPValue);
      } else {
        // it is an Rvalue
        RValue RV = EmitAnyExpr(E);
        intValue = RV.getScalarVal();
      }
      
      Builder.CreateStore(intValue, field);
    }

    // need access to these field decls so we
    // can determine if we will dynamically allocate
    // memory for each field
    const MeshType* MT = cast<MeshType>(T.getTypePtr());
    MeshDecl* MD = MT->getDecl();
    MeshDecl::field_iterator itr = MD->field_begin();
    MeshDecl::field_iterator itr_end = MD->field_end();

    llvm::Type *structTy = Alloc->getType()->getContainedType(0);
    for(unsigned i = rank+1, e = structTy->getNumContainedTypes(); i < e; ++i) {

      // Compute size of needed field memory in bytes
      llvm::Type *fieldTy = structTy->getContainedType(i);
      // If this is a externally allocated field, go on
      MeshFieldDecl* FD = *itr;
    
      if (itr != itr_end)
        ++itr;

      // SC_TODO - does this introduce a bug?  Fix me???  -PM 
      if (FD->hasExternalFormalLinkage())
        continue;

      uint64_t fieldTyBytes;
      fieldTyBytes = CGM.getDataLayout().getTypeAllocSize(fieldTy);

      llvm::Value *fieldTotalBytes = 0;
      llvm::Value *fieldTyBytesValue = Builder.getInt64(fieldTyBytes);
      fieldTotalBytes = Builder.CreateNUWMul(numElements, fieldTyBytesValue);

      // Dynamically allocate memory.
      llvm::Value *val = CreateMemAllocForValue(fieldTotalBytes);
      val = Builder.CreateBitCast(val, structTy->getContainedType(i));

      llvm::Value *field;
      field = Builder.CreateConstInBoundsGEP2_32(Alloc, 0, i);
      Builder.CreateStore(val, field);
    }

    // debugger support -- SC_TODO: do we want to name this to
    // more closely match the mesh name (vs. mesh.tmp)???
    llvm::Value *ScoutDeclDebugPtr = 0;
    ScoutDeclDebugPtr = CreateMemTemp(getContext().VoidPtrTy, "mesh.temp");
    llvm::Value* AllocVP = Builder.CreateBitCast(Alloc, VoidPtrTy);
    Builder.CreateStore(AllocVP, ScoutDeclDebugPtr);
  }
}
