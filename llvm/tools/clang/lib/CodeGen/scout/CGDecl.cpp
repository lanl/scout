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
    MeshType::MeshDimensionVec dims;
    dims = cast<MeshType>(T.getTypePtr())->dimensions();

    // Maybe dimensions needs to hold values???

    // Need to make this different for variable dims as
    // we want to evaluate each dim and if its a variable
    // then we want to make an expression multiplying
    // the dims to get numElts as a variable.
    llvm::Value *numElements = Builder.getInt64(1);
    for(unsigned i = 0, e = dims.size(); i < e; ++i) {
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
    for(size_t i = 0; i < 3; ++i){
      llvm::Value* field = Builder.CreateConstInBoundsGEP2_32(Alloc, 0, i+1);

      if(i >= dims.size()){
        // store a 0 in that dim if above size
        llvm::Value* intValue;
        intValue = llvm::ConstantInt::getSigned(llvm::IntegerType::get(getLLVMContext(), 32), 0);
        Builder.CreateStore(intValue, field);
      } else {
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
        //CurFn->viewCFG();
        Builder.CreateStore(intValue, field);
      }
    }

    // need access to these field decls so we
    // can determine if we will dynamically allocate
    // memory for each field
    const MeshType* MT = cast<MeshType>(T.getTypePtr());
    MeshDecl* MD = MT->getDecl();
    MeshDecl::mesh_field_iterator itr = MD->mesh_field_begin();
    MeshDecl::mesh_field_iterator itr_end = MD->mesh_field_end();

    llvm::Type *structTy = Alloc->getType()->getContainedType(0);
    for(unsigned i = 4, e = structTy->getNumContainedTypes(); i < e; ++i) {

      // Compute size of needed field memory in bytes
      llvm::Type *fieldTy = structTy->getContainedType(i);
      // If this is a externally allocated field, go on
      MeshFieldDecl* FD = *itr;
    
      if (itr != itr_end)
        ++itr;

      if (FD->isExternAlloc())
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




