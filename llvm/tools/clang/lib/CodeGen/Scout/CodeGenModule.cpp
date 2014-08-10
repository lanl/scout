/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
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

#include "CodeGenModule.h"
#include "clang/AST/Scout/MeshDecl.h"
#include "Scout/CGLegionRuntime.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

void CodeGenModule::UpdateCompletedType(const MeshDecl *MD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(MD);
}

llvm::Function *CodeGenModule::lsciMainFunction() {

  std::string funcName = "lsci_main";

  llvm::Function *lsciMainFunc = TheModule.getFunction(funcName);
  if(!lsciMainFunc) {
   std::vector<llvm::Type*> args = {LegionRuntime->Int32Ty, LegionRuntime->PointerTy(LegionRuntime->PointerTy(LegionRuntime->Int8Ty))};

   llvm::FunctionType *FTy =
   llvm::FunctionType::get(llvm::Type::getInt32Ty(getLLVMContext()),
                            args, false /* not var args */);

   lsciMainFunc = llvm::Function::Create(FTy,
                                     llvm::Function::ExternalLinkage,
                                     funcName,
                                     &TheModule);
  }
  return lsciMainFunc;
}

void CodeGenModule::startLsciMainFunction() {

  // need to create main_task()
  llvm::LLVMContext& context = getLLVMContext();
  llvm::IRBuilder<> Builder(TheModule.getContext());

  CGLegionRuntime& r = getLegionRuntime();

  //fetch or create type for lsci_task_args_t
  llvm::StructType *TaskArgsTy = TheModule.getTypeByName("struct.lsci_task_args_t");
  if (!TaskArgsTy) {
    std::vector<llvm::Type*> fields = {
      r.ContextTy,
      r.RuntimeTy,
      r.Int32Ty,
      r.Int64Ty,
      r.PhysicalRegionsTy,
      r.VoidPtrTy};
    TaskArgsTy = llvm::StructType::create(context, fields, "struct.lsci_task_args_t");
  }

  llvm::PointerType* TaskArgsPtrTy = llvm::PointerType::get(TaskArgsTy, 0);

  // use lsci_task_args_t to create main_task function type and function
  std::vector<llvm::Type*> params = {TaskArgsPtrTy};
  llvm::FunctionType* main_task_ft = llvm::FunctionType::get(VoidTy, params, false);

  llvm::Function* main_task = llvm::Function::Create(main_task_ft,
      llvm::Function::ExternalLinkage,
      "MainTaskFunction",
      &getModule());

  llvm::BasicBlock *BB = llvm::BasicBlock::Create(getLLVMContext(), "entry", main_task);

  Builder.SetInsertPoint(BB);

  // make call to main_prime(), which contains the original stuff in main()
  // but we don't know argc/argv?

  Builder.CreateRetVoid();

  // Fill in initial part of lsci_main(), which contains sets the top level task and registers main_task
  CodeGenFunction CGF(*this);

  // make call to lsci_set_toplevel_task_id
  llvm::Function* lsciMainFunc = lsciMainFunction();
  BB = llvm::BasicBlock::Create(getLLVMContext(), "entry", lsciMainFunc);
  Builder.SetInsertPoint(BB);
  llvm::ConstantInt* mainTID = llvm::ConstantInt::get(Int32Ty, 0);
  std::vector<llvm::Value*> args = {mainTID};
  Builder.CreateCall(LegionRuntime->SetTopLevelTaskIdFunc(), args); 

  // get lsci_register_void_legion_task function type
  std::string name = "lsci_register_void_legion_task_aux";
  llvm::Function* regVoidLegionTaskAuxFunc = TheModule.getFunction(name);
  if(!regVoidLegionTaskAuxFunc){
    std::vector<llvm::Type*> params =
    {r.Int32Ty, r.Int32Ty, r.Int1Ty, r.Int1Ty, r.Int1Ty, r.VariantIdTy, VoidPtrTy /* ptr to Int8Ty */, llvm::PointerType::get(main_task_ft,0)};

    llvm::FunctionType* regVoidLegionTaskAuxFuncTy = llvm::FunctionType::get(Int32Ty, params, false);

    regVoidLegionTaskAuxFunc = llvm::Function::Create(regVoidLegionTaskAuxFuncTy,
        llvm::Function::ExternalLinkage,
        name,
        &TheModule);
  }  

  // make call to lsci_register_void_legion_task
  std::vector<llvm::Value*> reg_main_task_params;

  reg_main_task_params.push_back(mainTID);

  llvm::ConstantInt* lsci_loc_proc = llvm::ConstantInt::get(Int32Ty, 1);
  reg_main_task_params.push_back(lsci_loc_proc);

  llvm::ConstantInt* true_val = llvm::ConstantInt::get(context, llvm::APInt(1, StringRef("-1"), 10));
  reg_main_task_params.push_back(true_val);

  llvm::ConstantInt* false_val = llvm::ConstantInt::get(context, llvm::APInt(1, StringRef("0"), 10));
  reg_main_task_params.push_back(false_val);

  reg_main_task_params.push_back(false_val);

  // must be a better way to get umax
  llvm::ConstantInt* umax = llvm::ConstantInt::get(Int64Ty, 4294967295);
  reg_main_task_params.push_back(umax);

  llvm::Value* main_task_name = Builder.CreateGlobalStringPtr("main_task");
  reg_main_task_params.push_back(main_task_name);

  reg_main_task_params.push_back(main_task);
   
  // Call lsci_register_void_legion_task()
  Builder.CreateCall(regVoidLegionTaskAuxFunc, reg_main_task_params); 

  // create ret instruction
  llvm::ConstantInt* retVal = llvm::ConstantInt::get(getLLVMContext(), llvm::APInt::APInt(32, StringRef("0"), 10));
  Builder.CreateRet(retVal);
}

#if 0

void CodeGenModule::addToLsciMainFunction(int taskID, string funcName) {
  CodeGenFunction CGF(*this);
  llvm::Function* lsciMainFunc = lsciMainFunction();
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(llvm::getGlobalContext(), "entry", lsciMainFunc);
  llvm::IRBuilder<> Builder(TheModule.getContext());
  Builder.SetInsertPoint(BB);

  // create lsci_reg_task_data_t  -- may not need this right now
  llvm::StructType *RegTaskDataTy = TheModule.getTypeByName("struct.lsci_reg_task_data_t");
  if (!RegTaskDataTy) {
    std::vector<llvm::Type*> fields = {main_task_ft};
    TaskArgsTy = llvm::StructType::create(context, fields, "struct.lsci_reg_task_data_t");
  }


}
#endif
