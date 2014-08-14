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

   // name the two args argc and argv
   llvm::Function::arg_iterator argiter = lsciMainFunc->arg_begin();
   llvm::Value* int32_argc = argiter++;
   int32_argc->setName("argc");
   llvm::Value* ptr_argv = argiter++;
   ptr_argv->setName("argv");

  }
  return lsciMainFunc;
}

void CodeGenModule::startLsciMainFunction() {

  // need to create main_task()
  llvm::LLVMContext& context = getLLVMContext();
  llvm::IRBuilder<> Builder(TheModule.getContext());

  CGLegionRuntime& r = getLegionRuntime();

  llvm::PointerType* TaskArgsPtrTy = llvm::PointerType::get(r.TaskArgsTy, 0);

  // use lsci_task_args_t to create main_task function type and function
  std::vector<llvm::Type*> params = {TaskArgsPtrTy};
  llvm::FunctionType* main_task_ft = llvm::FunctionType::get(VoidTy, params, false);

  llvm::Function* main_task = llvm::Function::Create(main_task_ft,
      llvm::Function::ExternalLinkage,
      "main_task",
      &getModule());

// make it empty for now.  Will fill with main() stuff
#if 0
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(getLLVMContext(), "entry", main_task);

  Builder.SetInsertPoint(BB);

  // make call to main_prime(), which contains the original stuff in main()
  // but we don't know argc/argv?

  Builder.CreateRetVoid();
#endif

  // Fill in initial part of lsci_main(), which contains sets the top level task and registers main_task
  // Could just call regTaskInLsciMainFunction(0, main_task);

  CodeGenFunction CGF(*this);

  // make call to lsci_set_toplevel_task_id
  llvm::Function* lsciMainFunc = lsciMainFunction();
  llvm::BasicBlock* BB = llvm::BasicBlock::Create(getLLVMContext(), "entry", lsciMainFunc);
  Builder.SetInsertPoint(BB);
  llvm::ConstantInt* mainTID = llvm::ConstantInt::get(Int32Ty, 0);
  std::vector<llvm::Value*> args = {mainTID};
  Builder.CreateCall(LegionRuntime->SetTopLevelTaskIdFunc(), args); 

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
  Builder.CreateCall(r.RegisterVoidLegionTaskAuxFunc(), reg_main_task_params); 

  // start a new basic block to put the ending stuff in
  BB = llvm::BasicBlock::Create(getLLVMContext(), "end", lsciMainFunc);
  llvm::BasicBlock &firstBlock = lsciMainFunc->front();
  Builder.SetInsertPoint(&firstBlock);
  Builder.CreateBr(BB);

  // start ending stuff
  Builder.SetInsertPoint(BB);
  finishLsciMainFunction();

}

void CodeGenModule::regTaskInLsciMainFunction(int taskID, llvm::Function* taskFunc) {
  CodeGenFunction CGF(*this);
  CGLegionRuntime& r = getLegionRuntime();
  llvm::LLVMContext& context = getLLVMContext();
  llvm::Function* lsciMainFunc = lsciMainFunction();
  llvm::IRBuilder<> Builder(TheModule.getContext());

  // Go through and find first block in main() 
  llvm::BasicBlock &firstBlock = lsciMainFunc->front();

  // Find place to insert, at end of this block
  Builder.SetInsertPoint(&firstBlock);

  // Find place to insert, before last instruction in the block
  Builder.SetInsertPoint(&(firstBlock.back()));

  // make call to lsci_register_void_legion_task
  std::vector<llvm::Value*> reg_main_task_params;

  llvm::ConstantInt* task_id =  llvm::ConstantInt::get(Int32Ty, taskID);
  reg_main_task_params.push_back(task_id);

  llvm::ConstantInt* lsci_loc_proc = llvm::ConstantInt::get(Int32Ty, 1);
  reg_main_task_params.push_back(lsci_loc_proc);

  llvm::ConstantInt* true_val = llvm::ConstantInt::get(context, llvm::APInt(1, StringRef("-1"), 10));
  reg_main_task_params.push_back(true_val);

  llvm::ConstantInt* false_val = llvm::ConstantInt::get(context, llvm::APInt(1, StringRef("0"), 10));
  reg_main_task_params.push_back(false_val);

  reg_main_task_params.push_back(false_val);

  // must be a better way to get umax, since will be system-dependent?
  llvm::ConstantInt* umax = llvm::ConstantInt::get(Int64Ty, 4294967295);
  reg_main_task_params.push_back(umax);

  llvm::Value* task_name = Builder.CreateGlobalStringPtr(taskFunc->getName());
  reg_main_task_params.push_back(task_name);

  reg_main_task_params.push_back(taskFunc);
   
  // Call lsci_register_void_legion_task()
  Builder.CreateCall(r.RegisterVoidLegionTaskAuxFunc(), reg_main_task_params); 

}

void CodeGenModule::finishLsciMainFunction() {
  CodeGenFunction CGF(*this);
  llvm::Function* lsciMainFunc = lsciMainFunction();
  llvm::IRBuilder<> Builder(TheModule.getContext());
  CGLegionRuntime& r = getLegionRuntime();

  // Go through and find last block in main()
  llvm::BasicBlock &lastBlock = lsciMainFunc->back();

  // Find place to insert, after last block
  Builder.SetInsertPoint(&lastBlock);

  // allocate argc and argv
  llvm::AllocaInst* ptr_argc_addr = Builder.CreateAlloca(Int32Ty, 0, "argc.addr");
  llvm::AllocaInst* ptr_argv_addr = Builder.CreateAlloca(llvm::PointerType::get(llvm::PointerType::get(Int8Ty, 0), 0), 0, "argv.addr");

  llvm::Function::arg_iterator args = lsciMainFunc->arg_begin();
  llvm::Value* argcVal = args++;
  llvm::Value* argvVal = args++;

  // store to argc and argv
  Builder.CreateStore(argcVal, ptr_argc_addr);
  Builder.CreateStore(argvVal, ptr_argv_addr);

  // load argc and argc
  llvm::LoadInst* load_argc = Builder.CreateLoad(ptr_argc_addr);
  llvm::LoadInst* load_argv = Builder.CreateLoad(ptr_argv_addr);

  // call lsci_start(argc, argv)
  std::vector<llvm::Value*> params;
  params.push_back(load_argc);
  params.push_back(load_argv);
  llvm::CallInst* retVal = Builder.CreateCall(r.StartFunc(), params);

  // return result of lsci_start()
  Builder.CreateRet(retVal);
  //Builder.CreateRet(llvm::ConstantInt::get(Int32Ty, 0));
}
