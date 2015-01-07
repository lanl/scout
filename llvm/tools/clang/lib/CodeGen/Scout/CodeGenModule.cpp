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

#include <limits>

#include "clang/AST/Scout/MeshDecl.h"
#include "Scout/CGLegionRuntime.h"
#include "Scout/CGLegionCRuntime.h"
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

void CodeGenModule::startLsciMainFunction(){
  using namespace std;
  using namespace llvm;

  typedef vector<Value*> ValueVec;
  
  IRBuilder<> B(TheModule.getContext());
  auto& C = getLLVMContext();
  CGLegionCRuntime& R = getLegionCRuntime();
  llvm::Module& M = getModule();
  
  Function* main_task =
  Function::Create(R.VoidTaskFuncTy,
                   llvm::Function::ExternalLinkage,
                   "main_task",
                   &M);
  
  Function* lsciMainFunc = lsciMainFunction();
  BasicBlock* BB =
  BasicBlock::Create(C, "entry", lsciMainFunc);
  B.SetInsertPoint(BB);
  
  ValueVec args =
  {B.CreateGlobalStringPtr("main_task"), main_task};
    
  B.CreateCall(R.ScInitFunc(), args);
  
  BB = llvm::BasicBlock::Create(C, "end", lsciMainFunc);

  llvm::BasicBlock &firstBlock = lsciMainFunc->front();
  B.SetInsertPoint(&firstBlock);
  B.CreateBr(BB);
  
  B.SetInsertPoint(BB);
  finishLsciMainFunction();
}

void CodeGenModule::regTaskInLsciMainFunction(int taskID,
                                              llvm::Function* taskFunc){
  using namespace std;
  using namespace llvm;
  
  typedef vector<Value*> ValueVec;
  
  IRBuilder<> B(TheModule.getContext());
  CGLegionCRuntime& R = getLegionCRuntime();
  
  Function* lsciMainFunc = lsciMainFunction();
  
  // Go through and find first block in main() 
  BasicBlock &firstBlock = lsciMainFunc->front();

  // Find place to insert, at end of this block
  B.SetInsertPoint(&firstBlock);

  // Find place to insert, before last instruction in the block
  B.SetInsertPoint(&(firstBlock.back()));

  ValueVec args =
  {ConstantInt::get(R.TaskIdTy, taskID),
    B.CreateGlobalStringPtr(taskFunc->getName()),
    taskFunc};
  
  B.CreateCall(R.ScRegisterTaskFunc(), args);
}

void CodeGenModule::finishLsciMainFunction() {
  CodeGenFunction CGF(*this);
  llvm::Function* lsciMainFunc = lsciMainFunction();
  llvm::IRBuilder<> Builder(TheModule.getContext());
  CGLegionCRuntime& r = getLegionCRuntime();

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
  llvm::CallInst* retVal = Builder.CreateCall(r.ScStartFunc(), params);

  // return result of lsci_start()
  Builder.CreateRet(retVal);
  //Builder.CreateRet(llvm::ConstantInt::get(Int32Ty, 0));
}
