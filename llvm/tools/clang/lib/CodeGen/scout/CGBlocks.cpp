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
 * Notes: Scout specific code to emit blocks used by the threaded cpu runtime
 *
 * ##### 
 */ 

#include "CGBlocks.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CallSite.h"

// =============================================================================
// Scout: Include code extractor.
// 
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include <algorithm>

// =============================================================================

using namespace clang;
using namespace CodeGen;

void computeBlockInfo(CodeGenModule &CGM, CodeGenFunction *CGF,
        CGBlockInfo &info);
llvm::Constant *buildBlockDescriptor(CodeGenModule &CGM,
                                            const CGBlockInfo &blockInfo);


// =============================================================================
// Scout:
//
// SC_TODO: We need to provide *much* better documentation here about
// what is going on!  In a nutshell, we extract the body of forall
// loops into their own functions with a iteration range as a
// parameter (in addition to other data) so we can easily transform it
// into a multi-threaded execution (or prepare to ship it off to the
// GPU).  But the Clang/LLVM centric details should be described in
// more detail to help understand the nuances and design choices...
//
//   (PM)
//
llvm::Value*
CodeGenFunction::EmitScoutBlockLiteral(const BlockExpr *blockExpr,
                                       CGBlockInfo &blockInfo,
                                       const llvm::SmallVector< llvm::Value *, 3 >& ranges, 
                                       llvm::SetVector< llvm::Value * > &inputs) {
  DEBUG_OUT("EmitScoutBlockLiteral");

  // Start generating block function.
  llvm::BasicBlock *blockEntry = createBasicBlock("block_entry");
  Builder.CreateBr(blockEntry);
  EmitBlock(blockEntry);

  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  llvm::Instruction *BlockAllocaInsertPt =
    new llvm::BitCastInst(Undef, Int32Ty, "", Builder.GetInsertBlock());
  BlockAllocaInsertPt->setName("blk.allocapt");

  // Save the AllocaInsertPt.
  llvm::Instruction *savedAllocaInsertPt = AllocaInsertPt;
  AllocaInsertPt = BlockAllocaInsertPt;

  // Generate body of function.
  std::string TheName = CurFn->getName();

  blockInfo = CGBlockInfo(blockExpr->getBlockDecl(), TheName.c_str());
  blockInfo.BlockExpression = blockExpr;
  
  // Compute information about the layout, etc., of this block.
  computeBlockInfo(CGM, this, blockInfo);

  // Build a loop around the block declaration to facilitate
  // the induction variable range information.
  llvm::Function *Fn = blockEntry->getParent();

  std::vector< llvm::Value * > PtrStart;
  std::vector< llvm::Value * > PtrEnd;
  std::string dim[] = { "x", "y", "z" };
  for(int i = 0, e = ScoutIdxVars.size(); i < e; ++i) {
    ScoutIdxVars[i]->setName("var." + dim[i]);
    Builder.SetInsertPoint(Fn->begin(), Fn->begin()->begin());
    PtrStart.push_back(Builder.CreateAlloca(Int32Ty, 0, "start." + dim[i]));
    PtrEnd.push_back(Builder.CreateAlloca(Int32Ty, 0, "end." + dim[i]));
    Builder.SetInsertPoint(blockEntry);
  }

  // (start, end) pairs must start the function argument list.
  // SC_TODO: what does a CreateLoad do w/o a lhs??
  for(unsigned i = 0, e = ScoutIdxVars.size(); i < e; ++i) {
    Builder.CreateLoad(PtrStart[i]);
    Builder.CreateLoad(PtrEnd[i]);
  }

  // setup indvar and extent
  llvm::Value *PtrIndVar = Builder.CreateAlloca(Int32Ty, 0, "forall.indvar");
  llvm::Value *IndVar  = Builder.CreateLoad(PtrStart[0]);
  llvm::Value *PtrExtent = Builder.CreateAlloca(Int32Ty, 0, "forall.extent");
  llvm::Value *Extent = Builder.CreateLoad(PtrEnd[0]);

  for(int i = 1, e = ScoutIdxVars.size(); i < e; ++i) {
    llvm::Value *n;
    if(i == 1)
      n = Builder.CreateLoad(ranges[0]); //size_x
    if(i == 2)
      n = Builder.CreateMul(Builder.CreateLoad(ranges[0]),
                            Builder.CreateLoad(ranges[1])); //size_x * size_y
    IndVar = Builder.CreateAdd(IndVar, Builder.CreateMul(n, Builder.CreateLoad(PtrStart[i])));
    Extent = Builder.CreateAdd(Extent, Builder.CreateMul(n, Builder.CreateLoad(PtrEnd[i])));
  }

  Builder.CreateStore(IndVar, PtrIndVar);
  Builder.CreateStore(Extent, PtrExtent);
  ForallIndVar = PtrIndVar;

  // Start the loop with a block that tests the condition.
  JumpDest Continue = getJumpDestInCurrentScope("for.blk.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);
  llvm::Value *cond = Builder.CreateICmpSLT(Builder.CreateLoad(PtrIndVar),
                                            Builder.CreateLoad(PtrExtent),
                                            "cmptmp");

  llvm::BasicBlock *ForBody = createBasicBlock("for.blk.body");
  llvm::BasicBlock *ExitBlock = createBasicBlock("for.blk.end");
  Builder.CreateCondBr(cond, ForBody, ExitBlock);

  EmitBlock(ForBody);
  Builder.SetInsertPoint(ForBody);

  llvm::Value *lval;
  for(unsigned i = 0, e = ScoutIdxVars.size(); i < e; ++i) {
    lval = Builder.CreateLoad(PtrIndVar);
    llvm::Value *val;
    if(i > 0) {
      if(i == 1)
        val = Builder.CreateLoad(ranges[0]); //size_x
      if (i == 2)
        val = Builder.CreateMul(Builder.CreateLoad(ranges[0]),
                                Builder.CreateLoad(ranges[1])); // size_x*size y
      lval = Builder.CreateUDiv(lval, val);
    }
    
    lval = Builder.CreateURem(lval, Builder.CreateLoad(ranges[i]));
    Builder.CreateStore(lval, ScoutIdxVars[i]);
  }
  
  const BlockDecl *blockDecl = blockInfo.getBlockDecl();
  EmitStmt(blockDecl->getBody());

  lval = Builder.CreateLoad(PtrIndVar);
  llvm::Value *one = llvm::ConstantInt::get(Int32Ty, 1);
  Builder.CreateStore(Builder.CreateAdd(lval, one), PtrIndVar);
  Builder.CreateBr(CondBlock);

  EmitBlock(ExitBlock);

  llvm::ReturnInst *ret =
    llvm::ReturnInst::Create(getLLVMContext(),
                             llvm::ConstantInt::get(Int32Ty, 0),
                             Builder.GetInsertBlock());

  llvm::SetVector< llvm::BasicBlock * > region;

  typedef llvm::Function::iterator BBIterator;
  BBIterator BB = CurFn->begin(), BB_end = CurFn->end();
  llvm::BasicBlock *split;
  for( ; BB->getName() != blockEntry->getName(); ++BB){
    split = BB;
  }

  for( ; BB != BB_end; ++BB){
    region.insert(BB);
  }

  // Gather inputs to region.
  for(unsigned i = 0, e = region.size(); i < e; ++i) {
    llvm::BasicBlock *BB = region[i];
    for(llvm::BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // If a used value is defined outside the region or
      // used outside the region, it's an input.
      for(llvm::User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O) {
        if(llvm::Instruction *Instn = dyn_cast< llvm::Instruction >(*O)) {
          if(!region.count(Instn->getParent()) &&
             !(*O)->getName().startswith("var.")) {
            inputs.insert(*O);
            //llvm::outs() << (*O)->getName().str() << "\n"; //start, end pairs then dim_x, etc.
          }
        }
      }
    }
  }

  llvm::DominatorTree DT;
  DT.runOnFunction(*CurFn);
  
  llvm::CodeExtractor 
  codeExtractor(std::vector<llvm::BasicBlock *>(region.begin(),
                region.end()), &DT, false);
  
  llvm::Function *BlockFn = codeExtractor.extractCodeRegion();

  assert(BlockFn != 0 && "Failed to rip BlockExpr contents into a new function.");

  llvm::BasicBlock *continueBB = ret->getParent();
  ret->eraseFromParent();

  Builder.SetInsertPoint(continueBB);
    
  // Remove function call to blockExpr function.
  llvm::BasicBlock *CallBB = split->getTerminator()->getSuccessor(0);

  typedef llvm::BasicBlock::iterator InstIterator;
  InstIterator I = CallBB->begin(), IE = CallBB->end();
  for( ; I != IE; ++I) {
    if(llvm::CallInst *call = dyn_cast< llvm::CallInst >(I)) {
      call->eraseFromParent();
      break;
    }
  }

  // Add the function arguments to the block descriptor.
  llvm::StructType *structTy = blockInfo.StructureType;
  llvm::SmallVector< llvm::Type *, 8 > arrayTy;
  for(unsigned i = 0, e = structTy->getNumElements(); i < e; ++i)
    arrayTy.push_back(structTy->getElementType(i));

  typedef llvm::Function::arg_iterator ArgIterator;

  // Ordering of args is (start,end) pairs, then dim_x etc, then captured vars
  for(ArgIterator arg = BlockFn->arg_begin(),
        end = BlockFn->arg_end(); arg != end; ++arg)
    if(!(arg->getName().startswith("var."))) {
      //llvm::outs() << "arg " << arg->getName().str() << "\n";
      arrayTy.push_back(Int8PtrTy);
    }

  blockInfo.StructureType = llvm::StructType::get(getLLVMContext(), arrayTy, true);
  blockInfo.CanBeGlobal = false;

  // Create the new block function signature and insert it into the module.
  llvm::FunctionType *NFTy = llvm::FunctionType::get(VoidTy,
                                                     Int8PtrTy,
                                                     false);

  llvm::Function *NewBlockFn = llvm::Function::Create(NFTy, BlockFn->getLinkage());
  NewBlockFn->getArgumentList().begin()->setName(".block_descriptor");

  BlockFn->getParent()->getFunctionList().insert(BlockFn, NewBlockFn);
  NewBlockFn->takeName(BlockFn);

  // Slurp the BBs out of BlockFn and into NewBlockFn.
  NewBlockFn->getBasicBlockList().splice(NewBlockFn->begin(), BlockFn->getBasicBlockList());

  continueBB = Builder.GetInsertBlock();

  Builder.SetInsertPoint(&NewBlockFn->getEntryBlock(),
                         NewBlockFn->getEntryBlock().begin());

  BlockPointer = Builder.CreateBitCast(NewBlockFn->arg_begin(),
                                       blockInfo.StructureType->getPointerTo(),
                                       "block");

  // Unpack block_descriptor function arguments and connect them
  // to their local variable counterparts.
  int idx = 5;
  for(llvm::Function::arg_iterator I = BlockFn->arg_begin(),
        E = BlockFn->arg_end(); I != E; ++I) {

    if(I->getName().startswith("var.")) {
      llvm::Value *indvar = Builder.CreateAlloca(Int32Ty, 0);
      I->replaceAllUsesWith(indvar);
      indvar->takeName(I);
    } else {
      llvm::Value *src = Builder.CreateStructGEP(LoadBlockStruct(),
                                                 idx++,
                                                 "block.capture.addr");
      src = Builder.CreateBitCast(Builder.CreateLoad(src),
                                  I->getType());

      std::vector< llvm::User * > Users(I->use_begin(), I->use_end());
      for(std::vector< llvm::User * >::iterator use = Users.begin(), useE = Users.end();
          use != useE; ++use)
        if(llvm::Instruction *Instn = dyn_cast< llvm::Instruction >(*use))
          if(region.count(Instn->getParent())) {
            Instn->replaceUsesOfWith(I, src);
            src->takeName(I);
          }
    }
  }

  // Delete old BlockFn.
  BlockFn->dropAllReferences();
  BlockFn->eraseFromParent();

  // Reset builder insert point.
  Builder.SetInsertPoint(continueBB);

  // Restore the AllocaInsertPtr.
  AllocaInsertPt = savedAllocaInsertPt;

  return NewBlockFn;
}


llvm::Value 
*CodeGenFunction::EmitScoutBlockFnCall(llvm::Value *blockFn,
                                       const CGBlockInfo &blockInfo,
                                       const llvm::SmallVector< llvm::Value *, 3 >& ranges,
                                       llvm::SetVector< llvm::Value * > &inputs) {
  DEBUG_OUT("EmitScoutBlockFnCall");

  blockFn = llvm::ConstantExpr::getBitCast(cast< llvm::Function >(blockFn),
                                           VoidPtrTy);
  llvm::Constant *isa = CGM.getNSConcreteStackBlock();
  isa = llvm::ConstantExpr::getBitCast(isa, VoidPtrTy);

  // Build the block descriptor.
  llvm::Constant *descriptor = buildBlockDescriptor(CGM, blockInfo);

  llvm::Type *intTy = ConvertType(getContext().IntTy);

  llvm::AllocaInst *blockAddr =
    CreateTempAlloca(blockInfo.StructureType, "block");
  blockAddr->setAlignment(blockInfo.BlockAlign.getQuantity());

  BlockPointer = Builder.CreateBitCast(blockAddr,
                                       blockInfo.StructureType->getPointerTo(),
                                       "block");

  // Compute the initial on-stack block flags.
  BlockFlags flags = BLOCK_HAS_SIGNATURE;
  if(blockInfo.NeedsCopyDispose) flags |= BLOCK_HAS_COPY_DISPOSE;
  if(blockInfo.HasCXXObject) flags |= BLOCK_HAS_CXX_OBJ;
  if(blockInfo.UsesStret) flags |= BLOCK_USE_STRET;

  // Initialize the block literal.
  Builder.CreateStore(isa, Builder.CreateStructGEP(blockAddr, 0, "block.isa"));
  Builder.CreateStore(llvm::ConstantInt::get(intTy, flags.getBitMask()),
                      Builder.CreateStructGEP(blockAddr, 1, "block.flags"));
  Builder.CreateStore(llvm::ConstantInt::get(intTy, 0),
                      Builder.CreateStructGEP(blockAddr, 2, "block.reserved"));
  Builder.CreateStore(blockFn, Builder.CreateStructGEP(blockAddr, 3,
                                                       "block.invoke"));
  Builder.CreateStore(descriptor, Builder.CreateStructGEP(blockAddr, 4,
                                                          "block.descriptor"));

  size_t numDimensions = 0;

  // Captured variables (i.e. inputs).
  for(unsigned i = 0, e = inputs.size(); i < e; ++i) {
    // This will be a [[type]]*, except that a byref entry will just be
    // an i8**.
    llvm::Value *blockField =
      Builder.CreateStructGEP(blockAddr, i + 5,
                              "block.captured");

    llvm::Value *I = inputs[i];

    llvm::Value *var = I;
    llvm::Value *zero = llvm::ConstantInt::get(Int32Ty, 0);
    if(I->getName().startswith("start.")) {
      Builder.CreateStore(zero, I); var = I;
    } else if(I->getName().startswith("end.")) {
      llvm::Value *val;
      if(I->getName().startswith("end.x")) val = Builder.CreateLoad(ranges[0]);
      if(I->getName().startswith("end.y")) val = Builder.CreateLoad(ranges[1]);
      if(I->getName().startswith("end.z")) val = Builder.CreateLoad(ranges[2]);
      Builder.CreateStore(val, I); var = I;
      ++numDimensions;
    } else if(I->getName().startswith("var.")) {
      var = Builder.CreateAlloca(Int32Ty, 0, I->getName());
      Builder.CreateStore(zero, var);
    }

    // Write that void* into the capture field.
    Builder.CreateStore(Builder.CreateBitCast(var, Int8PtrTy),
                        blockField);
  }
  // Cast to the converted block-pointer type, which happens (somewhat
  // unfortunately) to be a pointer to function type.
  llvm::FunctionType *funcTy = llvm::FunctionType::get(Int32Ty, false);
  llvm::Type *funcPtrTy = llvm::PointerType::get(funcTy, 0);
  llvm::Value *blockVal = Builder.CreateBitCast(blockAddr, funcPtrTy);

  // Allocate a block.
  llvm::Value *blk = Builder.CreateAlloca(funcPtrTy, 0, "blk");

  Builder.CreateStore(blockVal, blk);
  blk = Builder.CreateLoad(blk);

  llvm::Type *genericBlkTy = llvm::PointerType::getUnqual(CGM.getGenericBlockLiteralType());
  llvm::Value *genericBlk = Builder.CreateBitCast(blk, genericBlkTy, "block.literal");

  blk = Builder.CreateConstInBoundsGEP2_32(genericBlk, 0, 3);
  blk = Builder.CreateLoad(blk);

  funcTy = llvm::FunctionType::get(Int32Ty, Int8PtrTy, false);
  funcPtrTy = llvm::PointerType::get(funcTy, 0);
  blk = Builder.CreateBitCast(blk, funcPtrTy);

  genericBlk = Builder.CreateBitCast(genericBlk, Int8PtrTy);

  return EmitScoutQueueBlock(genericBlk, numDimensions, inputs.size());

}

// Emit call to __scrt_queue_block runtime function
llvm::Value *CodeGenFunction::EmitScoutQueueBlock(llvm::Value *genericBlk, size_t numDimensions, size_t numInputs) {

  llvm::Function* queueBlockFunc =
  CGM.getModule().getFunction("__scrt_queue_block");

  if(!queueBlockFunc){
    llvm::PointerType* p1 =
    llvm::PointerType::get(llvm::Type::getInt8Ty(getLLVMContext()), 0);

    llvm::Type* p2 = llvm::Type::getInt32Ty(getLLVMContext());
    llvm::Type* p3 = llvm::Type::getInt32Ty(getLLVMContext());

    std::vector<llvm::Type*> args;

    args.push_back(p1);
    args.push_back(p2);
    args.push_back(p3);

    llvm::FunctionType* ft =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                            args, false);

    queueBlockFunc =
    llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                           "__scrt_queue_block", &CGM.getModule());
  }

  llvm::Value* Dims = llvm::ConstantInt::get(Int32Ty, numDimensions);
  llvm::Value* Inputs = llvm::ConstantInt::get(Int32Ty, numInputs);

  return Builder.CreateCall3(queueBlockFunc, genericBlk, Dims, Inputs);
}


//
// =============================================================================
