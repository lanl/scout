/**
 * @file   PTXPasses.h
 * @date   08.08.2009
 * @author Helge Rhodin
 *
 *
 * Copyright (C) 2009, 2010 Saarland University
 *
 * This file is part of llvmptxbackend.
 *
 * llvmptxbackend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * llvmptxbackend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with llvmptxbackend.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PTXPASSES_H
#define PTXPASSES_H

#include "PTXTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
//#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Config/config.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "PTXBackend.h"

using namespace llvm;

class PTXBackendInsertSpecialInstructions : public BasicBlockPass
{
  // replaces exp,log with ex2,lg2
#define LOG2_E 1.442695041
#define LOG2_E_REC 0.6931471806

  Function* ex2fFun;
  Function* lg2fFun;
  Function* sinfFun;
  Function* cosfFun;

  Function* ex2Fun;
  Function* lg2Fun;
  Function* sinFun;
  Function* cosFun;

  std::map<const Value *, const Value *>& parentPointers;

 public:
  static char ID;

 PTXBackendInsertSpecialInstructions(std::map<const Value *, const Value *>&
                                     parentCompositePointer)
   : BasicBlockPass(ID),parentPointers(parentCompositePointer) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }

  virtual const char *getPassName() const {
    return "PTX backend: insert special ptx instructions";
  }

  using BasicBlockPass::doInitialization;
  virtual bool doInitialization(Module &M)
  {
    //    M.dump();

    Type *floatTy = Type::getPrimitiveType(M.getContext(), Type::FloatTyID);
    Type *doubleTy = Type::getPrimitiveType(M.getContext(), Type::DoubleTyID);
    //create all the functions we need, not possible in runOnBasicBlock()

    ex2fFun = Intrinsic::getDeclaration(&M, Intrinsic::exp2, floatTy);
    lg2fFun = Intrinsic::getDeclaration(&M, Intrinsic::log2, floatTy);
    sinfFun = Intrinsic::getDeclaration(&M, Intrinsic::sin, floatTy);
    cosfFun = Intrinsic::getDeclaration(&M, Intrinsic::cos, floatTy);

    ex2Fun = Intrinsic::getDeclaration(&M, Intrinsic::exp2, doubleTy);
    lg2Fun = Intrinsic::getDeclaration(&M, Intrinsic::log2, doubleTy);
    sinFun = Intrinsic::getDeclaration(&M, Intrinsic::sin, doubleTy);
    cosFun = Intrinsic::getDeclaration(&M, Intrinsic::cos, doubleTy);

    return true;
  }

  bool replaceSpecialFunctionsWithPTXInstr(CallInst* callI);
  bool simplifyGEPInstructions(GetElementPtrInst* GEPInst);

  virtual bool runOnBasicBlock(BasicBlock &BB)
  {
    bool changedBlock = false;

    iplist<Instruction>::iterator I = BB.getInstList().begin();
    for (iplist<Instruction>::iterator nextI = I,
           E = --BB.getInstList().end(); I != E; I = nextI)
      {
        iplist<Instruction>::iterator I = nextI++;

        // check for special functions implemented in ptx
        if(CallInst* call = dyn_cast<CallInst>(&*I))
          changedBlock = (replaceSpecialFunctionsWithPTXInstr(call)
                          || changedBlock);

        // simplify GEP instructions wit mul,add etc. to the trivial calse
        // (GEP (pointer, constant offset))
        else if(GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(&*I))
          changedBlock = (simplifyGEPInstructions(gep) || changedBlock);
      }
    return changedBlock;
  }
};

#define CONSTWRAPPERNAME "constWrapper"
class PTXPolishBeforeCodegenPass : public BasicBlockPass
{
 public:
  static char ID;

 PTXPolishBeforeCodegenPass()
   : BasicBlockPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
  }

  virtual const char *getPassName() const {
    return "PTX backend: removes constWrapper fun(generated by GEP simplifi.)";
  }

  using BasicBlockPass::doInitialization;
  virtual bool doInitialization(Module &M)
  {
    return true;
  }

  virtual bool runOnBasicBlock(BasicBlock &BB)
  {
    bool changedBlock = false;

    iplist<Instruction>::iterator I = BB.getInstList().begin();
    for (iplist<Instruction>::iterator nextI = I,
           E = --BB.getInstList().end(); I != E; I = nextI)
      {
        iplist<Instruction>::iterator I = nextI++;

        // check for wrapperfunctions
        if(CallInst* callI = dyn_cast<CallInst>(&*I))
          if(callI->getCalledFunction()->getName().str().compare(CONSTWRAPPERNAME)==0)
        {
          //replace function call result with its parameter
          callI->replaceAllUsesWith(callI->getArgOperand(0));
          callI->eraseFromParent();
          changedBlock = true;
        }
      }
    return changedBlock;
  }
};

class PTXBackendNameAllUsedStructsAndMergeFunctions : public ModulePass
{
 public:
  static char ID;
  PTXBackendNameAllUsedStructsAndMergeFunctions()
    : ModulePass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<FindUsedTypes>();
  }

  virtual const char *getPassName() const {
    return "PTX backend type canonicalizer";
  }

  virtual bool runOnModule(Module &M);
};

#endif //CTARGETMACHINE_H
