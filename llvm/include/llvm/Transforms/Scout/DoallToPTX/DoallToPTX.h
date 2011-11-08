/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */

#ifndef _SC_LLVM_DOALLTOPTX_H_
#define _SC_LLVM_DOALLTOPTX_H_

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/TargetRegistry.h"

#include <string>
#include <iostream>

#include <llvm/Transforms/Scout/Driver/CudaDriver.h>
#include <llvm/Transforms/Scout/Driver/PTXDriver.h>

class DoallToPTX : public llvm::ModulePass {
 public:
  typedef llvm::DenseSet< const llvm::Value * > ValueSet;

  static char ID;
  DoallToPTX();

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
  const char *getPassName() const;
  bool runOnModule(llvm::Module &M);

  llvm::GlobalValue *embedPTX(llvm::Module &ptxModule, llvm::Module &cpuModule);
  void generatePTXHandler(CudaDriver &cuda, llvm::Module &module,
                          std::string funcName, llvm::GlobalValue *ptxAsm);
  llvm::Module *CloneModule(const llvm::Module *M, llvm::ValueToValueMapTy &VMap);
  void pruneModule(llvm::Module &module, llvm::ValueToValueMapTy &valueMap,
                   llvm::Function &func);

  void setGPUThreading(CudaDriver &cuda, llvm::Function *FN, bool uniform);
  void translateVarToTid(CudaDriver &cuda, llvm::Instruction *inst, bool uniform);
};

llvm::ModulePass *createDoallToPTXPass();

#endif
