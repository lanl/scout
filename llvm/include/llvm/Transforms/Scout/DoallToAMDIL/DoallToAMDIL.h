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

#ifndef _SC_LLVM_DOALLTOAMDIL_H_
#define _SC_LLVM_DOALLTOAMDIL_H_

#include "llvm/Pass.h"

class DoallToAMDIL : public llvm::ModulePass {
 public:
  static char ID;

  DoallToAMDIL();

  ~DoallToAMDIL();

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  const char *getPassName() const;

  bool runOnModule(llvm::Module &M);

private:
  
};

llvm::ModulePass *createDoallToAMDILPass();

#endif
