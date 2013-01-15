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

#include <string>
#include <map>

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

class DoallToAMDIL : public llvm::ModulePass {
 public:
  static char ID;
  
  typedef std::map<std::string, llvm::MDNode*> FunctionMDMap;

  DoallToAMDIL(const std::string& sccPath);

  ~DoallToAMDIL();

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  const char *getPassName() const;

  bool runOnModule(llvm::Module &M);

  llvm::Module* CloneGPUModule(const llvm::Module *M,
			       llvm::ValueToValueMapTy &VMap);

  llvm::Module* createGPUModule(const llvm::Module& m);

private:
  FunctionMDMap functionMDMap;
  std::string sccPath_;
};

llvm::ModulePass *createDoallToAMDILPass();

#endif
