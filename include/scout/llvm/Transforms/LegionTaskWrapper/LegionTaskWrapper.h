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

#ifndef LLVM_SCOUT_LEGIONTASKWRAPPER_H
#define LLVM_SCOUT_LEGIONTASKWRAPPER_H


#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Module;

class LegionTaskWrapper : public ModulePass {

  public:

  LegionTaskWrapper(); 
  ~LegionTaskWrapper() {}
  bool runOnModule(Module &M) override;

  static char ID;
 
}; // end class LeginTaskWrapper

ModulePass *createLegionTaskWrapperPass();

} //end llvm namespace

#endif
