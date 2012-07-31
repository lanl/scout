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

#include "llvm/Transforms/Scout/DoallToAMDIL/DoallToAMDIL.h"

#include "llvm/Analysis/Dominators.h"

using namespace llvm;

DoallToAMDIL::DoallToAMDIL()
  : ModulePass(ID){
}

DoallToAMDIL::~DoallToAMDIL(){

}

ModulePass *createDoallToAMDILPass() {
  return new DoallToAMDIL();
}

void DoallToAMDIL::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
}

const char *DoallToAMDIL::getPassName() const {
  return "Doall-to-AMDIL";
}

bool DoallToAMDIL::runOnModule(Module &M) {
  return true;
}

char DoallToAMDIL::ID = 1;
RegisterPass<DoallToAMDIL> DoallToAMDIL("Doall-to-AMDIL", "Generate LLVM bitcode for GPU kernels and embed as a global value.");
