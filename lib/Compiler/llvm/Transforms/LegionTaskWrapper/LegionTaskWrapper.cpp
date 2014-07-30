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

#include "llvm/Transforms/Scout/LegionTaskWrapper/LegionTaskWrapper.h"
#include "llvm/IR/Module.h"

using namespace llvm;

char LegionTaskWrapper::ID = 1;

LegionTaskWrapper::LegionTaskWrapper() : ModulePass(LegionTaskWrapper::ID) {}

bool LegionTaskWrapper::runOnModule(Module &M) {

  // iterate through functions in module M
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {

    // get function
    Function& F = *I;

    // figure out if f can be a legion task by checking its metadata
    // metadata for a function is on a basic block after entry, so check through BBs
    for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E;++BBI) {

      BasicBlock &BB = *BBI;

      // if name of basic block is task.md, it can be wrapped as a task
      StringRef BBName = BB.getName();
      std::string MDStr = "task.md";
      StringRef MDName(MDStr); 
      if (BBName.equals(MDName) ) {
//        errs() << "Task found: ";
//        errs().write_escaped(F.getName()) << '\n';

        // TO DO: create legion task wrapper 

      }
    }
  }

  // return true if modified IR
  return false;
}

ModulePass* llvm::createLegionTaskWrapperPass(){
  return new LegionTaskWrapper;
}

RegisterPass<LegionTaskWrapper> LegionTaskWrapper("sclegion-task-wrapper", "Generate Legion task wrappers for Scout functions.");

