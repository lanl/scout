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
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

char LegionTaskWrapper::ID = 1;

LegionTaskWrapper::LegionTaskWrapper() : ModulePass(LegionTaskWrapper::ID) {}

bool LegionTaskWrapper::runOnModule(Module &M) {

  bool modifiedIR = false;

  // iterate through functions in module M
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {

    // get function
    Function& F = *I;

    // Check if function is "main()" and if so, extract it and make a new function "main_prime()".
    // Make main() call main_prime().

    if (F.getName() == "main") {
      //errs() << "main found: ";
      //errs().write_escaped(F.getName()) << '\n';

      // This extractor code is mostly from clang/lib/CodeGen/CGStmt.cpp: CodeGenFunction::ExtractRegion()
      std::vector< llvm::BasicBlock * > Blocks;

      llvm::Function::iterator BB = F.begin();

      // collect forall basic blocks up to exit
      for( ; BB != F.end(); ++BB) {

        // look for function local metadata
        for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end(); II != IE; ++II) {

          for(unsigned i = 0, e = II->getNumOperands(); i!=e; ++i){
            
            if(MDNode *N = dyn_cast_or_null<MDNode>(II->getOperand(i))){

              if (N->isFunctionLocal()) {

                // just remove function local metadata
                // see http://lists.cs.uiuc.edu/pipermail/llvmdev/2013-November/068205.html
                N->replaceOperandWith(i, 0);
              }
            } 
          }
        }
        Blocks.push_back(BB);
      }

      //SC_TODO: should we be using a DominatorTree?
      //llvm::DominatorTree DT;

      llvm::CodeExtractor codeExtractor(Blocks, 0/*&DT*/, false);

      llvm::Function *FprimeFn = codeExtractor.extractCodeRegion();

      const std::string name("main_prime");

      FprimeFn->setName(name);

      modifiedIR = true;

    }

    // Generate code to do the following:
    // set top level task ID (get MAIN_TID from metadata)
    // register main task data
    // register other tasks (get info from metadata)
    // start legion

  }
  // return true if modified IR
  return modifiedIR;
}

ModulePass* llvm::createLegionTaskWrapperPass(){
  return new LegionTaskWrapper;
}

RegisterPass<LegionTaskWrapper> LegionTaskWrapper("sclegion-task-wrapper", "Generate Legion task wrappers for Scout functions.");

