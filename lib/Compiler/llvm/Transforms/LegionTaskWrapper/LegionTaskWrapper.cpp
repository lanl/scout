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
#include "llvm/IR/IRBuilder.h" 

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

      // make main_prime call lsci_main() at the end to do lsci startup stuff

      IRBuilder<> builder(M.getContext());

      // Go through and find last block in main()
      BasicBlock &lastBlock = F.back();

      errs() << "found last block of main()\n";

      // Find place to insert call to lsci_main(), before last instruction in the block
      builder.SetInsertPoint(&(lastBlock.back()));

      // create call instruction
      llvm::Function::arg_iterator arg_iter = F.arg_begin();
      llvm::Value* argcValue = arg_iter++;
      llvm::Value* argvValue = arg_iter++;

      llvm::SmallVector< llvm::Value *, 2 > args;
      args.push_back(argcValue);
      args.push_back(argvValue);

      // call lsci_main() 
      llvm::Function* lsci_mainFunc;
      lsci_mainFunc = M.getFunction("lsci_main"); 
      llvm::Value* lsciCallRet = builder.CreateCall(lsci_mainFunc, args);

      // create ret instruction
      builder.CreateRet(lsciCallRet);

      // Now get last instruction from this block (the previous return instruction) and delete it
      llvm::Instruction& lastInst = lastBlock.back();
      lastInst.eraseFromParent();
    }
  }
  // return true if modified IR
  return modifiedIR;
}

ModulePass* llvm::createLegionTaskWrapperPass(){
  return new LegionTaskWrapper;
}

RegisterPass<LegionTaskWrapper> LegionTaskWrapper("sclegion-task-wrapper", "Generate Legion task wrappers for Scout functions.");

