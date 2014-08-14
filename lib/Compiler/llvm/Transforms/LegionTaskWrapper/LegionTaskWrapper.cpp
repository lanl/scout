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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

char LegionTaskWrapper::ID = 1;

LegionTaskWrapper::LegionTaskWrapper() : ModulePass(LegionTaskWrapper::ID) {}

bool LegionTaskWrapper::runOnModule(Module &M) {

  bool modifiedIR = false;

  //M.dump();

  // iterate through functions in module M
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {

    // get function
    Function& F = *I;

    // Check if function is "main()" 
    if (F.getName() == "main") {
      //errs() << "main found: ";

      // This extractor code is mostly from clang/lib/CodeGen/CGStmt.cpp: CodeGenFunction::ExtractRegion()
      // Do we want to remove function local metadata?
      std::vector< llvm::BasicBlock * > Blocks;

      llvm::Function::iterator BB = F.begin();

      // collect basic blocks up to exit
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

      llvm::CodeExtractor codeExtractor(Blocks, 0/*&DT*/, false);

      // iterate through functions in module M
      Function* mainTaskFunc = NULL;

      for (Module::iterator I2 = M.begin(), E2 = M.end(); I2 != E2; ++I2) {

        // get function
        Function &F2 = *I2;

        // Check if function is "main_task()" and if so, keep a ptr to it

        if (F2.getName() == "main_task") {
            mainTaskFunc = &F2;
            //errs() << "found main_task()\n";
        }

        // extract code from main() and put it in main_task()
        // TODO check if we have a main_task().  If not, we have a problem.
        if (!mainTaskFunc) {
              //errs() << "Did not find main_task()\n";
        }
      }


      codeExtractor.extractCodeRegionIntoMainTaskFunc(mainTaskFunc);
      //errs() << "extracted code into main_task()\n";

      // Remove the int return instruction at the end
      BasicBlock &lastBlock = mainTaskFunc->back();
      llvm::Instruction& lastInst = lastBlock.back();
      //errs() << "erasing previous return instruction\n";
      lastInst.eraseFromParent();

      // Put in a void return instruction at the end
      IRBuilder<> builder(M.getContext());
      builder.SetInsertPoint(&lastBlock);
      builder.CreateRetVoid();

      modifiedIR = true;

      // Go through main_task() instructions and don't store argc and argv into argc.addr and argv.addr
      // since the argc and argv are not args to main_task().
      // TODO fix codegen to store argc and argv in task args in lsci_main(), then use
      // liblsci interface to to allow you to get them back within main_task().

      Instruction* argc_store;
      Instruction* argv_store;

      // for each basic block in main_task()
      for(BB = mainTaskFunc->begin() ; BB != mainTaskFunc->end(); ++BB) {

        // for each instruction in the basic block
        for (BasicBlock::InstListType::iterator ii = BB->begin(); ii != BB->end(); ++ii) {

          if ((*ii).getOpcode() == llvm::Instruction::Store) {

            // for each operand in the instruction
            for(unsigned i = 0, e = (*ii).getNumOperands(); i!=e; ++i){

              if ((*ii).getOperand(i)->getName() ==  "argc.addr") {
                //errs() << "found argc.addr\n";
                argc_store = &(*ii);
              }

              if ((*ii).getOperand(i)->getName() ==  "argv.addr") {
                //errs() << "found argv.addr\n";
                argv_store = &(*ii);
              }
            } 
          }
        }
      }
      //errs() << "erasing argc.addr\n";
      argc_store->eraseFromParent();
      //errs() << "erasing argv.addr\n";
      argv_store->eraseFromParent();

      // TODO Go through blocks in main_task and if find call to a func that is a task,
      // substitute with a call to LegionTaskInitFunctionX(lsci_unimesh_t*, char* , char* ).


      // make main() call lsci_main() at the end to do lsci startup stuff

      Function::BasicBlockListType &blocks = F.getBasicBlockList();
      llvm::BasicBlock *newheader = BasicBlock::Create(M.getContext(), "entry");
      blocks.push_back(newheader);
      
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
      CallInst* lsciCall =  CallInst::Create(lsci_mainFunc, args);
      newheader->getInstList().push_back(lsciCall);

      // create ret instruction
      ReturnInst* retInst = ReturnInst::Create(M.getContext(), lsciCall);
      newheader->getInstList().push_back(retInst);

    }
  }

  //errs() << "done with modifying main and main_task in the LegionTaskWRapper pass.\n";

  //M.dump();

  // return true if modified IR
  return modifiedIR;
}

ModulePass* llvm::createLegionTaskWrapperPass(){
  return new LegionTaskWrapper;
}

RegisterPass<LegionTaskWrapper> LegionTaskWrapper("sclegion-task-wrapper", "Generate Legion task wrappers for Scout functions.");

