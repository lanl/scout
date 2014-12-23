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
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <llvm/IR/ValueSymbolTable.h>
#include "legion/lsci.h"

using namespace llvm;

char LegionTaskWrapper::ID = 1;

LegionTaskWrapper::LegionTaskWrapper() : ModulePass(LegionTaskWrapper::ID) {}

void printInst(const Instruction* inst) {
  errs() << inst->getOpcodeName();
  errs() << " ";
  for(unsigned i = 0, e = inst->getNumOperands(); i!=e; ++i){
    errs() << inst->getOperand(i)->getName();
    errs() << " ";
  }
  errs() << "\n";
}
  
// return true if leafInst is a use of rootInst 
bool isUseOf(Value* leafVal, Value* rootVal) {
  for(Value::user_iterator i = rootVal->user_begin(), ie = rootVal->user_end(); i!=ie; ++i){
    Instruction *vi = dyn_cast<Instruction>(*i);
    printInst(vi);
    if (vi != rootVal) {
      if (vi) {
        errs() << "use: " << vi->getName() << "\n";
      } else {
        errs() << "use is NULL!\n";
      }
      if (vi == leafVal) {
        errs() << "Found use connection!\n";
        return true;
      } else if (isUseOf(leafVal, vi)) {
        return true;
      }
    }
  }
  return false;
}

// check if one of the operands that defined this leafInst was the rootInst
bool isDefinedFrom(Instruction* leafInst, Instruction* rootInst) {
  for(User::op_iterator i = leafInst->op_begin(), ie = leafInst->op_end(); i!=ie; ++i){
    Instruction *vi = dyn_cast<Instruction>(*i);
    //printInst(vi);
    if (vi) {
      //errs() << "vi: " << vi->getName() << "\n";
    } else {
      //errs() << "vi is NULL!\n";
    }
    if (vi == rootInst) {
      return true;
    } 
  }
  return false;
}

bool LegionTaskWrapper::runOnModule(Module &M) {

  bool modifiedIR = false;

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

      Instruction* argc_store = NULL;
      Instruction* argv_store = NULL;

      // for each basic block in main_task()
      for(BB = mainTaskFunc->begin() ; BB != mainTaskFunc->end(); ++BB) {

        // for each instruction in the basic block
        for (BasicBlock::InstListType::iterator ii = BB->begin(); ii != BB->end(); ++ii) {

          if ((*ii).getOpcode() == llvm::Instruction::Store) {

            Instruction &storeInst = (*ii);

            // for each operand in the instruction
            for(unsigned i = 0, e = storeInst.getNumOperands(); i!=e; ++i){

              if (storeInst.getOperand(i)->getName() ==  "argc.addr") {
                //errs() << "found argc.addr\n";
                argc_store = &storeInst;
              }

              if (storeInst.getOperand(i)->getName() ==  "argv.addr") {
                //errs() << "found argv.addr\n";
                argv_store = &storeInst;
              }
            } 
          }
        }
      }
      //errs() << "erasing argc.addr\n";
      argc_store->eraseFromParent();
      //errs() << "erasing argv.addr\n";
      argv_store->eraseFromParent();

      // In main_task() we need to subst a call to LegionTaskInitFunctionX(lsci_unimesh_t*, char* , char* ).
      // instead of the original function call, but in order to do that, you need to be able to pass it the
      // context and runtime, which we need to get first from the task_args that have been passed to main_task()

      // Go to beginning of main_task and get lsci runtime and context
      BasicBlock &firstBlock = mainTaskFunc->front();
      llvm::Instruction& firstInst = firstBlock.front();
      // inserts before first instruction
      builder.SetInsertPoint(&firstInst);
      
      //mainTaskFunc->dump();

      // get lsci_task_args_t variable
      Function::arg_iterator arg_iter = mainTaskFunc->arg_begin();
      
      Value* task = arg_iter++;
      Value* regions = arg_iter++;
      Value* numRegions = arg_iter++;
      Value* context = arg_iter++;
      Value* runtime = arg_iter;

      // go through instructions in this block and look for loads of @__scrt_legion_context
      // and replace with my context address.  Same with runtime.

      // for each instruction in the basic block
      for (BasicBlock::InstListType::iterator ii = firstBlock.begin(); ii != firstBlock.end(); ++ii) {

          // if a load
          if ((*ii).getOpcode() == llvm::Instruction::Load) {

            Instruction &loadInst = *ii;

            // if the address being loaded is @__scrt_legion_context, then remove the instruction
            // and replace all uses of the result with context
            if (loadInst.getOperand(0)->getName() == "__scrt_legion_context" ) {
                loadInst.replaceAllUsesWith(context);
            } 
            // if the address being loaded is @__scrt_legion_runtime, then remove the instruction
            // and replace all uses of the result with runtime
            if (loadInst.getOperand(0)->getName() == "__scrt_legion_runtime" ) {
                loadInst.replaceAllUsesWith(runtime);
            } 
          }
        }

      //M.dump();

      // Go through blocks in main_task and if find call to a func that is a task,
      // substitute with a call to LegionTaskInitFunctionX(lsci_unimesh_t*, char* context, char* runtime ).

      // for each basic block in main_task()
      for(BB = mainTaskFunc->begin() ; BB != mainTaskFunc->end(); ++BB) {

        std::vector<Instruction*> instToErase;
 
        // for each instruction in the basic block
        for (BasicBlock::InstListType::iterator ii = BB->begin(); ii != BB->end(); ++ii) {

          if ((*ii).getOpcode() == llvm::Instruction::Call) {

            CallInst& callInst = cast <CallInst> (*ii);

            if (callInst.getNumOperands() == 0) break;

            Value* argVal = callInst.getArgOperand(0);

            // retrieve corresponding function for this call
            llvm::Function *calledFN = cast< llvm::Function > (callInst.getCalledFunction());
            //errs() << "Looking at call of " << calledFN->getName() << "\n";
        
            // TODO check if it is a call to a task and substitute in a call to the correct LegionTaskInit function;
           
            // get function name and look it up to see if it is a task. 
            NamedMDNode* NMDN = M.getOrInsertNamedMetadata("scout.tasks");

            Value* lsciUnimeshVal = nullptr;

            // Go through each MDNode in the NamedMDNodes and search for metadata related to task function
            // Metadata for scout.tasks is in the form of a small vector of 3 Value*:  taskID, taskFunc and taskInit
            for (unsigned i = 0, e = NMDN->getNumOperands(); i != e; ++i) {


              // get the ith MDNode operand
              //errs() << "Looking at: " << i << " MDNode operand\n";
              MDNode *MDN = cast< MDNode >(NMDN->getOperand(i));

              // 1st Operand  of MDNodes is a function ptr
              Function *FN = 
                cast < Function > (cast<ValueAsMetadata>(MDN->getOperand(1))->getValue());

              if (FN == calledFN) {
                //errs() << "This is a task\n";

                // get metadata connecting mesh alloc and lsci_unimesh_t alloc
                NamedMDNode* lsciNMDN = M.getOrInsertNamedMetadata("scout.lscimeshmd");

                for (unsigned i = 0, e = lsciNMDN->getNumOperands(); i != e; ++i) {

                  // get the ith MDNode operand of the NamedMDNode 
                  MDNode *lsciMDN = cast< MDNode >(lsciNMDN->getOperand(i));

                  // 0th Operand  of MDNodes is name of mesh alloc
                  MDString* allocMDStr = cast < MDString > (lsciMDN->getOperand(0));
                  StringRef allocStr = allocMDStr->getString();
                  //errs() << "allocStr:" << allocStr << "\n";

                  //MDString* lsciallocMDStr = cast < MDString > (lsciMDN->getOperand(1));
                  //StringRef lsciallocStr = lsciallocMDStr->getString();
                  //errs() << "lsciallocStr:" << lsciallocStr << "\n";

                  // if task argument string is in the def-use chain of the metadata string value,
                  // then get the lsci_unimesh_t value
                  // if (allocStr.equals(argStr)) 
                  Value* allocVal = mainTaskFunc->getValueSymbolTable().lookup(allocStr);
                  Instruction* argInst = dyn_cast<Instruction>(argVal);
                  //errs() << "argVal: " << argVal->getName() << "\n";
                  //errs() << "argInst: " << argInst->getName() << "\n";
                  Instruction* allocInst = dyn_cast<Instruction>(allocVal);
                  //errs() << "allocVal: " << allocVal->getName() << "\n";
                  //errs() << "allocInst: " << allocInst->getName() << "\n";

                  //if (isUseOf(argVal, allocVal)) 
                  if (isDefinedFrom(argInst, allocInst)) {
                    //errs() << "Found match btwn task arg val and metadata str:" << argVal->getName() << "\n";

                    // 1st Operand  of MDNodes is name of lsci_unimesh_t alloc
                    MDString* lsciUnimeshMDStr = cast < MDString > (lsciMDN->getOperand(1));
                    StringRef lsciUnimeshStr = lsciUnimeshMDStr->getString();
                    //errs() << "Found lsci_unimesh_t str:" << lsciUnimeshStr << "\n";

                    // lookup lsci_unimesh_t alloc name to get value
                    lsciUnimeshVal = mainTaskFunc->getValueSymbolTable().lookup(lsciUnimeshStr);
                    //lsciUnimeshVal = F.getValueSymbolTable().lookup(lsciUnimeshStr);
                  }

                  if (lsciUnimeshVal) break;
                }

                assert (lsciUnimeshVal && "no val for lsci_unimesh_t");

                // create the arguments to the call to the LegionTaskInitFunction
                // (lsci_mesh, any other args, then context and runtime)

                std::vector<llvm::Value*> Args = {}; 
                Args.push_back(lsciUnimeshVal);

                // Go through each operand after mesh to original call to task and add it
                // Also don't want last operand, because that is the callee.

                unsigned numArgs = callInst.getNumOperands(); 
                for (unsigned i = 1; i < numArgs-1; i++) {
                  Args.push_back(callInst.getArgOperand(i));
                }

                Args.push_back(context);
                Args.push_back(runtime);

                // replace call to function with call to LegionTaskInitFunction
                // which we can get rom the metadata
                Function *legionTaskInitFN = cast < Function > (cast<ValueAsMetadata>(MDN->getOperand(2))->getValue());

                builder.SetInsertPoint(&callInst);
                //errs() << "create call\n";
                builder.CreateCall(legionTaskInitFN, ArrayRef<llvm::Value*> (Args));
                instToErase.push_back(&callInst);
              } 
              if (lsciUnimeshVal) break;
            }
          }
        }

        // now that we've iterated through the instructions in this block, remove the original task function calls
        for (std::vector<Instruction*>::iterator iter = instToErase.begin(); iter != instToErase.end(); ++iter) {
          Instruction* instr = *iter;
          instr->eraseFromParent();
        }
      }

      // make main() call lsci_main() at the end to do lsci startup stuff

      Function::BasicBlockListType &blocks = F.getBasicBlockList();
      llvm::BasicBlock *newheader = BasicBlock::Create(M.getContext(), "entry");
      blocks.push_back(newheader);

      // create call instruction
      arg_iter = F.arg_begin();
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


  // return true if modified IR
  return modifiedIR;
}

ModulePass* llvm::createLegionTaskWrapperPass(){
  return new LegionTaskWrapper;
}

RegisterPass<LegionTaskWrapper> LegionTaskWrapper("sclegion-task-wrapper", "Generate Legion task wrappers for Scout functions.");

