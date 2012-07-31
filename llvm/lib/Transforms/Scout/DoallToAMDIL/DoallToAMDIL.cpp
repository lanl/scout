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
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include <vector>
#include <iostream>

using namespace std;
using namespace llvm;

DoallToAMDIL::DoallToAMDIL()
  : ModulePass(ID){
}

DoallToAMDIL::~DoallToAMDIL(){

}

ModulePass *createDoallToAMDILPass() {
  return new DoallToAMDIL();
}

void DoallToAMDIL::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<DominatorTree>();
}

const char *DoallToAMDIL::getPassName() const {
  return "Doall-to-AMDIL";
}

bool DoallToAMDIL::runOnModule(Module &m) {
  ValueToValueMapTy valueMap;
  Module* nm = CloneModule(&m, valueMap);  

  //cerr << "------------------ before pruning" << endl;
  //nm->dump();
  
  typedef vector<GlobalVariable*> GlobalVec;
  GlobalVec globalsToRemove;

  Module::global_iterator itr = nm->global_begin();
  while(itr != nm->global_end()){
    GlobalVariable* global = &*itr;
    
    Type* type = global->getType();

    if(PointerType* pointerType = dyn_cast<PointerType>(type)){
      if(pointerType->getAddressSpace() == 0){
        globalsToRemove.push_back(global);
        global->replaceAllUsesWith(UndefValue::get(type));
      }
    }
    ++itr;
  }
  
  for(size_t i = 0; i < globalsToRemove.size(); ++i){
    globalsToRemove[i]->eraseFromParent();
  }
  
  typedef vector<Function*> FunctionVec;
  FunctionVec functionsToRemove;

  for(Module::iterator fitr = nm->begin(),
        fitrEnd = nm->end(); fitr != fitrEnd; ++fitr){
    Function* function = &*fitr;

    /*
      f->removeFnAttr(Attribute::UWTable|
      Attribute::StackProtect);
    */

    if(!(function->getName().startswith("renderall") || 
         function->getName().startswith("forall"))){
    
      Type* type = function->getType();
         
      function->replaceAllUsesWith(UndefValue::get(type));

      functionsToRemove.push_back(function);
    }
  }
  
  for(size_t i = 0; i < functionsToRemove.size(); ++i){
    functionsToRemove[i]->eraseFromParent();
  }

  //cerr << "------------------ after pruning" << endl;
  //nm->dump();

  string bitcode;
  raw_string_ostream bs(bitcode);
  formatted_raw_ostream fbs(bs);

  WriteBitcodeToFile(nm, fbs);

  fbs.flush();

  Constant *bitcodeData =
  ConstantDataArray::getString(m.getContext(), bitcode); 

  GlobalVariable* gv = 
    new GlobalVariable(m,
                       bitcodeData->getType(),
                       true,
                       GlobalValue::PrivateLinkage,
                       bitcodeData, "gpu.module");

  //cout << "------------------- final module" << endl;
  //m.dump();

  return true;
}

char DoallToAMDIL::ID = 1;
RegisterPass<DoallToAMDIL> DoallToAMDIL("Doall-to-AMDIL", "Generate LLVM bitcode for GPU kernels and embed as a global value.");
