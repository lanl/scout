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
#include "llvm/Constants.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Instructions.h"
#include "llvm/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <vector>
#include <iostream>
#include <cxxabi.h>

using namespace std;
using namespace llvm;

namespace{

  string demangleName(const string& str){
    int status;
    char* dn = abi::__cxa_demangle(str.c_str(), 0, 0, &status);
    string ret = dn;
    free(dn);
    return ret;
  }

} // end namespace

// based on LLVM function to clone a function, plus some modifications 
// needed for AMDIL   
static void CloneGPUFunctionInto(Function *NewFunc, const Function *OldFunc,
				 ValueToValueMapTy &VMap,
				 bool ModuleLevelChanges,
				 SmallVectorImpl<ReturnInst*> &Returns,
				 const char *NameSuffix = "",
				 ClonedCodeInfo *CodeInfo = 0,
				 ValueMapTypeRemapper *TypeMapper = 0) {
  assert(NameSuffix && "NameSuffix cannot be null!");
  
#ifndef NDEBUG
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
	 E = OldFunc->arg_end(); I != E; ++I)
    assert(VMap.count(I) && "No mapping from source argument specified!");
#endif
  
  if (NewFunc->arg_size() == OldFunc->arg_size())
    NewFunc->copyAttributesFrom(OldFunc);
  else {
    for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
           E = OldFunc->arg_end(); I != E; ++I)
      if (Argument* Anew = dyn_cast<Argument>(VMap[I]))
        Anew->addAttr( OldFunc->getAttributes()
                       .getParamAttributes(I->getArgNo() + 1));
    NewFunc->setAttributes(NewFunc->getAttributes()
                           .addAttr(0, OldFunc->getAttributes()
				    .getRetAttributes()));
    NewFunc->setAttributes(NewFunc->getAttributes()
                           .addAttr(~0, OldFunc->getAttributes()
				    .getFnAttributes()));

  }

  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;

    BasicBlock *CBB = 
      CloneBasicBlock(&BB, VMap, NameSuffix, NewFunc, CodeInfo);

    VMap[&BB] = CBB;

    if (BB.hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function*>(OldFunc),
                                              const_cast<BasicBlock*>(&BB));
      VMap[OldBBAddr] = BlockAddress::get(NewFunc, CBB);                                         
    }

    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  for (Function::iterator BB = cast<BasicBlock>(VMap[OldFunc->begin()]),
         BE = NewFunc->end(); BB != BE; ++BB)
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II)
      RemapInstruction(II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper);
}

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

// based on LLVM function to clone a module, plus some modifications
// needed for AMDIL
Module* DoallToAMDIL::CloneGPUModule(const Module *M, 
				     ValueToValueMapTy &VMap) {
  Module *New = new Module(M->getModuleIdentifier(), M->getContext());
  New->setDataLayout(M->getDataLayout());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());
   
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    New->addLibrary(*I);

  for (Module::const_global_iterator I = M->global_begin(), 
	 E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = new GlobalVariable(*New, 
                                            I->getType()->getElementType(),
                                            I->isConstant(), I->getLinkage(),
                                            (Constant*) 0, I->getName(),
                                            (GlobalVariable*) 0,
                                            I->getThreadLocalMode(),
                                            I->getType()->getAddressSpace());
    GV->copyAttributesFrom(I);
    VMap[I] = GV;
  }

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    // modify the address space of the params to correspond
    // to the OpenCL address spaces
    FunctionType* FT = cast<FunctionType>(I->getType()->getElementType());

    vector<Type*> args;
    for(FunctionType::param_iterator pitr = FT->param_begin(),
	  pitrEnd = FT->param_end(); pitr != pitrEnd; ++pitr){
      const Type* ty = *pitr;
      if(const PointerType* pty = dyn_cast<PointerType>(ty)){
	PointerType* npt = PointerType::get(pty->getElementType(), 1);
	args.push_back(npt);
      }
      else{
	// is it ok to const cast here instead of recreating
	// the type?
	args.push_back(const_cast<Type*>(ty));
      }
    }

    FunctionType* NFT = 
      FunctionType::get(FT->getReturnType(), args, FT->isVarArg());

    Function* NF = 
      Function::Create(NFT,
		       I->getLinkage(), I->getName(), New);  

    NF->copyAttributesFrom(I);
    VMap[I] = NF;
  }

  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    GlobalAlias *GA = new GlobalAlias(I->getType(), I->getLinkage(),
                                      I->getName(), NULL, New);
    GA->copyAttributesFrom(I);
    VMap[I] = GA;
  }
  
  for (Module::const_global_iterator I = M->global_begin(), 
	 E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[I]);
    if (I->hasInitializer())
      GV->setInitializer(MapValue(I->getInitializer(), VMap));
  }

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *F = cast<Function>(VMap[I]);
    if (!I->isDeclaration()) {
      Function::arg_iterator DestI = F->arg_begin();
      for (Function::const_arg_iterator J = I->arg_begin(); J != I->arg_end();
           ++J) {
        DestI->setName(J->getName());
        VMap[J] = DestI++;
      }

      SmallVector<ReturnInst*, 8> Returns;
      CloneGPUFunctionInto(F, I, VMap, true, Returns);
    }
  }

  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    GlobalAlias *GA = cast<GlobalAlias>(VMap[I]);
    if (const Constant *C = I->getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
         E = M->named_metadata_end(); I != E; ++I) {
    const NamedMDNode &NMD = *I;
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapValue(NMD.getOperand(i), VMap));
  }

  return New;
}

bool DoallToAMDIL::runOnModule(Module &m) {
  IRBuilder<> builder(m.getContext());
  
  ValueToValueMapTy valueMap;
  Module* nm = CloneGPUModule(&m, valueMap);  

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

    if(function->getName().startswith("renderall") ||
       function->getName().startswith("forall")){
      
    }
    else{
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

  Function* mainFunction = 0;
  for(Module::iterator itr = m.begin(), itrEnd = m.end();
      itr != itrEnd; ++itr){
    if(itr->getName() == "main"){
      mainFunction = itr;
      break;
    }
  }
  assert(mainFunction);
  BasicBlock& mainEntry = *mainFunction->begin();

  CallInst* initCall = 0;
  Instruction* insertInst;
  int status;
  for(BasicBlock::iterator itr = mainEntry.begin(), itrEnd = mainEntry.end();
      itr != itrEnd; ++itr){
    if(CallInst* callInst = dyn_cast<CallInst>(itr)){
      Function* calledFunc = callInst->getCalledFunction();
      
      string name = demangleName(calledFunc->getName().str());

      if(name.find("__sc_init(") == 0){
	initCall = callInst;
	++itr;
	insertInst = itr;
	break;
      }
    }
  }
  assert(initCall);

  Type* i32Ty = IntegerType::get(m.getContext(), 32);
  Type* i8PtrTy = PointerType::get(IntegerType::get(m.getContext(), 8), 0);
  
  Function* buildFunc = m.getFunction("__sc_opencl_build_program");
  if(!buildFunc){
    vector<Type*> args;    
    args.push_back(i8PtrTy);
    args.push_back(i32Ty);

    FunctionType* retType = 
      FunctionType::get(Type::getVoidTy(m.getContext()), args, false);
    
    buildFunc = Function::Create(retType, 
				 Function::ExternalLinkage,
				 "__sc_opencl_build_program", 
				 &m);
  }

  builder.SetInsertPoint(insertInst);
  Value* bc = builder.CreateBitCast(gv, i8PtrTy, "bitcode");
  builder.CreateCall2(buildFunc, bc, 
		      ConstantInt::get(i32Ty, bitcode.length()));

  //cerr << "-------------- dumping final module" << endl;
  //m.dump();

  return true;
}

char DoallToAMDIL::ID = 1;
RegisterPass<DoallToAMDIL> DoallToAMDIL("Doall-to-AMDIL", "Generate LLVM bitcode for GPU kernels and embed as a global value.");
