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

#include "llvm/Transforms/Scout/DoallToPTX/DoallToPTX.h"

#include "llvm/Support/InstVisitor.h"

#include <map>

using namespace llvm;

DoallToPTX::DoallToPTX()
  : ModulePass(ID)
{
}

ModulePass *createDoallToPTXPass() {
  return new DoallToPTX();
}

void DoallToPTX::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTree>();
}

const char *DoallToPTX::getPassName() const {
  return "Doall-to-PTX";
}

GlobalValue *DoallToPTX::embedPTX(Module &ptxModule, Module &cpuModule) {
  PassManager pm;
  pm.add(createVerifierPass());
  pm.run(ptxModule);

  const Target *PTXTarget = 0;
  for(TargetRegistry::iterator it = TargetRegistry::begin(),
        ie = TargetRegistry::end(); it != ie; ++it) {
    if(strcmp(it->getName(), "simple-ptx") == 0) {
      PTXTarget = &*it;
      break;
    }
  }

  assert(PTXTarget && "PTXBackend failed to load!");

  std::string AssemblyCode;
  raw_string_ostream StringOut(AssemblyCode);
  formatted_raw_ostream PTXOut(StringOut);

  std::string target_triple = sys::getDefaultTargetTriple();
  
  Triple TargetTriple = Triple(target_triple);
  TargetTriple.setArch(Triple::x86);

  const std::string CPU = "";
  const std::string featuresStr = "";
  const CodeGenOpt::Level Lvl = CodeGenOpt::Aggressive;

  TargetMachine *TheTarget =
    PTXTarget->createTargetMachine(TargetTriple.getTriple(),
                                   CPU,
                                   featuresStr,
				   TargetOptions(),
				   Reloc::Default,
				   CodeModel::Default,
				   Lvl);

  std::auto_ptr< TargetMachine > Target(TheTarget);
  assert(Target.get() && "Could not allocate target machine!");

  const TargetMachine::CodeGenFileType FileType = TargetMachine::CGFT_AssemblyFile;

  const bool DisableVerify = false;

  Target->addPassesToEmitFile(pm, PTXOut, FileType, DisableVerify);

  pm.add(createVerifierPass());
  pm.run(ptxModule);
  PTXOut.flush();

  std::string ptxStrName = "ptxAssembly";

  Constant *AssemblyCodeArray =
    ConstantArray::get(cpuModule.getContext(), AssemblyCode);
  
  return new GlobalVariable(cpuModule,
                            AssemblyCodeArray->getType(),
                            true,
                            GlobalValue::PrivateLinkage,
                            AssemblyCodeArray,
                            ptxStrName);
}


void DoallToPTX::identifyDependentFns(FnSet &fnSet, llvm::Function *FN) {
  typedef llvm::Function::iterator BasicBlockIterator;
  typedef llvm::BasicBlock::iterator InstIterator;
  for(BasicBlockIterator BB = FN->begin(), BB_end = FN->end(); BB != BB_end; ++BB)
    for(InstIterator inst = BB->begin(), inst_end = BB->end(); inst != inst_end; ++inst)
      if(isa< llvm::CallInst >(inst)) {
        llvm::Function *new_FN = cast< llvm::CallInst >(inst)->getCalledFunction();
        if(!fnSet.count(new_FN->getName())) {
          fnSet.insert(new_FN->getName());
          identifyDependentFns(fnSet, new_FN);
        }
      }
}

void DoallToPTX::pruneModule(Module &module, ValueToValueMapTy &valueMap,
                             Function &FN) {
  PassManager pm;

  FnSet fnSet;
  fnSet.insert(FN.getName());
  identifyDependentFns(fnSet, &FN);

  typedef llvm::Module::FunctionListType FuncList;
  typedef FuncList::iterator FuncListIterator;
  FuncList &FNs = module.getFunctionList();
  for(FuncListIterator it = FNs.begin(), it_end = FNs.end(); it != it_end; ) {
    FuncListIterator curr = it++;
    curr->setLinkage(GlobalValue::ExternalLinkage);
    if(!fnSet.count(curr->getName())) {
      curr->replaceAllUsesWith(UndefValue::get(curr->getType()));
      curr->eraseFromParent();
    }
  }

  pm.add(createVerifierPass());
  pm.run(module);
}

void DoallToPTX::translateVarToTid(CudaDriver &cuda, llvm::Instruction *inst, bool uniform) {
  llvm::BasicBlock *BB = inst->getParent();
  llvm::Function *FN = BB->getParent();
  llvm::Module *module = FN->getParent();
  IRBuilder<> Builder(module->getContext());
  PTXDriver ptx(*module, Builder);

  ptx.setInsertPoint(BB, BB->begin());

  llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());

  llvm::Value *row = ptx.insertGetGlobalThreadIdx(PTXDriver::X);
  llvm::Value *col = ptx.insertGetGlobalThreadIdx(PTXDriver::Y);
  llvm::Value *blockDimX  = Builder.CreateZExt(ptx.insertGetGridDim(PTXDriver::X), i32Ty);
  llvm::Value *threadDimX = Builder.CreateZExt(ptx.insertGetBlockDim(PTXDriver::X), i32Ty);
  col = Builder.CreateMul(col, Builder.CreateMul(blockDimX, threadDimX));
  llvm::Value *tid = Builder.CreateAdd(row, col, "threadidx");

  std::vector< Instruction * > uses;
  typedef llvm::Value::use_iterator UseIterator;
  for(UseIterator use = inst->use_begin(), end = inst->use_end(); use != end; ++use) {
    uses.push_back(dyn_cast< Instruction >(*use));
  }

  for(unsigned i = 0, e = uses.size(); i < e; ++i) {
    uses[i]->replaceAllUsesWith(tid);
    uses[i]->eraseFromParent();
  }

  if(!uniform) {
    llvm::Value *total = llvm::ConstantInt::get(i32Ty, cuda.getLinearizedMeshSize());
    llvm::Value *cond = Builder.CreateICmpUGE(tid, total, "cmp");

    llvm::Instruction *inst = cast< llvm::Instruction >(cond);
    llvm::BasicBlock *parent = inst->getParent();

    typedef llvm::BasicBlock::iterator BBIterator;
    BBIterator it = parent->begin(), end = parent->end();
    for( ; it != end; ++it)
      if(it->getName() == inst->getName())
        break;
    ++it;

    llvm::BasicBlock *child  = parent->splitBasicBlock(*it, "continue");
    llvm::BasicBlock *exit = llvm::BasicBlock::Create(module->getContext(),
                                                      "thread_guard", FN, child);
    parent = child->getSinglePredecessor();
    parent->getTerminator()->eraseFromParent();

    Builder.SetInsertPoint(parent);
    Builder.CreateCondBr(cond, exit, child);
    Builder.SetInsertPoint(exit);
    Builder.CreateRetVoid();
  }
}

void DoallToPTX::setGPUThreading(CudaDriver &cuda, llvm::Function *FN, bool uniform) {
  llvm::SmallVector< llvm::Value *, 3 > indvars;
  typedef llvm::Function::iterator BasicBlockIterator;
  typedef llvm::BasicBlock::iterator InstIterator;
  for(BasicBlockIterator BB = FN->begin(), BB_end = FN->end(); BB != BB_end; ++BB)
    for(InstIterator inst = BB->begin(), inst_end = BB->end(); inst != inst_end; ++inst)
      if(isa< AllocaInst >(inst) && inst->getName().startswith("indvar")) {
        translateVarToTid(cuda, inst, uniform);
        return;
      }
}

void DoallToPTX::generatePTXHandler(CudaDriver &cuda, Module &module,
                                    std::string name, GlobalValue *ptxAsm,
				    Value* meshName) {
  Function *ptxHandler = module.getFunction(name);

  // Remove the body of function.
  ptxHandler->deleteBody();

  // Generate new body of function. This function will be the
  // handler for CUDA-related API calls.
  BasicBlock *entryBB = BasicBlock::Create(module.getContext(), "entry", ptxHandler);
  cuda.setInsertPoint(entryBB);

  FunctionMDMap::iterator itr = functionMDMap.find(name);
  assert(itr != functionMDMap.end());
  cuda.create(ptxHandler, ptxAsm, meshName, itr->second);

  //ReturnInst::Create(module.getContext(), entryBB);
}

// Copied from llvm/lib/Transforms/Utils/CloneModule.cpp.
Module *DoallToPTX::CloneModule(const Module *M, ValueToValueMapTy &VMap) {
  // First off, we need to create the new module.
  Module *New = new Module(M->getModuleIdentifier(), M->getContext());
  New->setDataLayout(M->getDataLayout());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());

  // Copy all of the dependent libraries over.
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    New->addLibrary(*I);

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = new GlobalVariable(*New,
                                            I->getType()->getElementType(),
                                            false,
                                            GlobalValue::ExternalLinkage, 0,
                                            I->getName());
    GV->setAlignment(I->getAlignment());
    VMap[I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *NF =
      Function::Create(cast<FunctionType>(I->getType()->getElementType()),
                       GlobalValue::ExternalLinkage, I->getName(), New);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;
  }

  // Loop over the aliases in the module
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    VMap[I] = new GlobalAlias(I->getType(), GlobalAlias::ExternalLinkage,
                                  I->getName(), NULL, New);

  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[I]);
    if (I->hasInitializer())
      GV->setInitializer(MapValue(I->getInitializer(), VMap));
    GV->setLinkage(I->getLinkage());
    GV->setThreadLocal(I->isThreadLocal());
    GV->setConstant(I->isConstant());
  }

  // Similarly, copy over function bodies now...
  //
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *F = cast<Function>(VMap[I]);
    if (!I->isDeclaration()) {
      Function::arg_iterator DestI = F->arg_begin();
      for (Function::const_arg_iterator J = I->arg_begin(); J != I->arg_end();
           ++J) {
        DestI->setName(J->getName());
        VMap[J] = DestI++;
      }

      SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
      CloneFunctionInto(F, I, VMap, true, Returns);
    }

    F->setLinkage(I->getLinkage());
  }

  // And aliases
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    GlobalAlias *GA = cast<GlobalAlias>(VMap[I]);
    GA->setLinkage(I->getLinkage());
    if (const Constant *C = I->getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  // And named metadata....
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
         E = M->named_metadata_end(); I != E; ++I) {
    const NamedMDNode &NMD = *I;
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapValue(NMD.getOperand(i), VMap));
  }

  return New;
}

namespace{

  class ForAllVisitor : public InstVisitor<ForAllVisitor>{
  public:
    ForAllVisitor(Module& module, DoallToPTX::FunctionMDMap& functionMDMap)
      : module(module),
	functionMDMap(functionMDMap){

    }

    void visitCallInst(CallInst& I){
      Function* f = I.getCalledFunction();

      if(f->getName().startswith("llvm.memcpy")){
	Value* v = I.getArgOperand(0);
	std::string vs = v->getName().str();
	if(!vs.empty()){
	  symbolMap[vs] = true;
	}
	else{
	  ValueMap::iterator itr = valueMap.find(v);
	  if(itr != valueMap.end()){
	    symbolMap[itr->second] = true;
	  }
	}
      }
      else if(f->getName().startswith("forall") ||
	 f->getName().startswith("renderall")){
	SmallVector< llvm::Value *, 3 > args;
	unsigned numArgs = I.getNumArgOperands();
	for(unsigned i = 0; i < numArgs; ++i){
	  Value* arg = I.getArgOperand(i);
	  std::string s = arg->getName().str();
	  
	  SymbolMap::iterator itr = symbolMap.find(s);
	  if(itr != symbolMap.end()){
	    args.push_back(arg);
	    symbolMap.erase(itr);
	  }
	}

	functionMDMap[f->getName().str()] = 
	  MDNode::get(module.getContext(), args);
      }
    }
    
    void visitStoreInst(StoreInst& I){
      std::string s = I.getPointerOperand()->getName().str();
      if(!s.empty()){
	symbolMap[s] = true;
      }
    }

    void visitBitCastInst(BitCastInst& I){
      std::string vs = I.getOperand(0)->getName().str();
      if(!vs.empty()){
	valueMap[&I] = vs;
      }
    }

    typedef std::map<std::string, bool> SymbolMap;
    typedef std::map<Value*, std::string> ValueMap;

    SymbolMap symbolMap;
    ValueMap valueMap;
    Module& module;
    DoallToPTX::FunctionMDMap& functionMDMap;
  };

} // end namespace

bool DoallToPTX::runOnModule(Module &M) {
  // Interface to CUDA Driver API
  IRBuilder<> Builder(M.getContext());
  CudaDriver cuda(M, Builder, true);

  for(Module::iterator itr = M.begin(), itrEnd = M.end(); 
      itr != itrEnd; ++itr){
    Function& f = *itr;
    ForAllVisitor visitor(M, functionMDMap);
    visitor.visit(f);
  }

  NamedMDNode *NMDN = M.getNamedMetadata("scout.kernels");
  for(unsigned i = 0, e = NMDN->getNumOperands(); i < e; i+=1) {

    MDNode *node = cast< MDNode >(NMDN->getOperand(i)->getOperand(0));
    Function *FN = cast< Function >(node->getOperand(0));

    llvm::SmallVector< llvm::ConstantInt *, 3 > args;
    node = cast< MDNode >(NMDN->getOperand(i)->getOperand(1));
    for(unsigned j = 0, f = node->getNumOperands(); j < f; ++j) {
      args.push_back(cast< ConstantInt >(node->getOperand(j)));
    }
    cuda.setFnArgAttributes(args);

    args.clear();
    node = cast< MDNode >(NMDN->getOperand(i)->getOperand(2));
    for(unsigned j = 0, f = node->getNumOperands(); j < f; ++j) {
      args.push_back(cast< ConstantInt >(node->getOperand(j)));
    }
    cuda.setDimensions(args);

    node = cast< MDNode >(NMDN->getOperand(i)->getOperand(3));

    Value* meshName = node->getOperand(0);

    llvm::SmallVector< llvm::Value *, 3 > meshFieldArgs;
    node = cast< MDNode >(NMDN->getOperand(i)->getOperand(4));
    for(unsigned j = 0, f = node->getNumOperands(); j < f; ++j) {
      meshFieldArgs.push_back(cast< llvm::Value >(node->getOperand(j)));
    }
    cuda.setMeshFieldNames(meshFieldArgs);

    // Clone module.
    ValueToValueMapTy valueMap;
    Module *ptxModule(CloneModule(&M, valueMap));

    // Remove instructions unrelated to FN.
    pruneModule(*ptxModule, valueMap, *FN);

    // set # threads, return # threads == # elements.
    bool uniform = cuda.setGridAndBlockSizes();

    // Substitute PTX thread id's for induction variables.
    setGPUThreading(cuda, ptxModule->getFunction(FN->getName()), uniform);

    // Embed ptx string in module.
    GlobalValue *ptxAsm = embedPTX(*ptxModule, M);

    // Insert CUDA API calls.
    generatePTXHandler(cuda, M, FN->getName().str(), ptxAsm, meshName);
  }
  return true;
}

char DoallToPTX::ID = 1;
RegisterPass< DoallToPTX > DoallToPTX("Doall-to-PTX", "Generate PTX assembly from doall region and embed assembly as a string literal");
