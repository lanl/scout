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

#include "llvm/IR/DataLayout.h"
#include "llvm/InstVisitor.h"
#include "llvm/Transforms/Utils/Cloning.h"

#ifdef SC_ENABLE_LIB_NVVM
#include "nvvm.h"
#endif
#include <map>
#include <vector>

using namespace llvm;

DoallToPTX::DoallToPTX()
  : ModulePass(ID) {
#ifdef SC_ENABLE_LIB_NVVM
  assert(nvvmInit() == NVVM_SUCCESS);
#endif
}

DoallToPTX::~DoallToPTX(){
#ifdef SC_ENABLE_LIB_NVVM
  assert(nvvmFini() == NVVM_SUCCESS);
#endif
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
#ifndef SC_ENABLE_LIB_NVVM
  PassManager pm;

  const Target *PTXTarget = 0;
  for(TargetRegistry::iterator it = TargetRegistry::begin(),
        ie = TargetRegistry::end(); it != ie; ++it) {
    //if(strcmp(it->getName(), "nvptx") == 0) {
    if(strcmp(it->getName(), "nvptx64") == 0) {
      PTXTarget = &*it;
      break;
    }
  }

  assert(PTXTarget && "NVPTXBackend failed to load!");
  
  std::string AssemblyCode;
  raw_string_ostream StringOut(AssemblyCode);
  formatted_raw_ostream PTXOut(StringOut);

  std::string target_triple = sys::getDefaultTargetTriple();
  
  Triple TargetTriple = Triple(target_triple);
  TargetTriple.setArch(Triple::nvptx64);

  const std::string CPU = "sm_20";
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

  if(const DataLayout* dataLayout = TheTarget->getDataLayout())
    pm.add(new DataLayout(*dataLayout));
  else
    pm.add(new DataLayout(&ptxModule));

  std::auto_ptr<TargetMachine> Target(TheTarget);
  assert(Target.get() && "Could not allocate target machine!");

  const TargetMachine::CodeGenFileType FileType = TargetMachine::CGFT_AssemblyFile;

  const bool DisableVerify = true;

  Target->addPassesToEmitFile(pm, PTXOut, FileType, DisableVerify);

  pm.add(createVerifierPass());
#endif // SC_ENABLE_LIB_NVVM  

  typedef std::vector<GlobalVariable*> GlobalVec;
  GlobalVec globalsToRemove;

  Module::global_iterator itr = ptxModule.global_begin();
  while(itr != ptxModule.global_end()) {
    GlobalVariable* global = &*itr;
    
    Type* type = global->getType();

    if (PointerType* pointerType = dyn_cast<PointerType>(type)) {
      if (pointerType->getAddressSpace() == 0) {
        globalsToRemove.push_back(global);
        global->replaceAllUsesWith(UndefValue::get(type));
      }
    }
    ++itr;
  }

  for(size_t i = 0; i < globalsToRemove.size(); ++i) {
    globalsToRemove[i]->eraseFromParent();
  }

  llvm::NamedMDNode* annotations = ptxModule.getOrInsertNamedMetadata("nvvm.annotations");

  for(Module::iterator fitr = ptxModule.begin(), fitrEnd = ptxModule.end();
      fitr != fitrEnd; ++fitr) {
    Function* f = &*fitr;

    /*
    f->removeFnAttr(Attribute::UWTable|
                    Attribute::StackProtect);
    */

    if (f->getName().startswith("uniRenderall") || 
        f->getName().startswith("uniForall")) {
      
      SmallVector<Value*, 3> av;
      
      av.push_back(f);
      av.push_back(MDString::get(ptxModule.getContext(), "kernel"));
      av.push_back(ConstantInt::get(IntegerType::get(ptxModule.getContext(), 32), 1));
      
      annotations->addOperand(MDNode::get(ptxModule.getContext(), av)); 
    }
  }

  //std::cerr << "----------------- pruned module" << std::endl;
  //ptxModule.dump();
  //std::cerr << "----------------- end pruned module" << std::endl;

  const std::string ptxStrName = "ptxAssembly";

#ifdef SC_ENABLE_LIB_NVVM
  nvvmCU cu;

  assert(nvvmCreateCU(&cu) == NVVM_SUCCESS);
  
  std::string ptxIR;
  raw_string_ostream ptxIROut(ptxIR);
  ptxModule.print(ptxIROut, 0);
  ptxIROut.flush();

  const char* options[] = 
    {"-target=ptx", /*verify*/
     "-arch=compute_20" /*compute_30*/};
  
  assert(nvvmCUAddModule(cu, ptxIR.c_str(), ptxIR.length()) == NVVM_SUCCESS);

  assert(nvvmCompileCU(cu, 2, options) == NVVM_SUCCESS);

  size_t bufferSize;
  assert(nvvmGetCompiledResultSize(cu, &bufferSize) == NVVM_SUCCESS);

  char* outBuffer = (char*)malloc(sizeof(char)*bufferSize);
  assert(nvvmGetCompiledResult(cu, outBuffer) == NVVM_SUCCESS);

  assert(nvvmDestroyCU(&cu) == NVVM_SUCCESS);

  Constant* AssemblyCodeArray =
  ConstantDataArray::getString(cpuModule.getContext(), outBuffer);  

  free(outBuffer);

#else // SC_ENABLE_LIB_NVVM

  pm.run(ptxModule);
  PTXOut.flush();

  Constant *AssemblyCodeArray =
  ConstantDataArray::getString(cpuModule.getContext(), AssemblyCode);  

#endif // SC_ENABLE_LIB_NVVM

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
  
  for(BasicBlockIterator BB = FN->begin(), BB_end = FN->end(); BB != BB_end; ++BB) {
    for(InstIterator inst = BB->begin(), inst_end = BB->end(); inst != inst_end; ++inst) {
      
      if (isa< llvm::CallInst >(inst)) {
        llvm::Function *new_FN = cast< llvm::CallInst >(inst)->getCalledFunction();
        if (!fnSet.count(new_FN->getName())) {
          fnSet.insert(new_FN->getName());
          identifyDependentFns(fnSet, new_FN);
        }
      }
    }
  }
      
}

void DoallToPTX::pruneModule(Module &module, ValueToValueMapTy &valueMap, Function &FN) {

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
  
  for(BasicBlockIterator BB = FN->begin(), BB_end = FN->end(); BB != BB_end; ++BB) {
    for(InstIterator inst = BB->begin(), inst_end = BB->end(); inst != inst_end; ++inst) {
      if (isa< AllocaInst >(inst) && inst->getName().startswith("indvar")) {
        translateVarToTid(cuda, inst, uniform);
        return;
      }
    }
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

namespace{

  class ForAllVisitor : public InstVisitor<ForAllVisitor> {
    
   public:
    ForAllVisitor(Module& module, DoallToPTX::FunctionMDMap& functionMDMap)
      : module(module),
	functionMDMap(functionMDMap) {
    }

    void visitCallInst(CallInst& I) {
      
      Function* f = I.getCalledFunction();

      if (!f) {
        return;
      }

      if (f->getName().startswith("llvm.memcpy")) {
	Value* v = I.getArgOperand(0);
	std::string vs = v->getName().str();
	if (!vs.empty()) {
	  symbolMap[vs] = true;
	} else {
	  ValueMap::iterator itr = valueMap.find(v);
	  if (itr != valueMap.end()) {
	    symbolMap[itr->second] = true;
	  }
	}
      } else if (f->getName().startswith("uniForall") ||
                f->getName().startswith("uniRenderall")) {
        
	SmallVector< llvm::Value *, 3 > args;
	unsigned numArgs = I.getNumArgOperands();
        
	for(unsigned i = 0; i < numArgs; ++i) {
	  Value* arg = I.getArgOperand(i);
	  std::string s = arg->getName().str();
	  
	  SymbolMap::iterator itr = symbolMap.find(s);
	  if (itr != symbolMap.end()) {
	    args.push_back(arg);
	    symbolMap.erase(itr);
	  }
	}

	functionMDMap[f->getName().str()] = MDNode::get(module.getContext(), args);
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
RegisterPass< DoallToPTX > DoallToPTX("Doall-to-PTX", "Generate PTX from forall region and embed as a string literal");
