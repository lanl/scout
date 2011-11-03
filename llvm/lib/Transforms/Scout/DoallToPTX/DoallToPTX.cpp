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

#include "compiler/DoallToPTX/DoallToPTX.h"

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

  std::string target_triple = sys::getHostTriple();
  Triple TargetTriple = Triple(target_triple);
  TargetTriple.setArch(Triple::x86);

  const std::string CPU = "";
  const std::string featuresStr = "";
  TargetMachine *TheTarget =
    PTXTarget->createTargetMachine(TargetTriple.getTriple(),
                                   CPU,
                                   featuresStr);

  std::auto_ptr< TargetMachine > Target(TheTarget);
  assert(Target.get() && "Could not allocate target machine!");

  const TargetMachine::CodeGenFileType FileType = TargetMachine::CGFT_AssemblyFile;
  const CodeGenOpt::Level Lvl = CodeGenOpt::Aggressive;
  const bool DisableVerify = false;

  Target->addPassesToEmitFile(pm, PTXOut, FileType, Lvl, DisableVerify);

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

void DoallToPTX::pruneModule(Module &module, ValueToValueMapTy &valueMap,
                             Function &func) {
  PassManager pm;

  ValueSet users;
  users.insert(valueMap.find(&func)->second);

  typedef llvm::Function::iterator BasicBlockIterator;
  typedef llvm::BasicBlock::iterator InstIterator;
  BasicBlockIterator BB = func.begin(), BB_end = func.end();
  for(; BB != BB_end; ++BB) {

    InstIterator inst = BB->begin(), inst_end = BB->end();
    for(; inst != inst_end; ++inst) {
      users.insert(&*inst);
    }
  }

  typedef llvm::Module::FunctionListType FuncList;
  typedef FuncList::iterator FuncListIterator;
  FuncList &funcs = module.getFunctionList();
  FuncListIterator it = funcs.begin(), it_end = funcs.end();
  for( ; it != it_end; ) {
    FuncListIterator curr = it++;
    curr->setLinkage(GlobalValue::ExternalLinkage);
    if(!users.count(&*curr)) {
      curr->replaceAllUsesWith(UndefValue::get(curr->getType()));
      curr->eraseFromParent();
    }
  }

  pm.add(createVerifierPass());
  pm.run(module);
}

void DoallToPTX::generatePTXHandler(CudaDriver &cuda, Module &module,
                                    std::string name, GlobalValue *ptxAsm) {
  Function *ptxHandler = module.getFunction(name);

  // Remove the body of function.
  ptxHandler->deleteBody();

  // Generate new body of function. This function will be the
  // handler for CUDA-related API calls.
  BasicBlock *entryBB = BasicBlock::Create(module.getContext(), "entry", ptxHandler);
  cuda.setInsertPoint(entryBB);

  cuda.create(ptxHandler, ptxAsm);

  ReturnInst::Create(module.getContext(), entryBB);
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

bool DoallToPTX::runOnModule(Module &M) {
  // Interface to CUDA Driver API
  IRBuilder<> Builder(getGlobalContext());
  CudaDriver cuda(M, Builder, true);

  cuda.initialize();
  cuda.finalize();

  NamedMDNode *NMDN = M.getNamedMetadata("scout.kernels");
  for(unsigned i = 0, e = NMDN->getNumOperands(); i < e; ++i) {
    Function *Fn = cast< Function >(NMDN->getOperand(i)->getOperand(0));

    // Clone module.
    ValueToValueMapTy valueMap;
    Module *ptxModule(CloneModule(&M, valueMap));

    // Remove instructions unrelated to Fn.
    pruneModule(*ptxModule, valueMap, *Fn);

    // Embed ptx string in module.
    GlobalValue *ptxAsm = embedPTX(*ptxModule, M);

    // Insert CUDA API calls.
    generatePTXHandler(cuda, M, Fn->getName().str(), ptxAsm);
  }
  return true;
}

char DoallToPTX::ID = 1;
RegisterPass< DoallToPTX > DoallToPTX("Doall-to-PTX", "Generate PTX assembly from doall region and embed assembly as a string literal");
