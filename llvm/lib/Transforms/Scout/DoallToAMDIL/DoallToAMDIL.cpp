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

#include <vector>
#include <iostream>
#include <cxxabi.h>

#include "llvm/Constants.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Instructions.h"
#include "llvm/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/InstVisitor.h"

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

  class ForAllVisitor : public InstVisitor<ForAllVisitor>{
  public:
    ForAllVisitor(Module& module, DoallToAMDIL::FunctionMDMap& functionMDMap)
      : module(module),
        functionMDMap(functionMDMap){

    }

    void visitCallInst(CallInst& I){
      Function* f = I.getCalledFunction();

      if(!f){
        return;
      }

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
    DoallToAMDIL::FunctionMDMap& functionMDMap;
  };

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

llvm::Module* DoallToAMDIL::createGPUModule(const llvm::Module& m){
  ValueToValueMapTy valueMap;
  Module* nm = CloneGPUModule(&m, valueMap);

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

    if(function->getName().startswith("__OpenCL_renderall") ||
       function->getName().startswith("__OpenCL_forall")){

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

  return nm;
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

    GlobalVariable* GV;
    if(I->getName().startswith("scout.gpu")){
      GV = new GlobalVariable(*New,
			      I->getType()->getElementType(),
			      I->isConstant(), GlobalValue::InternalLinkage,
			      (Constant*) 0, I->getName(),
			      (GlobalVariable*) 0,
			      I->getThreadLocalMode(),
			      2);
    }
    else{
      GV = new GlobalVariable(*New,
			      I->getType()->getElementType(),
			      I->isConstant(), I->getLinkage(),
			      (Constant*) 0, I->getName(),
			      (GlobalVariable*) 0,
			      I->getThreadLocalMode(),
			      I->getType()->getAddressSpace());
      GV->copyAttributesFrom(I);
    }


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

    string fullName = "__OpenCL_" + I->getName().str() + "_kernel";

    Function* NF = 
      Function::Create(NFT,
		       I->getLinkage(), fullName, New);  

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

  Type* i32Ty = IntegerType::get(m.getContext(), 32);
  Type* i8PtrTy = PointerType::get(IntegerType::get(m.getContext(), 8), 0);

  for(Module::iterator itr = m.begin(), itrEnd = m.end();
      itr != itrEnd; ++itr){
    Function& f = *itr;
    ForAllVisitor visitor(m, functionMDMap);
    visitor.visit(f);
  }

  //cerr << "----------- dumping original module" << endl;
  //m.dump();

  Module* gm = createGPUModule(m);
  gm->setDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:"
		    "64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:"
		    "64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:"
		    "256-v256:256:256-v512:512:512-v1024:1024:1024-a0:0:"
		    "64-f80:32:32");
  gm->setTargetTriple("amdil-pc-amdopencl");

  ArrayType* sgvArrayType =
    ArrayType::get(IntegerType::get(gm->getContext(), 8), 1);

  ArrayType* fgvArrayType =
    ArrayType::get(IntegerType::get(gm->getContext(), 8), 1);

  ArrayType* lvgvArrayType =
    ArrayType::get(IntegerType::get(gm->getContext(), 8), 0);

  ArrayType* rvgvArrayType =
    ArrayType::get(IntegerType::get(gm->getContext(), 8), 0);

  vector<Type*> annotationStructFields;

  for(size_t i = 0; i < 5; ++i){
    annotationStructFields.push_back(i8PtrTy);
  }

  annotationStructFields.push_back(IntegerType::get(gm->getContext(), 32));

  StructType* annotationStructTy = StructType::get(gm->getContext(),
						   annotationStructFields,
						   false);

  vector<Constant*> annotations;

  NamedMDNode* mdn = gm->getNamedMetadata("scout.kernels");
  for(size_t i = 0; i < mdn->getNumOperands(); ++i){
    MDNode* node = cast<MDNode>(mdn->getOperand(i)->getOperand(0));
    Function* f = cast<Function>(node->getOperand(0));

    // --------------------------------- signed args metadata

    node = cast<MDNode>(mdn->getOperand(i)->getOperand(5));
    Function::arg_iterator aitr = f->arg_begin();

    vector<Constant*> signedArgGlobals;

    for(size_t j = 0; j < node->getNumOperands(); ++j){
      ConstantInt* signedArg = cast<ConstantInt>(node->getOperand(j));
      if(signedArg->isOne()){
	Constant* signedArgConstant =
	  ConstantDataArray::getString(gm->getContext(), aitr->getName());

	string name = aitr->getName().str() + ".str";

	GlobalVariable* signedArgGlobal =
	  new GlobalVariable(*gm,
			     signedArgConstant->getType(),
			     true,
			     GlobalValue::InternalLinkage,
			     signedArgConstant, name, 0,
			     GlobalVariable::NotThreadLocal, 2);

	signedArgGlobals.push_back(ConstantExpr::getBitCast(signedArgGlobal, 
							    i8PtrTy));
      }
      ++aitr;
    }

    Constant* signedArgsConstant =
      ConstantArray::get(ArrayType::get(i8PtrTy, signedArgGlobals.size()),
			 signedArgGlobals);

    string name = 
      "llvm.signedOrSignedpointee.annotations." + f->getName().str();

    GlobalVariable* signedArgsGlobal =
      new GlobalVariable(*gm,
			 signedArgsConstant->getType(),
			 false,
			 GlobalValue::ExternalLinkage,
			 signedArgsConstant, name, 0);

    signedArgsGlobal->setSection("llvm.metadata");

    // --------------------------------- type args metadata

    node = cast<MDNode>(mdn->getOperand(i)->getOperand(6));
    aitr = f->arg_begin();

    vector<Constant*> typeArgGlobals;

    for(size_t j = 0; j < node->getNumOperands(); ++j){
      ConstantDataArray* typeConstant = 
	cast<ConstantDataArray>(node->getOperand(j));

      Constant* typeArgConstant =
	ConstantDataArray::getString(gm->getContext(), 
				     typeConstant->getAsString(), false);

      string name = aitr->getName().str() + ".type.str";

      GlobalVariable* typeArgGlobal =
	new GlobalVariable(*gm,
			   typeArgConstant->getType(),
			   true,
			   GlobalValue::InternalLinkage,
			   typeArgConstant, name, 0,
			   GlobalVariable::NotThreadLocal, 2);
      
      typeArgGlobals.push_back(ConstantExpr::getBitCast(typeArgGlobal,
							i8PtrTy));

      ++aitr;
    }

    Constant* typeArgsConstant =
      ConstantArray::get(ArrayType::get(i8PtrTy, typeArgGlobals.size()),
			 typeArgGlobals);

    name = "llvm.argtypename.annotations." + f->getName().str();

    GlobalVariable* typeArgsGlobal =
      new GlobalVariable(*gm,
			 typeArgsConstant->getType(),
			 false,
			 GlobalValue::ExternalLinkage,
			 typeArgsConstant, name, 0);

    typeArgsGlobal->setSection("llvm.metadata");

    // --------------------------------- llvm.global.annotations
    vector<Constant*> annotationElems;
    annotationElems.push_back(ConstantExpr::getBitCast(f, i8PtrTy));

    ConstantAggregateZero* sgvArray =
      ConstantAggregateZero::get(sgvArrayType);
    
    GlobalVariable* sgvGlobal =
      new GlobalVariable(*gm,
			 sgvArrayType,
			 true,
			 GlobalValue::InternalLinkage,
			 sgvArray,
			 "sgv");
    

    annotationElems.push_back(ConstantExpr::getBitCast(sgvGlobal, i8PtrTy));

    ConstantAggregateZero* fgvArray =
      ConstantAggregateZero::get(fgvArrayType);
    
    GlobalVariable* fgvGlobal =
      new GlobalVariable(*gm,
			 fgvArrayType,
			 true,
			 GlobalValue::InternalLinkage,
			 fgvArray,
			 "fgv");
    
    annotationElems.push_back(ConstantExpr::getBitCast(fgvGlobal, i8PtrTy));
    
    ConstantAggregateZero* lvgvArray =
      ConstantAggregateZero::get(lvgvArrayType);
    
    GlobalVariable* lvgvGlobal =
      new GlobalVariable(*gm,
			 lvgvArrayType,
			 true,
			 GlobalValue::InternalLinkage,
			 lvgvArray,
			 "lvgv");
    
    annotationElems.push_back(ConstantExpr::getBitCast(lvgvGlobal, i8PtrTy));

    ConstantAggregateZero* rvgvArray =
      ConstantAggregateZero::get(rvgvArrayType);
    
    GlobalVariable* rvgvGlobal =
      new GlobalVariable(*gm,
			 rvgvArrayType,
			 true,
			 GlobalValue::InternalLinkage,
			 rvgvArray,
			 "rvgv");
    
    annotationElems.push_back(ConstantExpr::getBitCast(rvgvGlobal, i8PtrTy));

    annotationElems.push_back(ConstantInt::get(gm->getContext(), APInt(32, 0)));
    Constant* annotation = 
      ConstantStruct::get(annotationStructTy, annotationElems);
    annotations.push_back(annotation);
  }

  ArrayType* annotationsArrayTy = 
    ArrayType::get(annotationStructTy, annotations.size());

  Constant* annotationsArray = 
    ConstantArray::get(annotationsArrayTy, annotations);

  GlobalVariable* annotationsGlobal =
    new GlobalVariable(*gm,
                       annotationsArrayTy,
                       false,
                       GlobalValue::AppendingLinkage,
                       annotationsArray,
                       "llvm.global.annotations");

  annotationsGlobal->setSection("llvm.metadata");

  mdn->eraseFromParent();

  //cerr << "------------------ gpu module" << endl;
  //gm->dump();

  string bitcode;
  raw_string_ostream bs(bitcode);
  formatted_raw_ostream fbs(bs);

  WriteBitcodeToFile(gm, fbs);

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

  mdn = m.getNamedMetadata("scout.kernels");
  for(size_t i = 0; i < mdn->getNumOperands(); ++i){
    MDNode* node = cast<MDNode>(mdn->getOperand(i)->getOperand(0));
    Function* f = cast<Function>(node->getOperand(0));

    SmallVector<ConstantInt*, 4> dims;
    node = cast<MDNode>(mdn->getOperand(i)->getOperand(2));
    for(size_t j = 0; j < node->getNumOperands(); ++j){
      dims.push_back(cast<ConstantInt>(node->getOperand(j)));
    }

    SmallVector<Value*, 3> fields;
    node = cast<MDNode>(mdn->getOperand(i)->getOperand(4));
    for(size_t j = 0; j < node->getNumOperands(); ++j){
      fields.push_back(cast<Value>(node->getOperand(j)));
    }

    f->deleteBody();
    BasicBlock* entry = BasicBlock::Create(m.getContext(), "entry", f);
    builder.SetInsertPoint(entry);

    Function* initKernelFunc = m.getFunction("__sc_opencl_init_kernel");
    if(!initKernelFunc){
      vector<Type*> args;
      args.push_back(i8PtrTy);
      
      FunctionType* retType =
	FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

      initKernelFunc = Function::Create(retType,
					Function::ExternalLinkage,
					"__sc_opencl_init_kernel",
					&m);
    }

    Constant* kc =
      ConstantDataArray::getString(m.getContext(), f->getName());

    GlobalVariable* kg =
      new GlobalVariable(m,
			 kc->getType(),
			 true,
			 GlobalValue::PrivateLinkage,
			 kc, "kernel.name");


    Value* kn = builder.CreateBitCast(kg, i8PtrTy, "bitcode");
    builder.CreateCall(initKernelFunc, kn);

    GlobalVariable* gv =
      new GlobalVariable(m,
			 bitcodeData->getType(),
                       true,
			 GlobalValue::PrivateLinkage,
			 bitcodeData, "gpu.module");

    builder.CreateRetVoid();
  }

  //cerr << "-------------- dumping final module" << endl;
  //m.dump();

  return true;
}

char DoallToAMDIL::ID = 1;
RegisterPass<DoallToAMDIL> DoallToAMDIL("Doall-to-AMDIL", "Generate LLVM bitcode for GPU kernels and embed as a global value.");
