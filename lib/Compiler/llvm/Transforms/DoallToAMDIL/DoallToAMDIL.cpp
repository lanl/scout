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
#include <fstream>
#include <cstdio>

#include "llvm/Support/Program.h"
#include "llvm/Support/PathV1.h"
#include "llvm/Support/PathV2.h"
#include "llvm/IR/Constants.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/InstVisitor.h"

using namespace std;
using namespace llvm;

namespace{

  static const uint8_t FIELD_READ = 0x01;
  static const uint8_t FIELD_WRITE = 0x02;
  static const uint8_t FIELD_READ_WRITE = 0x03;

  static const uint8_t FIELD_READ_MASK = 0x01;
  static const uint8_t FIELD_WRITE_MASK = 0x02;

  string demangleName(const string& str){
    int status;
    char* dn = abi::__cxa_demangle(str.c_str(), 0, 0, &status);
    string ret = dn;
    free(dn);
    return ret;
  }

  size_t getSizeInBytes(Type* type){
    if(type->isSingleValueType() && !type->isPointerTy()){
      return type->getPrimitiveSizeInBits() / 8;
    }
    else if(type->isArrayTy()){
      size_t numElements = cast<ArrayType >(type)->getNumElements();
      return numElements * getSizeInBytes(type->getContainedType(0));
    }
    else{
      size_t size = 0;

      typedef Type::subtype_iterator SubTypeIterator;
      SubTypeIterator subtype = type->subtype_begin();
      for(; subtype != type->subtype_end(); ++subtype){
	size += getSizeInBytes(*subtype);
      }
      return size;
    }
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
	string vs = v->getName().str();
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
	SmallVector<llvm::Value *, 3> args;
        unsigned numArgs = I.getNumArgOperands();
	for(unsigned i = 0; i < numArgs; ++i){
          Value* arg = I.getArgOperand(i);
	  string s = arg->getName().str();

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
      string s = I.getPointerOperand()->getName().str();
      if(!s.empty()){
        symbolMap[s] = true;
      }
    }

    void visitBitCastInst(BitCastInst& I){
      string vs = I.getOperand(0)->getName().str();
      if(!vs.empty()){
	valueMap[&I] = vs;
      }
    }

    typedef map<string, bool> SymbolMap;
    typedef map<Value*, string> ValueMap;

    SymbolMap symbolMap;
    ValueMap valueMap;
    Module& module;
    DoallToAMDIL::FunctionMDMap& functionMDMap;
  };

} // end namespace

static BasicBlock* 
CloneGPUBasicBlock(const BasicBlock *BB,
                   ValueToValueMapTy &VMap,
                   const Twine &NameSuffix,
                   Function *F,
                   ClonedCodeInfo *CodeInfo) {
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "", F);
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over.
  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II) {

    Instruction* NewInst;

    if(const GetElementPtrInst* gep = 
       dyn_cast<GetElementPtrInst>(II)){
      vector<Value*> indices;
      
      for(size_t i = 1; i < gep->getNumOperands(); ++i){
        indices.push_back(gep->getOperand(i));
      }

      Value* op0 = VMap[II->getOperand(0)];

      NewInst = 
        GetElementPtrInst::Create(op0, ArrayRef<Value*>(indices));

      VMap[op0] = op0;
    }
    else{
      NewInst = II->clone();
    }

    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= (isa<CallInst>(II) && !isa<DbgInfoIntrinsic>(II));
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (isa<ConstantInt>(AI->getArraySize()))
        hasStaticAllocas = true;
      else
        hasDynamicAllocas = true;
    }
  }
  
  if (CodeInfo) {
    CodeInfo->ContainsCalls          |= hasCalls;
    CodeInfo->ContainsDynamicAllocas |= hasDynamicAllocas;
    CodeInfo->ContainsDynamicAllocas |= hasStaticAllocas && 
      BB != &BB->getParent()->getEntryBlock();
  }
  return NewBB;
}

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
                           .addAttributes(NewFunc->getContext(),
                                          AttributeSet::ReturnIndex,
                                          OldFunc->getAttributes()));
    NewFunc->setAttributes(NewFunc->getAttributes()
                           .addAttributes(NewFunc->getContext(),
                                          AttributeSet::FunctionIndex,
                                          OldFunc->getAttributes()));
  }

  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;

    BasicBlock *CBB = 
      CloneGPUBasicBlock(&BB, VMap, NameSuffix, NewFunc, CodeInfo);

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
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II){
      RemapInstruction(II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper);
    }
}

DoallToAMDIL::DoallToAMDIL(const std::string& sccPath)
  : ModulePass(ID),
    sccPath_(sccPath){
 }

DoallToAMDIL::~DoallToAMDIL(){

}

ModulePass *createDoallToAMDILPass(const std::string& sccPath) {
  return new DoallToAMDIL(sccPath);
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

  //cerr << "---------------- dumping cloned module" << endl;
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

  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {    // modify the address space of the params to correspond
    // to the OpenCL address spaces
    FunctionType* FT = 
      cast<FunctionType>(I->getType()->getElementType());

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
    		       Function::ExternalLinkage, fullName, New);

    /*
    Function::const_arg_iterator itr = I->arg_begin();
    Function::arg_iterator itr2 = NF->arg_begin();
    while(itr != I->arg_end()){
      const Value* v1 = itr;
      Value* v2 = itr2;
      VMap[v1] = v2;
      ++itr;
      ++itr2;
    }
    */
    
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

  Type* i8Ty = IntegerType::get(m.getContext(), 8);
  Type* i32Ty = IntegerType::get(m.getContext(), 32);
  Type* i8PtrTy = PointerType::get(i8Ty, 0);

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

  ArrayType* sgvArrayType = ArrayType::get(i8Ty, 1);
  ArrayType* fgvArrayType = ArrayType::get(i8Ty, 1);
  ArrayType* lvgvArrayType = ArrayType::get(i8PtrTy, 0);
  ArrayType* rvgvArrayType = ArrayType::get(i8PtrTy, 0);

  vector<Type*> annotationStructFields;

  for(size_t i = 0; i < 5; ++i){
    annotationStructFields.push_back(i8PtrTy);
  }

  annotationStructFields.push_back(i32Ty);

  StructType* annotationStructTy = StructType::get(gm->getContext(),
						   annotationStructFields,
						   false);

  vector<Constant*> annotations;

  NamedMDNode* mdn = gm->getNamedMetadata("scout.kernels");
  for(size_t i = 0; i < mdn->getNumOperands(); ++i){
    MDNode* node = cast<MDNode>(mdn->getOperand(i)->getOperand(0));
    Function* f = cast<Function>(node->getOperand(0));

    for(Function::iterator itr = f->begin(), itrEnd = f->end(); 
	itr != itrEnd; ++itr){
      for(BasicBlock::iterator bitr = itr->begin(), bitrEnd = itr->end();
	  bitr != bitrEnd; ++bitr){
	if(isa<AllocaInst>(bitr) && bitr->getName().startswith("indvar")){
	  Function* getGlobalIdFunc = gm->getFunction("get_global_id");
	  if(!getGlobalIdFunc){
	    vector<Type*> args;
	    args.push_back(i32Ty);

	    FunctionType* retType =
	      FunctionType::get(i32Ty, args, false);

	    getGlobalIdFunc = Function::Create(retType,
					       Function::ExternalLinkage,
					       "get_global_id",
					       gm);

	    getGlobalIdFunc->addFnAttr(Attribute::NoUnwind);         
	  }
	  builder.SetInsertPoint(bitr);
	  Value* v = builder.CreateCall(getGlobalIdFunc,
					ConstantInt::get(i32Ty, 0), "threadidx");
	  ++bitr;
	  bitr->replaceAllUsesWith(v);
	  break;
	}
      }
    }

    f->addFnAttr(Attribute::NoUnwind);                                            f->addFnAttr(Attribute::ReadNone);  
    
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

      aitr->addAttr(AttributeSet::get(gm->getContext(), 0, 
                                      Attribute::NoCapture));
      ++aitr;
    }

    Constant* signedArgsConstant =
      ConstantArray::get(ArrayType::get(i8PtrTy, signedArgGlobals.size()),
			 signedArgGlobals);

    string name = 
      "llvm.signedOrSignedpointee.annotations." + f->getName().str();

    if(!signedArgGlobals.empty()){
      GlobalVariable* signedArgsGlobal =
	new GlobalVariable(*gm,
		       signedArgsConstant->getType(),
		       false,
		       GlobalValue::ExternalLinkage,
		       signedArgsConstant, name, 0);
    
      signedArgsGlobal->setSection("llvm.metadata");
    }

    // --------------------------------- type args metadata

    node = cast<MDNode>(mdn->getOperand(i)->getOperand(6));
    aitr = f->arg_begin();

    vector<Constant*> typeArgGlobals;

    for(size_t j = 0; j < node->getNumOperands(); ++j){
      ConstantDataArray* typeConstant = 
	cast<ConstantDataArray>(node->getOperand(j));

      Constant* typeArgConstant =
	ConstantDataArray::getString(gm->getContext(), 
				     typeConstant->getAsCString(), false);

      GlobalVariable* typeArgGlobal =
	new GlobalVariable(*gm,
			   typeArgConstant->getType(),
			   true,
			   GlobalValue::InternalLinkage,
			   typeArgConstant, ".str", 0,
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
			 "sgv", 0, GlobalVariable::NotThreadLocal, 2);
    
    annotationElems.push_back(ConstantExpr::getCast(Instruction::BitCast, sgvGlobal, i8PtrTy));

    ConstantAggregateZero* fgvArray =
      ConstantAggregateZero::get(fgvArrayType);
    
    GlobalVariable* fgvGlobal =
      new GlobalVariable(*gm,
			 fgvArrayType,
			 true,
			 GlobalValue::InternalLinkage,
			 fgvArray,
			 "fgv", 0, GlobalVariable::NotThreadLocal, 2);
    
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

  std::string strIR;
  raw_string_ostream IROut(strIR);
  gm->print(IROut, 0);

  SmallString<128> commandPath(sccPath_);
  sys::path::remove_filename(commandPath);
  sys::path::append(commandPath, "llvm-as-3.1");

  SmallString<128> inputPath;
  sys::path::system_temp_directory(false, inputPath);
  sys::path::append(inputPath, "sc_amdil_in.ll");
  std::string inPath = inputPath.str().str();
  
  std::ofstream istr(inPath.c_str());
  assert(istr.is_open() && 
         "Failed to open output file stream in DoallToAMDIL.");
  istr << strIR;
  istr.close();

  SmallString<128> outputPath;
  sys::path::system_temp_directory(false, outputPath);
  sys::path::append(outputPath, "sc_amdil_out.bc");
  std::string outPath = outputPath.str().str();

  typedef std::vector<const char*> ArgVec;
  ArgVec argVec;
  argVec.push_back(commandPath.str().str().c_str());
  argVec.push_back("-o");
  argVec.push_back(outPath.c_str());
  argVec.push_back(inPath.c_str());
  argVec.push_back(0);

  sys::Path executePath(commandPath.str());
  
  std::string errOut;

  //cerr << "outputPath: " << outPath << endl;
  //cerr << "inputPath: " << inPath << endl;
  //cerr << "commandPath: " << commandPath.str().str() << endl;

  int status = 
    sys::Program::ExecuteAndWait(executePath,
                                 (const char**)argVec.data(),
                                 0, 0, 0, 0, &errOut);

  //cerr << "errOut is: " << errOut << endl;

  assert(status == 0 && 
         "Failed to run llvm-as in DoallToAMDIL.");

  FILE* fh = fopen(outPath.c_str(), "r");
  fseek(fh, 0, SEEK_END);

  size_t bitcodeSize = ftell(fh);

  rewind(fh);

  char* buf = (char*)malloc(bitcodeSize + 1);
  buf[bitcodeSize] = '\0';
  fread(buf, sizeof(char), bitcodeSize, fh);
  fclose(fh);

  remove(inPath.c_str());
  remove(outPath.c_str());

  Constant* bitcodeData =                                                    
    ConstantDataArray::get(m.getContext(),
                           ArrayRef<unsigned char>((unsigned char*)buf, bitcodeSize));  

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
		      ConstantInt::get(i32Ty, bitcodeSize));

  mdn = m.getNamedMetadata("scout.kernels");
  for(size_t i = 0; i < mdn->getNumOperands(); ++i){
    MDNode* node = cast<MDNode>(mdn->getOperand(i)->getOperand(0));
    Function* f = cast<Function>(node->getOperand(0));

    uint32_t meshDims[] = {1,1,1};

    size_t meshSize = 1;
    node = cast<MDNode>(mdn->getOperand(i)->getOperand(2));
    for(size_t j = 0; j < node->getNumOperands(); j += 2){
      ConstantInt* start = cast<ConstantInt>(node->getOperand(j));
      ConstantInt* end = cast<ConstantInt>(node->getOperand(j+1));
      uint32_t dim = end->getValue().getZExtValue() -
        start->getValue().getZExtValue();

      meshSize *= dim;
      meshDims[j/2] = dim;
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
      args.push_back(i8PtrTy);
      args.push_back(i32Ty);
      args.push_back(i32Ty);
      args.push_back(i32Ty);
      
      FunctionType* retType =
	FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

      initKernelFunc = Function::Create(retType,
					Function::ExternalLinkage,
					"__sc_opencl_init_kernel",
					&m);
    }

    MDNode* meshNameNode = 
      cast<MDNode>(mdn->getOperand(i)->getOperand(3));
    Constant* mc = cast<Constant>(meshNameNode->getOperand(0));

    GlobalVariable* mg =
      new GlobalVariable(m,
			 mc->getType(),
			 true,
			 GlobalValue::PrivateLinkage,
			 mc, "mesh.name");

    Value* mn = builder.CreateBitCast(mg, i8PtrTy, "meshName");

    Constant* kc =
      ConstantDataArray::getString(m.getContext(), f->getName());

    GlobalVariable* kg =
      new GlobalVariable(m,
			 kc->getType(),
			 true,
			 GlobalValue::PrivateLinkage,
			 kc, "kernel.name");


    Value* kn = builder.CreateBitCast(kg, i8PtrTy, "kernel");

    Value* widthValue = ConstantInt::get(i32Ty, meshDims[0]);
    Value* heightValue = ConstantInt::get(i32Ty, meshDims[1]);
    Value* depthValue = ConstantInt::get(i32Ty, meshDims[2]);

    builder.CreateCall5(initKernelFunc, mn, kn,
                        widthValue, heightValue, depthValue);
    
    Function* setFieldFunc = m.getFunction("__sc_opencl_set_kernel_field");
    if(!setFieldFunc){
      vector<Type*> args;
      args.push_back(i8PtrTy);
      args.push_back(i8PtrTy);
      args.push_back(i32Ty);
      args.push_back(i8PtrTy);
      args.push_back(i32Ty);
      args.push_back(i8Ty);

      FunctionType* retType =
        FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

      setFieldFunc = Function::Create(retType,
                                      Function::ExternalLinkage,
                                      "__sc_opencl_set_kernel_field",
                                      &m);
    }

    FunctionMDMap::iterator fitr = functionMDMap.find(f->getName().str());
    assert(fitr != functionMDMap.end());
    MDNode* readArgs = fitr->second;

    Function::arg_iterator aitr = f->arg_begin();
    node = cast<MDNode>(mdn->getOperand(i)->getOperand(1));

    MDNode* argNode = cast<MDNode>(mdn->getOperand(i)->getOperand(4)); 

    for(size_t j = 0; j < node->getNumOperands(); ++j){
      ConstantDataArray* argNameConstant =
	cast<ConstantDataArray>(argNode->getOperand(j));

      string argName = argNameConstant->getAsCString();

      vector<Value*> params;

      ConstantInt* isMeshArg = cast<ConstantInt>(node->getOperand(j));
      
      // name of kernel
      params.push_back(kn);
      
      Constant* fc =
	ConstantDataArray::getString(m.getContext(), argName, false);
      
      GlobalVariable* fg =
	new GlobalVariable(m,
			   fc->getType(),
			   true,
			   GlobalValue::PrivateLinkage,
			   fc, "field.name");
      
      // name of mesh field
      params.push_back(builder.CreateBitCast(fg, i8PtrTy, "field.name"));
      
      // kernel argument position
      params.push_back(ConstantInt::get(m.getContext(), APInt(32, j)));
      
      // host ptr
      params.push_back(builder.CreateBitCast(aitr, i8PtrTy, "field.ptr"));
      
      PointerType* pointerType = cast<PointerType>(aitr->getType());
      
      size_t size = getSizeInBytes(pointerType->getElementType());
      if(isMeshArg->isOne()){
	size *= meshSize;
      }
      
      Value* fieldSize = ConstantInt::get(m.getContext(), APInt(32, size));
      
      params.push_back(fieldSize);

      uint8_t mode = 0;
      for(size_t k = 0; k < readArgs->getNumOperands(); ++k){
	Value* v = readArgs->getOperand(k);

	if(v->getName().str() == argName){
	  mode = FIELD_READ;
	  break;
	}
      }
      
      // debug - uncomment to write mesh fields back out
      if(isMeshArg->isOne()){
	mode |= FIELD_WRITE_MASK;
      }

      // read/write type
      params.push_back(ConstantInt::get(m.getContext(), APInt(8, mode)));

      builder.CreateCall(setFieldFunc, params);

      ++aitr;
    }

    Function* runKernelFunc = m.getFunction("__sc_opencl_run_kernel");
    if(!runKernelFunc){
      vector<Type*> args;
      args.push_back(i8PtrTy);
      
      FunctionType* retType =
        FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

      runKernelFunc = Function::Create(retType,
                                      Function::ExternalLinkage,
                                      "__sc_opencl_run_kernel",
                                      &m);
    }

    vector<Value*> params;
    params.push_back(kn);

    builder.CreateCall(runKernelFunc, params);

    builder.CreateRetVoid();
  }

  //cerr << "-------------- dumping final module" << endl;
  //m.dump();

  return true;
}

char DoallToAMDIL::ID = 1;
RegisterPass<DoallToAMDIL> DoallToAMDIL("Doall-to-AMDIL", "Generate LLVM bitcode for GPU kernels and embed as a global value.");
