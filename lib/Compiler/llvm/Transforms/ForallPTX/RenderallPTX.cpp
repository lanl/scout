/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2015. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ###########################################################################
 *
 * Notes
 *
 * #####
 */

#include "llvm/Transforms/Scout/RenderallPTX/RenderallPTX.h"

#include <iostream>
#include <memory>

#include "llvm/ADT/Triple.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace std;
using namespace llvm;

namespace{

using TypeVec = vector<Type*>;
using ValueVec = vector<Value*>;
using StringVec = vector<string>;
using FuncVec = vector<Function*>;

enum class ElementKind{
  Int32 = 0,
  Int64,
  Float,
  Double
};
  
class CUDAModule{
public:

  class Kernel{
  public:
   
    Kernel(CUDAModule& m, MDNode* volrenMD)
      : m_(m),
        kernelModule_(m.kernelModule()),
        context_(m_.context()),
        volrenMD_(volrenMD){}

    ConstantInt* getInt32(int32_t v){
      return ConstantInt::get(context_, APInt(32, v, true));
    }
    
    ConstantInt* getInt8(int8_t v){
      return ConstantInt::get(context_, APInt(8, v, true));
    }

    void getCalledFuncs(Function* f, FuncVec& funcs){
      for(Function::iterator itr = f->begin(), itrEnd = f->end();
          itr != itrEnd; ++itr){
      
        BasicBlock* b = itr;
        for(BasicBlock::iterator bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){
          Instruction* i = bitr;

          if(CallInst* ci = dyn_cast<CallInst>(i)){
            funcs.push_back(ci->getCalledFunction());
          }
        }
      }
    }

    Function* addFunction(Module& module,
                          ValueToValueMapTy& remap,
                          Function* f){

      FunctionType* ft = f->getFunctionType();

      TypeVec params;

      for(auto itr = ft->param_begin(), itrEnd = ft->param_end();
          itr != itrEnd; ++itr){
        params.push_back(*itr);
      }

      FunctionType* nft = 
        FunctionType::get(ft->getReturnType(), params, false);

    
      Function* nf = 
        Function::Create(nft, Function::ExternalLinkage,
                         f->getName(), &kernelModule_);

      Function::arg_iterator aitr = f->arg_begin();
      Function::arg_iterator naitr = nf->arg_begin();

      aitr = f->arg_begin();
      while(aitr != f->arg_end()){
        naitr->setName(aitr->getName());
        remap[aitr] = naitr;
        ++aitr;
        ++naitr;
      }

      for(auto itr = f->begin(), itrEnd = f->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        BasicBlock* nb = BasicBlock::Create(context_, b->getName(), nf);
        remap[b] = nb;

        for(auto bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){
          Instruction* i = bitr;
          Instruction* ni = i->clone();
         
          remap[i] = ni;
         
          ni->setName(i->getName());
          BasicBlock::InstListType& il = nb->getInstList();
          il.push_back(ni);
        }
      }

      for(auto itr = nf->begin(), itrEnd = nf->end();
          itr != itrEnd; ++itr){

        BasicBlock* b = itr;
        for(auto bitr = b->begin(), bitrEnd = b->end();
            bitr != bitrEnd; ++bitr){
          Instruction* i = bitr;
          RemapInstruction(i, remap, RF_IgnoreMissingEntries, 0);
        }
      }

      remap[f] = nf;

      return nf;
    }

    void init(){
      renderallFunc_ = cast<Function>(cast<ValueAsMetadata>(
                        volrenMD_->getOperand(0))->getValue());

      Function* wrapperFunc = cast<Function>(cast<ValueAsMetadata>(
                                volrenMD_->getOperand(1))->getValue());

      Function* transferFunc = cast<Function>(cast<ValueAsMetadata>(
                                volrenMD_->getOperand(2))->getValue());
      
      FuncVec calledFuncs;
      getCalledFuncs(transferFunc, calledFuncs);
      getCalledFuncs(wrapperFunc, calledFuncs);
      
      ValueToValueMapTy remap;

      for(Function* f : calledFuncs){
        addFunction(kernelModule_, remap, f); 
      }

      addFunction(kernelModule_, remap, transferFunc);

      Function* wf = addFunction(kernelModule_, remap, wrapperFunc);

      wrapperFunc->eraseFromParent();
      transferFunc->eraseFromParent();

      for(Function* f : calledFuncs){
        f->eraseFromParent();
      }

      NamedMDNode* annotations = 
        kernelModule_.getOrInsertNamedMetadata("nvvm.annotations");

      SmallVector<Metadata*, 3> av;
      av.push_back(ValueAsMetadata::get(wf));
      av.push_back(MDString::get(context_, "kernel"));
      av.push_back(ValueAsMetadata::get(getInt32(1)));

      annotations->addOperand(MDNode::get(context_, av));
    }

    void finishRenderall(){
      Module* module = m_.module();
      
      IRBuilder<> B(context_);

      Value* ptxDir = module->getNamedGlobal("scout.ptx.dir");
      assert(ptxDir);
      
      BasicBlock* b = BasicBlock::Create(context_, "entry", renderallFunc_);
      B.SetInsertPoint(b);

      auto aitr = renderallFunc_->arg_begin();
      Value* meshPtr = aitr++;
      Value* windowPtr = aitr++;
      Value* width = aitr++;
      Value* height = aitr++;
      Value* depth = aitr++;
      
      string meshName = 
        cast<MDString>(volrenMD_->getOperand(3))->getString().str();

      Value* ptx = B.CreateBitCast(m_.ptxGlobal(), m_.stringTy);

      Value* kernelName = 
        B.CreateGlobalStringPtr(renderallFunc_->getName());
      kernelName = B.CreateBitCast(kernelName, m_.stringTy);

      Function* f = module->getFunction("__scrt_volren_init_kernel");
      
      ValueVec args = 
        {B.CreateBitCast(ptxDir, m_.stringTy),
          B.CreateBitCast(B.CreateGlobalStringPtr(meshName), m_.stringTy),
          ptx, kernelName, windowPtr, width, height, depth};
      
      B.CreateCall(f, args);

      f = module->getFunction("__scrt_volren_init_field");

      MDNode* fieldsNode = cast<MDNode>(volrenMD_->getOperand(4));

      size_t numFields = fieldsNode->getNumOperands();

      for(size_t i = 0; i < numFields; ++i){
        MDNode* fieldNode = cast<MDNode>(fieldsNode->getOperand(i));

        string fieldNameStr =
          cast<MDString>(fieldNode->getOperand(0))->getString().str();

        Value* fieldName = B.CreateGlobalStringPtr(fieldNameStr);
        fieldName = B.CreateBitCast(fieldName, m_.stringTy);

        uint32_t index =
          cast<ConstantInt>(cast<ValueAsMetadata>(
            fieldNode->getOperand(1))->getValue())->getZExtValue();

        Value* hostPtr = B.CreateStructGEP(0, meshPtr, index);
        hostPtr = B.CreateLoad(hostPtr);
        hostPtr = B.CreateBitCast(hostPtr, m_.voidPtrTy);
        
        uint8_t et =
          cast<ConstantInt>(cast<ValueAsMetadata>(
            fieldNode->getOperand(2))->getValue())->getZExtValue();

        size_t scalarSize;

        switch(ElementKind(et)){
        case ElementKind::Int32:
        case ElementKind::Float:
          scalarSize = 4;
          break;
        case ElementKind::Int64:
        case ElementKind::Double:
          scalarSize = 8;
          break;
        default:
          assert(false && "invalid element kind");
        }

        Value* elementSize = getInt32(scalarSize);
        Value* elementType = getInt8(et);

        args = {kernelName, fieldName, hostPtr, elementSize, elementType};
        B.CreateCall(f, args);
      }
      
      f = module->getFunction("__scrt_volren_set_var");

      size_t offset = 0;
      while(aitr != renderallFunc_->arg_end()){
        Value* data = B.CreateBitCast(aitr, m_.voidPtrTy, "var.ptr");
        
        Type* elementType = aitr->getType()->getPointerElementType();
        size_t size = elementType->getPrimitiveSizeInBits()*8;

        args = {kernelName, getInt32(offset), data, getInt32(size)};
        B.CreateCall(f, args);
        offset += size;
        ++aitr;
      }

      f = module->getFunction("__scrt_volren_run");
      args = {kernelName};
      B.CreateCall(f, args);

      B.CreateRetVoid();
    }

  private:
    CUDAModule& m_;
    Module& kernelModule_;
    LLVMContext& context_;
    MDNode* volrenMD_;
    Function* renderallFunc_;
  };
  
  CUDAModule(Module* module)
  : module_(module),
    context_(module->getContext()),
    kernelModule_("kernel_module", context_),
    builder_(context_){

    int32Ty = Type::getInt32Ty(context_);
    int8Ty = Type::getInt8Ty(context_);
    voidTy = Type::getVoidTy(context_);
    stringTy = PointerType::get(int8Ty, 0);
    voidPtrTy = PointerType::get(int8Ty, 0);

    TypeVec params;

    params = 
      {stringTy, stringTy, stringTy, stringTy,
        voidPtrTy, int32Ty, int32Ty, int32Ty};
    
    createFunction("__scrt_volren_init_kernel", voidTy, params);

    params = {stringTy, stringTy, voidPtrTy, int32Ty, int8Ty};
    createFunction("__scrt_volren_init_field", voidTy, params);

    params = {stringTy, int32Ty, voidPtrTy, int32Ty};
    createFunction("__scrt_volren_set_var", voidTy, params);

    params = {stringTy};
    createFunction("__scrt_volren_run", voidTy, params);
  }

  void init(){
    //cerr << "-- CPU module before: " << endl;
    //module_->dump();
    //cerr << "======================" << endl;
    
    NamedMDNode* volrensMD = module_->getNamedMetadata("scout.volren");
    assert(volrensMD);

    vector<Kernel*> kernels;

    for(size_t i = 0; i < volrensMD->getNumOperands(); ++i){
      MDNode* volrenMD = volrensMD->getOperand(i);

      Kernel* kernel = new Kernel(*this, volrenMD);
      kernel->init();
      kernels.push_back(kernel);
    }

    string ptx = generatePTX();

    Constant* pc = ConstantDataArray::getString(context_, ptx);
    
    ptxGlobal_ = 
      new GlobalVariable(*module_,
                         pc->getType(),
                         true,
                         GlobalValue::PrivateLinkage,
                         pc,
                         "ptx");

    for(Kernel* kernel : kernels){
      kernel->finishRenderall();
    }

    //cerr << "--------- kernel module: " << endl;
    //kernelModule_.dump();
    //cerr << "=========================" << endl;

    //cerr << "--- CPU module after: " << endl;
    //module_->dump();
    //cerr << "======================" << endl;
  }

  Module* module(){
    return module_;
  }

  Module& kernelModule(){
    return kernelModule_;
  }

  Function* getSREGFunc(const string& suffix){
    string name = "llvm.nvvm.read.ptx.sreg." + suffix;

    Function* f = kernelModule_.getFunction(name);
    if(f){
      return f;
    }

    FunctionType* ft = FunctionType::get(int32Ty, false);
    f = Function::Create(ft, Function::ExternalLinkage, name, &kernelModule_);

    return f;
  }

  LLVMContext& context(){
    return context_;
  }

  const Target* findPTXTarget(){
    for(TargetRegistry::iterator itr =  TargetRegistry::targets().begin(),
          itrEnd =  TargetRegistry::targets().end(); itr != itrEnd; ++itr) {
      if(strcmp(itr->getName(), "nvptx64") == 0) {
        return &*itr;
      }
    }
    
    return 0;
  }

  TargetMachine* createTargetMachine(const Target* target){
    Triple triple(sys::getDefaultTargetTriple());
    triple.setArch(Triple::nvptx64);
    
    return 
      target->createTargetMachine(triple.getTriple(),
                                  "sm_20",
                                  "",
                                  TargetOptions(),
                                  Reloc::Default,
                                  CodeModel::Default,
                                  CodeGenOpt::Aggressive);
  }

  string generatePTX(){
    const Target* target = findPTXTarget();
    assert(target && "failed to find NVPTX target");

    TargetMachine* targetMachine = createTargetMachine(target);
    const DataLayout* dataLayout = targetMachine->getDataLayout();

    assert(dataLayout && "failed to get data layout");

    kernelModule_.setDataLayout(*dataLayout);

    legacy::PassManager* passManager = new legacy::PassManager;

    passManager->add(createVerifierPass());

    SmallVector<char, 65536> buf;
    raw_svector_ostream ostr(buf);
    
    bool fail =
    targetMachine->addPassesToEmitFile(*passManager,
                                       ostr,
                                       TargetMachine::CGFT_AssemblyFile,
                                       false);

    assert(!fail);
    
    passManager->run(kernelModule_);
    
    ostr.flush();
    
    delete passManager;
        
    return ostr.str().str();
  }

  Function* createFunction(const string& name,
                           Type* returnType,
                           const TypeVec& argTypes){
    FunctionType* ft =
      FunctionType::get(returnType, argTypes, false);
    
    Function* f =
      Function::Create(ft,
                       Function::ExternalLinkage,
                       name.c_str(), module_);
    
    return f;
  }

  GlobalVariable* ptxGlobal(){
    return ptxGlobal_;
  }
  
  Type* int32Ty;
  Type* int8Ty;
  Type* voidTy;
  Type* stringTy;
  Type* voidPtrTy;

private:
  Module* module_;
  LLVMContext& context_;
  Module kernelModule_;
  IRBuilder<> builder_;
  GlobalVariable* ptxGlobal_;
};

class RenderallPTX : public ModulePass{
public:
  static char ID;

  RenderallPTX() : ModulePass(ID){}

  void getAnalysisUsage(AnalysisUsage& AU) const override{}

  const char *getPassName() const {
    return "Renderall-to-PTX";
  }

  bool runOnModule(Module& M) override{
    CUDAModule cudaModule(&M);
    cudaModule.init();
    return true;
  }
};

char RenderallPTX::ID;

} // end namespace

ModulePass* llvm::createRenderallPTXPass(){
  return new RenderallPTX;
}
