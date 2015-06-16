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
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
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

using namespace std;
using namespace llvm;

namespace{

typedef vector<Type*> TypeVec;
typedef vector<Value*> ValueVec;
typedef vector<string> StringVec;

class CUDAModule{
public:
  
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
  }

  void init(){
    //cerr << "-- CPU module before: " << endl;
    //module_->dump();
    //cerr << "======================" << endl;
    
    NamedMDNode* volrensMD = module_->getNamedMetadata("scout.volren");

    for(size_t i = 0; i < volrensMD->getNumOperands(); ++i){
      MDNode* volrenMD = volrensMD->getOperand(i);
      volrenMD->dump();
    }

    //cerr << "--------- kernel module: " << endl;
    //kernelModule_.dump();
    //cerr << "=========================" << endl;

    string ptx = generatePTX();

    Constant* pc = ConstantDataArray::getString(context_, ptx);
    
    ptxGlobal_ = 
      new GlobalVariable(*module_,
                         pc->getType(),
                         true,
                         GlobalValue::PrivateLinkage,
                         pc,
                         "ptx");

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

  ConstantInt* getInt32(int32_t v){
    return ConstantInt::get(context_, APInt(32, v, true));
  }

  ConstantInt* getInt8(int8_t v){
    return ConstantInt::get(context_, APInt(8, v, true));
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
