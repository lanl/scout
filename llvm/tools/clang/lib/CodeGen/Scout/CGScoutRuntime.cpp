/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
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

#include "Scout/CGScoutRuntime.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;

CGScoutRuntime::~CGScoutRuntime() {}

// call the correct scout runtime Initializer
llvm::Function *CGScoutRuntime::ModuleInitFunction(CodeGenFunction &CGF, SourceLocation Loc) {

  llvm::Function *rtInitFn = EmitRuntimeInitFunc(Loc, &CGF);
  return rtInitFn;
  
  /*
  std::string funcName;
  if(CGM.getLangOpts().ScoutNvidiaGPU){
   funcName = "__scrt_cuda_init";
  } else if(CGM.getLangOpts().ScoutAMDGPU){
   funcName = "__scrt_init_opengl";
  } else {
   funcName = "__scrt_init_cpu";
  }

  llvm::Function *scrtInit = CGM.getModule().getFunction(funcName);
  if(!scrtInit) {
   std::vector<llvm::Type*> args;

   llvm::FunctionType *FTy =
   llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()),
                            args, false);

   scrtInit = llvm::Function::Create(FTy,
                                     llvm::Function::ExternalLinkage,
                                     funcName,
                                     &CGM.getModule());
  }
  return scrtInit;
  */
}

// build a function call to a scout runtime function w/ no arguments
// SC_TODO: could we use CreateRuntimeFunction? or GetOrCreateLLVMFunction?
llvm::Function *CGScoutRuntime::ScoutRuntimeFunction(std::string funcName, std::vector<llvm::Type*> Params ) {

  llvm::Function *Func = CGM.getModule().getFunction(funcName);
  if(!Func){
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()),
            Params, false);

    Func = llvm::Function::Create(FTy,
        llvm::Function::ExternalLinkage,
        funcName,
        &CGM.getModule());
  }
  return Func;
}

llvm::Function *CGScoutRuntime::ScoutRuntimeFunction(std::string funcName, std::vector<llvm::Type*> Params, 
    llvm::Type* retType) {

  llvm::Function *Func = CGM.getModule().getFunction(funcName);
  if(!Func){
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(retType, Params, false);

    Func = llvm::Function::Create(FTy,
        llvm::Function::ExternalLinkage,
        funcName,
        &CGM.getModule());
  }
  return Func;
}

// function call to runtime malloc
llvm::Function *CGScoutRuntime::MemAllocFunction() {
  std::string funcName = "__scrt_malloc";
  llvm::Function *Func = CGM.getModule().getFunction(funcName);

  if(!Func) {
    llvm::FunctionType *FTy = llvm::FunctionType::get(CGM.Int8PtrTy,
        CGM.Int64Ty, /*isVarArg=*/false);
    Func = llvm::Function::Create(FTy,
       llvm::GlobalValue::ExternalLinkage,
       funcName,
       &CGM.getModule());
  }
  return Func;
}

// build function call to __scrt_renderall_uniform_begin()
llvm::Function *CGScoutRuntime::RenderallUniformBeginFunction() {
  std::string funcName = "__scrt_renderall_uniform_begin";
  std::vector<llvm::Type*> Params;

  for(int i=0; i < 3; i++) {
    Params.push_back(llvm::Type::getInt32Ty(CGM.getLLVMContext()));
  }
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));
  return ScoutRuntimeFunction(funcName, Params);
}

// build function call to __scrt_renderall_end()
llvm::Function *CGScoutRuntime::RenderallEndFunction() {
  std::string funcName = "__scrt_renderall_end";

  std::vector<llvm::Type*> Params;
  return ScoutRuntimeFunction(funcName, Params);
}

// build function call to __scrt_create_window()
llvm::Function *CGScoutRuntime::CreateWindowFunction() {
  std::string funcName = "__scrt_create_window";

  std::vector<llvm::Type*> Params;
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 16));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 16));

  return ScoutRuntimeFunction(
      funcName, Params, 
      /*retType*/ 
      llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));
}

// build function call to create renderable if necessary and return color pointer()
llvm::Function *CGScoutRuntime::CreateWindowQuadRenderableColorsFunction() {
  std::string funcName = "__scrt_window_quad_renderable_colors";

  std::vector<llvm::Type*> Params;

  // params for width, height, depth of renderable
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));

  // param for pointer to window
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));

  return ScoutRuntimeFunction(
      funcName, Params,
      /*retType -- pointer to float for colors*/
      llvm::PointerType::get(
        llvm::VectorType::get(
          llvm::Type::getFloatTy(CGM.getModule().getContext()), 4), 0));
}

llvm::Function
*CGScoutRuntime::CreateWindowQuadRenderableVertexColorsFunction() {
  std::string funcName = "__scrt_window_quad_renderable_vertex_colors";
  
  std::vector<llvm::Type*> Params;
  
  // params for width, height, depth of renderable
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  
  // param for pointer to window
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));
  
  return ScoutRuntimeFunction(
                              funcName, Params,
                              /*retType -- pointer to float for colors*/
                              llvm::PointerType::get(
                                                     llvm::VectorType::get(
                                                                           llvm::Type::getFloatTy(CGM.getModule().getContext()), 4), 0));
}

llvm::Function
*CGScoutRuntime::CreateWindowQuadRenderableEdgeColorsFunction() {
  std::string funcName = "__scrt_window_quad_renderable_edge_colors";
  
  std::vector<llvm::Type*> Params;
  
  // params for width, height, depth of renderable
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));
  
  // param for pointer to window
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));
  
  return ScoutRuntimeFunction(
                              funcName, Params,
                              /*retType -- pointer to float for colors*/
                              llvm::PointerType::get(
                                                     llvm::VectorType::get(
                                                                           llvm::Type::getFloatTy(CGM.getModule().getContext()), 4), 0));
}

// build function call to do window paint 
llvm::Function *CGScoutRuntime::CreateWindowPaintFunction() {
  std::string funcName = "__scrt_window_paint";

  std::vector<llvm::Type*> Params;

  // param for pointer to window
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getModule().getContext(), 8), 0));

  return ScoutRuntimeFunction(funcName, Params);
}

// build function to set all colors to red
//llvm::Function *CGcoutRuntime::SetColorsRedFunction() { }

// get Value for global runtime variable __scrt_renderall_uniform_colors
llvm::Value *CGScoutRuntime::RenderallUniformColorsGlobal(CodeGenFunction &CGF) {
  std::string varName = "__scrt_renderall_uniform_colors";
  llvm::Type *flt4PtrTy = llvm::PointerType::get(
      llvm::VectorType::get(llvm::Type::getFloatTy(CGM.getLLVMContext()), 4), 0);

  if (!CGM.getModule().getNamedGlobal(varName)) {

    new llvm::GlobalVariable(CGM.getModule(),
        flt4PtrTy,
        false,
        llvm::GlobalValue::ExternalLinkage,
        0,
        varName);
  }

  llvm::Value *Color = CGM.getModule().getNamedGlobal(varName);

  llvm::Value *ColorPtr  = CGF.Builder.CreateAlloca(flt4PtrTy, 0, "color.ptr");
  CGF.Builder.CreateStore(CGF.Builder.CreateLoad(Color, "runtime.color"), ColorPtr);

  return ColorPtr;
}


llvm::Type *CGScoutRuntime::convertScoutSpecificType(const Type *T) {
  llvm::LLVMContext& Ctx = CGM.getLLVMContext();
  if (T->isScoutWindowType()) {
    return llvm::PointerType::get(llvm::StructType::create(Ctx, "scout.window_t"), 0);
  } else if (T->isScoutImageType()) {
    return llvm::PointerType::get(llvm::StructType::create(Ctx, "scout.image_t"), 0);
  } else if (T->isScoutQueryType()) {
    llvm::StructType* qt = llvm::StructType::create(Ctx, "scout.query_t");
    qt->setBody(CGM.VoidPtrTy, CGM.VoidPtrTy, NULL);
    return qt;
  }
  else {
    llvm_unreachable("Unexpected scout type!");
    return 0;
  }
}

llvm::Function *CGScoutRuntime::EmitRuntimeInitFunc(SourceLocation Loc,
                                                    CodeGenFunction *CGF) {
  auto InitFunctionTy = llvm::FunctionType::get(CGM.VoidTy, /* isVarArg */ false);
  auto InitFunction   = CGM.CreateGlobalInitOrDestructFunction(InitFunctionTy,
                                                               ".__sc_init__.");
  CodeGenFunction InitCGF(CGM);
  FunctionArgList ArgList; /* empty */
  InitCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, InitFunction,
                        CGM.getTypes().arrangeNullaryFunction(), ArgList, Loc);

  EmitRuntimeInitializationCall(InitCGF, Loc);
  
  InitCGF.FinishFunction();
  return InitFunction;
}

void CGScoutRuntime::EmitRuntimeInitializationCall(CodeGenFunction &CGF,
                                                   SourceLocation Loc) {

  std::string funcName("__scrt_init_cpu");
  // main high-level runtime initialization.
  llvm::Function *rtInitFn = CGM.getModule().getFunction(funcName);
  if (!rtInitFn) {
    std::vector<llvm::Type*> args;
    llvm::FunctionType *FTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()), args, false);
    rtInitFn = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                      funcName, &CGM.getModule());
  }
  
  std::vector<llvm::Value*> params; /* void */ 
  CGF.Builder.CreateCall(rtInitFn, params, "");
  //return rtInitFn;
}
