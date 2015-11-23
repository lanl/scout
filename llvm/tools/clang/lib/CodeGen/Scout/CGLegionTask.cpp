/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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
 *
 */

#include <stdio.h>
#include <cassert>
#include "Scout/CGLegionTask.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "Scout/ASTVisitors.h"

using namespace clang;
using namespace CodeGen;

CGLegionTask::CGLegionTask(const FunctionDecl* FD,
                           llvm::Function* TF,
                           CodeGenModule& codeGenModule,
                           CGBuilderTy& builder,
                           CodeGenFunction* codeGenFunction)
: CGM(codeGenModule),
B(builder),
R(codeGenModule.getLegionCRuntime()),
CGF(codeGenFunction){

  assert(FD && TF);

  funcDecl = FD;
  meshDecl = nullptr;
  taskFunc = TF;
  legionTaskInitFunc = nullptr;
  legionTaskFunc = nullptr;
  meshType = nullptr;

  taskId = CGM.NextLegionTaskId++;
  legionContext = nullptr;
  legionRuntime = nullptr;
}
  
void CGLegionTask::EmitLegionTask() {
  using namespace llvm;
  
  auto aitr = taskFunc->arg_begin();
 
  // do some checking of arguments 
  llvm::PointerType* meshPtrType = dyn_cast<llvm::PointerType>(aitr->getType());
  assert(meshPtrType && "Expected a mesh ptr");
  
  meshType = dyn_cast<StructType>(meshPtrType->getElementType());
  assert(meshType && "Expected a mesh");

  BasicBlock* prevBlock = B.GetInsertBlock();
  BasicBlock::iterator prevPoint = B.GetInsertPoint();

  EmitLegionTaskInitFunction();

  legionContext = nullptr;
  legionRuntime = nullptr;

  EmitLegionTaskFunction();

  B.SetInsertPoint(prevBlock, prevPoint);
 
  assert(legionTaskFunc); 
  
  CGM.regTaskInLegionMainFunction(taskId, legionTaskFunc);
}

void CGLegionTask::EmitLegionTaskInitFunction() {
  using namespace std;
  using namespace llvm;
  
  assert(funcDecl && taskFunc);
  
  LLVMContext& C = CGM.getLLVMContext();
  
  TypeVec params;
  
  size_t meshPos;
  Value* meshPtr;
  
  // Note that we allow more than one task argument.
  // The mesh is the first argument, then you can have structs and/or scalars
  // after that.
  size_t idx = 0;
  const UniformMeshType* mt = 0;
  auto aitr = taskFunc->arg_begin();
  
  for(auto itr = funcDecl->param_begin(), itrEnd = funcDecl->param_end();
      itr != itrEnd; ++itr){
    ParmVarDecl* pd = *itr;
    const Type* t = pd->getType().getTypePtr();
  
    if(t->isPointerType() || t->isReferenceType()){
      if(const UniformMeshType* ct = dyn_cast<UniformMeshType>(t->getPointeeType())){
        assert(!mt && "more than one mesh param found");
        mt = ct;
        meshPos = idx;
        params.push_back(R.ScUniformMeshTy);
      }
      else{
        params.push_back(aitr->getType());
      }
    }
    else{
      params.push_back(aitr->getType());
    }
    ++idx;
    ++aitr;
  }
  
  assert(mt && "expected a mesh param");
  
  meshDecl = mt->getDecl();
  
  params.push_back(R.ContextTy);
  params.push_back(R.RuntimeTy);
  llvm::FunctionType* ft = llvm::FunctionType::get(R.VoidTy, params, false);
  
  legionTaskInitFunc =
  llvm::Function::Create(ft,
                         llvm::Function::ExternalLinkage,
                         "LegionTaskInitFunction",
                         &CGM.getModule());
  
  // adding metadata
  NamedMDNode* tasks =
  CGM.getModule().getOrInsertNamedMetadata("scout.tasks");
  
  
  SmallVector<Metadata*, 3> taskInfo;
  taskInfo.push_back(ConstantAsMetadata::get(ConstantInt::get(R.Int32Ty, taskId)));
  taskInfo.push_back(ValueAsMetadata::get(taskFunc));
  taskInfo.push_back(ValueAsMetadata::get(legionTaskInitFunc));
  
  tasks->addOperand(MDNode::get(C, taskInfo));
  
  assert(legionTaskInitFunc);

  // get, meshPtr, legionContext and legionRuntime  
  aitr = legionTaskInitFunc->arg_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      meshPtr = &*aitr;
    }
    ++aitr;
  }
  
  legionContext = &*aitr;
  aitr++;
  legionRuntime = &*aitr;
 
  // emit entry block
  BasicBlock* entry = BasicBlock::Create(C, "entry", legionTaskInitFunc);
  B.SetInsertPoint(entry);
  
  ValueVec args = {meshPtr, ConstantInt::get(R.TaskIdTy, taskId)};
  
  Value* launcher =
  B.CreateCall(R.ScUniformMeshCreateLauncherFunc(), args);
  
  TaskDeclVisitor taskDeclVisitor(funcDecl);
  
  // visit the function to determine read and write uses of the mesh fields
  taskDeclVisitor.VisitStmt(funcDecl->getBody());
  
  const MeshFieldMap& LHS = taskDeclVisitor.getLHSmap();
  const MeshFieldMap& RHS = taskDeclVisitor.getRHSmap();
  
  const MeshNameMap& MN = taskDeclVisitor.getMeshNamemap();
  assert(MN.size() == 1 && "expected one mesh");
  
  const string& meshName = MN.begin()->first;
  
  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* fd = *itr;
    
    string fieldName = meshName + "." + fd->getName().str();
    
    bool read = RHS.find(fieldName) != RHS.end();
    bool write = LHS.find(fieldName) != LHS.end();
    
    Value* mode;
    
    if(read){
      if(write){
        mode = R.ReadWriteVal;
      }
      else{
        mode = R.ReadOnlyVal;
      }
    }
    else if(write){
      mode = R.WriteOnlyVal;
    }
    else{
      continue;
    }
    
    args = {launcher, B.CreateGlobalStringPtr(fd->getName()), mode};
    B.CreateCall(R.ScUniformMeshLauncherAddFieldFunc(), args);
  }
  
  args = {legionContext, legionRuntime, launcher};
  B.CreateCall(R.ScUniformMeshLauncherExecuteFunc(), args);
  
  B.CreateRetVoid();

  // not valid outside LegionTaskInitFunction
  legionContext = nullptr;
  legionRuntime = nullptr;
}

void CGLegionTask::EmitLegionTaskFunction() { 
  using namespace llvm;
  
  legionTaskFunc = llvm::Function::Create(R.VoidTaskFuncTy,
                                          llvm::Function::ExternalLinkage,
                                          "LegionTaskFunction",
                                          &CGM.getModule());
  
  auto aitr = legionTaskFunc->arg_begin();
  Value* task = &*aitr;
  aitr++;
  Value* regions = &*aitr;
  aitr++;
  Value* numRegions = &*aitr;
  aitr++;
  Value* context = &*aitr;
  aitr++;
  Value* runtime = &*aitr;
  
  LLVMContext& C = CGM.getLLVMContext();
  BasicBlock* entry = BasicBlock::Create(C, "entry", legionTaskFunc);
  B.SetInsertPoint(entry);
  
  ValueVec args = {task, regions, numRegions, context, runtime};
  Value* mp = B.CreateCall(R.ScUniformMeshReconstructFunc(), args);
  
  args = {B.CreateBitCast(mp, R.PointerTy(meshType))};
  B.CreateCall(taskFunc, args);
  
  B.CreateRetVoid();
}
