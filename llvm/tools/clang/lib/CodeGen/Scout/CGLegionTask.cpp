/*
 *
 * ###########################################################################
 *
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
 *
 */

#include <stdio.h>
#include <cassert>
#include "Scout/CGLegionTask.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "Scout/ASTVisitors.h"
#include "legion/lsci.h"

using namespace clang;
using namespace CodeGen;

CGLegionTask::CGLegionTask(const FunctionDecl* FD, llvm::Function* TF, CodeGenModule& codeGenModule, 
  CGBuilderTy& builder, CodeGenFunction* codeGenFunction)
  :CGM(codeGenModule), B(builder), R(codeGenModule.getLegionRuntime()), CGF(codeGenFunction) {

  assert(FD && TF);

  funcDecl = FD;
  meshDecl = NULL;
  taskFunc = TF;
  legionTaskInitFunc = NULL;
  legionTaskFunc = NULL;
  meshType = NULL;
  fields = {};
  firstField = NULL;

  taskId = CGM.NextLegionTaskId++;
  legionContext = NULL;
  legionRuntime = NULL;
  meshTaskArgs = NULL;

  indexLauncher = NULL; 
  argMap = NULL;
  taskDeclVisitor = NULL;
  meshPos = 0;
  meshPtr = NULL;

  taskArgs = NULL;
  task = NULL;
  regions = NULL;
  taskFuncArgs = {};
  mesh = NULL;

}
  
void CGLegionTask::EmitLegionTask() {

  auto aitr = taskFunc->arg_begin();
 
  // do some checking of arguments 
  llvm::PointerType* meshPtrType = dyn_cast<llvm::PointerType>(aitr->getType());
  assert(meshPtrType && "Expected a mesh ptr");
  
  meshType = dyn_cast<llvm::StructType>(meshPtrType->getElementType());
  assert(meshType && "Expected a mesh");

  llvm::BasicBlock* prevBlock = B.GetInsertBlock();
  llvm::BasicBlock::iterator prevPoint = B.GetInsertPoint();

  EmitLegionTaskInitFunction();

  legionContext = NULL;
  legionRuntime = NULL;

  EmitLegionTaskFunction();

  B.SetInsertPoint(prevBlock, prevPoint);
 
  assert(legionTaskFunc); 
  CGM.regTaskInLsciMainFunction(taskId, legionTaskFunc);

}

void CGLegionTask::EmitLegionTaskInitFunction() {

  // Create the function we will be adding to
  EmitLegionTaskInitFunctionStart();

  assert(legionTaskInitFunc);

  // get, meshPtr, legionContext and legionRuntime  
  auto aitr = legionTaskInitFunc->arg_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      meshPtr = aitr;
    }
    ++aitr;
  }
  
  legionContext = aitr++;
  legionRuntime = aitr;
 

  // emit entry block 
  llvm::LLVMContext& llvmContext = CGM.getLLVMContext();

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvmContext, "entry", legionTaskInitFunc);
  B.SetInsertPoint(entry);

  taskDeclVisitor = new TaskDeclVisitor(funcDecl);

  EmitUnimeshGetVecByNameFuncCalls();

  EmitArgumentMapCreateFuncCall();

  llvm::Value* Zero = llvm::ConstantInt::get(R.Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(R.Int64Ty, 1);
  
  llvm::Value* iPtr = B.CreateAlloca(R.Int64Ty, 0, "i.ptr");
  B.CreateStore(Zero, iPtr);

  meshTaskArgs = CreateMeshTaskArgs();

  llvm::BasicBlock* cond = llvm::BasicBlock::Create(llvmContext, "cond", legionTaskInitFunc);
  llvm::BasicBlock* loop = llvm::BasicBlock::Create(llvmContext, "loop", legionTaskInitFunc);
  llvm::BasicBlock* merge = llvm::BasicBlock::Create(llvmContext, "merge", legionTaskInitFunc);
  
  B.CreateBr(cond);
  B.SetInsertPoint(cond);
  llvm::Value* i = B.CreateLoad(iPtr);
  
  assert(firstField);
  llvm::Value* launchDmn =
  B.CreateStructGEP(firstField, LSCI_VECTOR_LAUNCH_DOMAIN, "launchDomain.ptr");
  llvm::Value* volume =
  B.CreateLoad(B.CreateStructGEP(launchDmn, LSCI_DOMAIN_VOLUME), "volume");

  llvm::Value* cmp = B.CreateICmpULT(i, volume);
  B.CreateCondBr(cmp, loop, merge);
  
  B.SetInsertPoint(loop);

  // take this out for liblsci v2 
  EmitSubgridBoundsAtSetFuncCall(i);
 
  EmitArgumentMapSetPointFuncCall(i); 

  B.CreateStore(B.CreateAdd(i, One), iPtr);
  B.CreateBr(cond);
  
  B.SetInsertPoint(merge);
  
  EmitIndexLauncherCreateFuncCall();
 
  EmitAddMeshRegionReqAndFieldFuncCalls(); 

  // done with taskDeclVisitor 
  delete taskDeclVisitor;  
  taskDeclVisitor = NULL;

  EmitAddVectorRegionReqAndFieldFuncCalls();  


  EmitExecuteIndexSpaceFuncCall();

  B.CreateRetVoid();

  // not valid outside LegionTaskInitFunction
  legionContext = NULL;
  legionRuntime = NULL;
  meshTaskArgs = NULL;
}

 
void CGLegionTask::EmitLegionTaskInitFunctionStart()
{
 
  assert(funcDecl && taskFunc); 

  llvm::LLVMContext& llvmContext = CGM.getLLVMContext();
 
  TypeVec params;
 
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
        params.push_back(R.PointerTy(R.UnimeshTy));
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
  llvm::NamedMDNode* tasks =
  CGM.getModule().getOrInsertNamedMetadata("scout.tasks");
  
  
  SmallVector<llvm::Value*, 3> taskInfo;
  taskInfo.push_back(llvm::ConstantInt::get(R.Int32Ty, taskId));
  taskInfo.push_back(taskFunc);
  taskInfo.push_back(legionTaskInitFunc);
 
  tasks->addOperand(llvm::MDNode::get(llvmContext, taskInfo));
}

void CGLegionTask::EmitUnimeshGetVecByNameFuncCalls()
{
  assert(funcDecl && meshPtr && legionContext && legionRuntime);

  // visit the function to determine read and write uses of the mesh fields 
  taskDeclVisitor->VisitStmt(funcDecl->getBody());
  
  const MeshFieldMap& LHS = taskDeclVisitor->getLHSmap();
  const MeshFieldMap& RHS = taskDeclVisitor->getRHSmap();
  
  const MeshNameMap& MN = taskDeclVisitor->getMeshNamemap();
  assert(MN.size() == 1 && "expected one mesh");
  
  const std::string& meshName = MN.begin()->first;

  ValueVec args;

  // generate calls to lsci_unimesh_get_vec_by_name() for each field used

  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* fd = *itr;

    std::string fieldName = meshName + "." + fd->getName().str();

    bool read = RHS.find(fieldName) != RHS.end();
    bool write = LHS.find(fieldName) != LHS.end();

    if(read || write) {
      llvm::Value* field = B.CreateAlloca(R.VectorTy, 0, fd->getName() + ".ptr");
      fields.push_back(field);
      
      args = {meshPtr, B.CreateGlobalStringPtr(fd->getName()),
        field, legionContext, legionRuntime};
      
      B.CreateCall(R.UnimeshGetVecByNameFunc(), args);
      
      if(!firstField){
        firstField = field;
      }
    }
    else{
      fields.push_back(0);
    }
  }
  assert(firstField && "no mesh fields accessed");
}

 
void CGLegionTask::EmitArgumentMapCreateFuncCall() {

  argMap = B.CreateAlloca(R.ArgumentMapTy, 0, "argMap.ptr");
  ValueVec args;
  args = {argMap};
  llvm::Function* f = R.ArgumentMapCreateFunc();
  B.CreateCall(f, args);
}

llvm::Value* CGLegionTask::CreateMeshTaskArgs() {

  assert(firstField && meshPtr);
 
  llvm::Value* len =
  B.CreateLoad(B.CreateStructGEP(firstField, LSCI_VECTOR_SUBGRID_BOUNDS_LEN), "len");
  
  llvm::Value* rank =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_DIMS), "rank");
  
  llvm::Value* width =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_WIDTH), "width");
  
  llvm::Value* height =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_HEIGHT), "height");
  
  llvm::Value* depth =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_DEPTH), "depth");
  
  llvm::Value* mtargs =
  B.CreateAlloca(R.MeshTaskArgsTy, 0, "meshTaskArgs.ptr");

  B.CreateStore(rank, B.CreateStructGEP(mtargs, LSCI_MTARGS_RANK));
  B.CreateStore(width, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_WIDTH));
  B.CreateStore(height, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_HEIGHT));
  B.CreateStore(depth, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_DEPTH));
  B.CreateStore(len, B.CreateStructGEP(mtargs, LSCI_MTARGS_SUBGRID_LEN));
  
  return mtargs;
}

// Not used in liblsci v2
void CGLegionTask::EmitSubgridBoundsAtSetFuncCall(llvm::Value* i) {

  assert(firstField && meshTaskArgs && i);
 
  llvm::Value* bounds =
  B.CreateLoad(B.CreateStructGEP(firstField, LSCI_VECTOR_SUBGRID_BOUNDS), "bounds");
 
  llvm::Value* rect1dStoragePtr = B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_SUBGRID_BOUNDS);
  ValueVec args;
  args = {bounds, i, rect1dStoragePtr};
  B.CreateCall(R.SubgridBoundsAtSetFunc(), args);
}

void CGLegionTask::EmitArgumentMapSetPointFuncCall(llvm::Value* i) {  

  assert(argMap && meshTaskArgs && i);

  ValueVec args;

  args = {argMap, i, B.CreateBitCast(meshTaskArgs, R.VoidPtrTy),
    llvm::ConstantInt::get(R.Int64Ty, sizeof(lsci_mesh_task_args_t))};
  
  B.CreateCall(R.ArgumentMapSetPointFunc(), args);
}

void CGLegionTask::EmitIndexLauncherCreateFuncCall() {

  assert(firstField && argMap );

  indexLauncher =
  B.CreateAlloca(R.IndexLauncherTy, 0, "indexLauncher");
  
  llvm::Value* launchDomain =
  B.CreateStructGEP(firstField, LSCI_VECTOR_LAUNCH_DOMAIN, "launchDomain.ptr");

  llvm::Value* TaskId = llvm::ConstantInt::get(R.Int32Ty, taskId);

  llvm::Value* ConstantZero = llvm::ConstantInt::get(R.Int64Ty, 0);

  ValueVec args;
  args = {indexLauncher, TaskId, launchDomain,
    R.GetNull(R.Int8Ty), ConstantZero, argMap};

  B.CreateCall(R.IndexLauncherCreateFunc(), args);

}

// When emitting IR for LegionTaskInitFunction(), you need to emit
// region requirements and add fields for the first
// parameter, which is a mesh.
void CGLegionTask::EmitAddMeshRegionReqAndFieldFuncCalls() { 

  assert(funcDecl && meshDecl && (fields.size() > 0) && indexLauncher && taskDeclVisitor);

  // get mesh name
  const MeshNameMap& MN = taskDeclVisitor->getMeshNamemap();
  assert(MN.size() == 1 && "expected one mesh");
  const std::string& meshName = MN.begin()->first;

  const MeshFieldMap& LHS = taskDeclVisitor->getLHSmap();
  const MeshFieldMap& RHS = taskDeclVisitor->getRHSmap();

  ValueVec args;

  uint32_t j = 0;
  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    
    MeshFieldDecl* fd = *itr;
    llvm::Value* field = fields[j];
    
    if(field) {
      
      assert(fd);
      std::string fieldName = meshName + "." + fd->getName().str();
      bool read = RHS.find(fieldName) != RHS.end();
      bool write = LHS.find(fieldName) != LHS.end();
     
      llvm::Value* mode = read ? (write ? R.ReadWriteVal : R.ReadOnlyVal) : R.WriteDiscardVal;
      
      llvm::Value* fieldId =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
      
      llvm::Value* logicalPartition =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
      
      llvm::Value* logicalRegion =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
      
      args =
      {indexLauncher, logicalRegion, llvm::ConstantInt::get(R.Int32Ty, 0),
        mode, R.ExclusiveVal, logicalPartition};
      
      B.CreateCall(R.AddRegionRequirementFunc(), args);
      
      args =
      {indexLauncher, llvm::ConstantInt::get(R.Int32Ty, j), fieldId};
      B.CreateCall(R.AddFieldFunc(), args);
    }
    
    ++j;
  }
}

// While emitting IR for LegionTaskInitFunction(), you need to emit
// region requirements and add fields for the other
// task parameters, which can be structs or scalars.
void CGLegionTask::EmitAddVectorRegionReqAndFieldFuncCalls() { 

  assert(legionTaskInitFunc && funcDecl && indexLauncher);

  ValueVec args;
  llvm::Value* One = llvm::ConstantInt::get(R.Int64Ty, 1);

  uint32_t j = fields.size();
  auto aitr = legionTaskInitFunc->arg_begin();
  auto pitr = funcDecl->param_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      ++aitr;
      ++pitr;
      continue;
    }
    
    const llvm::Type* t = aitr->getType();
    llvm::Value* mode;
    
    if((*pitr)->getType().isConstQualified()){
      mode = R.ReadOnlyVal;
    }
    else{
      mode = R.ReadWriteVal;
    }

    if(const llvm::PointerType* pt = dyn_cast<llvm::PointerType>(t)){
      if(const llvm::StructType* st = dyn_cast<llvm::StructType>(pt->getElementType())){
        for(auto eitr = st->element_begin(), eitrEnd = st->element_end();
            eitr != eitrEnd; ++eitr){
          const llvm::Type* et = *eitr;

          llvm::Value* len;
          
          if(const llvm::ArrayType* at = dyn_cast<llvm::ArrayType>(et)){
            len = llvm::ConstantInt::get(R.Int64Ty, at->getNumElements());
            et = at->getElementType();
          }
          else{
            len = One;
          }
          
          llvm::Value* field = B.CreateAlloca(R.VectorTy, 0);
          
          if(et->isFloatTy()){
            args = {field, len, R.TypeFloatVal, legionContext, legionRuntime};
          }
          else if(et->isDoubleTy()){
            args = {field, len, R.TypeDoubleVal, legionContext, legionRuntime};
          }
          else if(et->isIntegerTy(32)){
            args = {field, len, R.TypeInt32Val, legionContext, legionRuntime};
          }
          else if(et->isIntegerTy(64)){
            args = {field, len, R.TypeInt64Val, legionContext, legionRuntime};
          }
          else{
            assert(false && "invalid task scalar param type");
          }
          
          B.CreateCall(R.VectorCreateFunc(), args);
          
          llvm::Value* fieldId =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
          
          llvm::Value* logicalPartition =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
          
          llvm::Value* logicalRegion =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
          
          args =
          {indexLauncher, logicalRegion, llvm::ConstantInt::get(R.Int32Ty, 0),
            mode, R.ExclusiveVal, logicalPartition};
          
          B.CreateCall(R.AddRegionRequirementFunc(), args);
          
          args =
          {indexLauncher, llvm::ConstantInt::get(R.Int32Ty, j++), fieldId};
          B.CreateCall(R.AddFieldFunc(), args);
          
        }
      }
      else{
        assert(false && "invalid pointer type");
      }
    }
    else{
      llvm::Value* field = B.CreateAlloca(R.VectorTy, 0, aitr->getName());
      
      if(t->isFloatTy()){
        args = {field, One, R.TypeFloatVal, legionContext, legionRuntime};
      }
      else if(t->isDoubleTy()){
        args = {field, One, R.TypeDoubleVal, legionContext, legionRuntime};
      }
      else if(t->isIntegerTy(32)){
        args = {field, One, R.TypeInt32Val, legionContext, legionRuntime};
      }
      else if(t->isIntegerTy(64)){
        args = {field, One, R.TypeInt64Val, legionContext, legionRuntime};
      }
      else{
        assert(false && "invalid task scalar param type");
      }
      
      B.CreateCall(R.VectorCreateFunc(), args);
      
      llvm::Value* fieldId =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
      
      llvm::Value* logicalPartition =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
      
      llvm::Value* logicalRegion =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
      
      args =
      {indexLauncher, logicalRegion, llvm::ConstantInt::get(R.Int32Ty, 0),
        mode, R.ExclusiveVal, logicalPartition};
      
      B.CreateCall(R.AddRegionRequirementFunc(), args);
      
      args =
      {indexLauncher, llvm::ConstantInt::get(R.Int32Ty, j++), fieldId};
      B.CreateCall(R.AddFieldFunc(), args);
    }
    
    ++aitr;
    ++pitr;
  }
} 

void CGLegionTask::EmitExecuteIndexSpaceFuncCall(){

  assert(legionRuntime && legionContext && indexLauncher);

  ValueVec args;
  args = {legionRuntime, legionContext, indexLauncher};
  B.CreateCall(R.ExecuteIndexSpaceFunc(), args);
}
  

void CGLegionTask::EmitLegionTaskFunction() { 


  // sets task args
  EmitLegionTaskFunctionStart();

  llvm::LLVMContext& llvmContext = CGM.getLLVMContext();

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvmContext, "entry", legionTaskFunc);
  B.SetInsertPoint(entry);

  task = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_TASK), "task");
  legionContext = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_CONTEXT), "legionContext");
  legionRuntime = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_RUNTIME), "legionRuntime");

  // for liblsci v2
  //EmitGetIndexSpaceDomainFuncCall();

  EmitScoutMesh();

  regions = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_REGIONS), "regions");

  EmitMeshRawRectPtr1dFuncCalls(); 

  EmitVectorRawRectPtr1dFuncCalls(); 

  EmitTaskFuncCall();

  B.CreateRetVoid();
}


void CGLegionTask::EmitLegionTaskFunctionStart() { 

  llvm::LLVMContext& llvmContext = CGM.getLLVMContext();

  //fetch or create type for lsci_task_args_t
  llvm::StructType *TaskArgsTy = CGM.getModule().getTypeByName("struct.lsci_task_args_t");
  if (!TaskArgsTy) {
    std::vector<llvm::Type*> structMembers = {
      R.ContextTy,
      R.RuntimeTy,
      R.Int32Ty,
      R.Int64Ty,
      R.PhysicalRegionsTy,
      R.VoidPtrTy};
    TaskArgsTy = llvm::StructType::create(llvmContext, structMembers, "struct.lsci_task_args_t");
  }

  llvm::PointerType* TaskArgsPtrTy = llvm::PointerType::get(TaskArgsTy, 0);

  // use lsci_task_args_t to create main_task function type and function
  TypeVec params = {TaskArgsPtrTy};

  llvm::FunctionType* funcType = llvm::FunctionType::get(R.VoidTy, params, false);

  legionTaskFunc = llvm::Function::Create(funcType,
      llvm::Function::ExternalLinkage,
      "LegionTaskFunction",
      &CGM.getModule());

  auto aitr = legionTaskFunc->arg_begin();
  taskArgs = aitr;
  taskArgs->setName("task_args_ptr");
}


// for liblsci v2
void CGLegionTask::EmitGetIndexSpaceDomainFuncCall() {

  assert(legionRuntime && legionContext && task);

  uint32_t j = 0;
  ValueVec args;

  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    
    llvm::Value* field = fields[j];

    if(field){
      llvm::Value* domain =  B.CreateAlloca(R.DomainTy, 0, "domain");
      args = {legionRuntime, legionContext, task, llvm::ConstantInt::get(R.Int64Ty, 0), domain};
      B.CreateCall(R.GetIndexSpaceDomainFunc(), args);
    }
  }
}

void CGLegionTask::EmitScoutMesh() { 

  // load the taskArgs pointer
  llvm::Value* taskArgsAddr = B.CreateAlloca(R.PointerTy(R.TaskArgsTy), 0, "task_args.addr");
  B.CreateAlignedStore(taskArgs, taskArgsAddr, 8);

  // load the mesh task args ptr from taskArgs local_argsp field
  llvm::LoadInst* loadTaskArgsPtr = B.CreateAlignedLoad(taskArgsAddr, 8, "task_args_loaded.ptr");
  llvm::Value* meshTaskArgsAddr = B.CreateAlloca(R.PointerTy(R.MeshTaskArgsTy), 0, "mtargs.addr");
  llvm::Value* localArgsp = B.CreateStructGEP(loadTaskArgsPtr, LSCI_TARGS_LOCAL_ARGSP); 
  llvm::LoadInst* loadedLocalArgsp = B.CreateAlignedLoad(localArgsp, 8, "local_argsp.loaded");

  // must cast to a lsci_mesh_task_args_t* , since it is a void*
  llvm::Value* loadedMtargsp = B.CreateBitCast(loadedLocalArgsp, R.PointerTy(R.MeshTaskArgsTy), "mtargsp.loaded"); 
  B.CreateAlignedStore(loadedMtargsp, meshTaskArgsAddr, 8);
  meshTaskArgs = B.CreateAlignedLoad(meshTaskArgsAddr, 8, "mesh_task_args.ptr");

  mesh = B.CreateAlloca(meshType, 0, "mesh.ptr");
}

void CGLegionTask::EmitMeshRawRectPtr1dFuncCalls() {

  assert(meshDecl && (fields.size() > 0) && task && legionContext && legionRuntime);

 
  uint32_t j = 0;
  ValueVec args;

  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    
    llvm::Value* field = fields[j];

    llvm::Value* mf = B.CreateStructGEP(mesh, j);
    
    if(field){
      MeshFieldDecl* fd = *itr;
      llvm::Type* ft = CGF->ConvertType(fd->getType());
      
      args = {regions};
      
      if(ft->isFloatTy()){
        args.push_back(R.TypeFloatVal);
      }
      else if(ft->isDoubleTy()){
        args.push_back(R.TypeDoubleVal);
      }
      else if(ft->isIntegerTy(32)){
        args.push_back(R.TypeInt32Val);
      }
      else if(ft->isIntegerTy(64)){
        args.push_back(R.TypeInt64Val);
      }
      else{
        assert(false && "unhandled mesh field type");
      }
      
      args.push_back(llvm::ConstantInt::get(R.Int64Ty, j));
      args.push_back(llvm::ConstantInt::get(R.Int32Ty, 0));
      
      args.push_back(task); 
      args.push_back(legionContext); 
      args.push_back(legionRuntime); 

      llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
      llvm::Value* cv = B.CreateBitCast(fp, meshType->getTypeAtIndex(j));
      
      B.CreateStore(cv, mf);
    }
    else{
      llvm::PointerType* pt = dyn_cast<llvm::PointerType>(mf->getType());
      assert(pt && "expected a pointer type");
      
      llvm::PointerType* et = dyn_cast<llvm::PointerType>(pt->getElementType());
      assert(et && "expected a pointer element type");
      
      B.CreateStore(llvm::ConstantPointerNull::get(et), mf);
    }
    
    ++j;
  }
}
 
void CGLegionTask::EmitVectorRawRectPtr1dFuncCalls() {

  assert((fields.size() > 0) && legionTaskInitFunc && funcDecl && mesh && regions && legionContext && legionRuntime);

  ValueVec args;
  uint32_t j = fields.size();
 
  // create calls to  lsci_raw_rect_ptr_1d
  uint32_t k = j;
  auto aitr = legionTaskInitFunc->arg_begin();
  auto pitr = funcDecl->param_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      taskFuncArgs.push_back(mesh);
      ++aitr;
      ++pitr;
      continue;
    }
    
    llvm::Type* t = aitr->getType();
    
    if(llvm::PointerType* pt = dyn_cast<llvm::PointerType>(t)){
      if(llvm::StructType* st = dyn_cast<llvm::StructType>(pt->getElementType())){
        llvm::Value* arg = B.CreateAlloca(st, 0, aitr->getName());
        taskFuncArgs.push_back(arg);
        
        size_t ei = 0;
        for(auto eitr = st->element_begin(), eitrEnd = st->element_end();
            eitr != eitrEnd; ++eitr){
          llvm::Type* et = *eitr;
          
          bool array;
          if(const llvm::ArrayType* at = dyn_cast<llvm::ArrayType>(et)){
            et = at->getElementType();
            array = true;
          }
          else{
            array = false;
          }
          
          args = {regions};
          
          if(et->isFloatTy()){
            args.push_back(R.TypeFloatVal);
          }
          else if(et->isDoubleTy()){
            args.push_back(R.TypeDoubleVal);
          }
          else if(et->isIntegerTy(32)){
            args.push_back(R.TypeInt32Val);
          }
          else if(et->isIntegerTy(64)){
            args.push_back(R.TypeInt64Val);
          }
          else{
            assert(false && "invalid task scalar param type");
          }
          
          args.push_back(llvm::ConstantInt::get(R.Int64Ty, k++));
          args.push_back(llvm::ConstantInt::get(R.Int32Ty, 0));
          args.push_back(task); 
          args.push_back(legionContext); 
          args.push_back(legionRuntime); 
  
          llvm::Value* ep = B.CreateStructGEP(arg, ei++);
          llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
          llvm::Value* cv = B.CreateBitCast(fp, R.PointerTy(et));

          if(array){
            ep = B.CreateBitCast(ep, R.PointerTy(R.PointerTy(et)));
          }
          else{
            cv = B.CreateLoad(cv);
          }
          
          B.CreateStore(cv, ep);
        }
      }
      else{
        assert(false && "invalid pointer type");
      }
    }
    else{
      llvm::Value* arg = B.CreateAlloca(t, 0, aitr->getName());
      
      args = {regions};
      
      if(t->isFloatTy()){
        args.push_back(R.TypeFloatVal);
      }
      else if(t->isDoubleTy()){
        args.push_back(R.TypeDoubleVal);
      }
      else if(t->isIntegerTy(32)){
        args.push_back(R.TypeInt32Val);
      }
      else if(t->isIntegerTy(64)){
        args.push_back(R.TypeInt64Val);
      }
      else{
        assert(false && "invalid task scalar param type");
      }
      
      args.push_back(llvm::ConstantInt::get(R.Int64Ty, k++));
      args.push_back(llvm::ConstantInt::get(R.Int32Ty, 0));
      args.push_back(task); 
      args.push_back(legionContext); 
      args.push_back(legionRuntime); 
      
      llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
      llvm::Value* cv = B.CreateBitCast(fp, R.PointerTy(t));
      B.CreateStore(B.CreateLoad(cv), arg);
      
      taskFuncArgs.push_back(B.CreateLoad(arg));
    }
    
    ++aitr;
    ++pitr;
  }
}

// This emits the code that was in the original function preceded by the "task" indicator in 
// the Scout program
void CGLegionTask::EmitTaskFuncCall() {

  assert((fields.size() > 0) && meshTaskArgs && mesh && taskFunc && (taskFuncArgs.size() > 0));
  
  // Must get dims and rank from mesh_task_args_t, since we
  // are parsing this function before the mesh has
  // been instantiated, so we don't know dimensions.
  // The "mesh" struct has pointers to the data for fields first, 
  //then width, height, depth, rank stored after the fields.

  int32_t j = fields.size();

  llvm::Value* TAwidthPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_WIDTH), R.PointerTy(R.Int32Ty));
  llvm::Value* TAwidth = B.CreateAlignedLoad(TAwidthPtr, 8, "task_arg_width");
  llvm::Value* widthPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAwidth, widthPtr, 4);

  llvm::Value* TAheightPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_HEIGHT), R.PointerTy(R.Int32Ty));
  llvm::Value* TAheight = B.CreateAlignedLoad(TAheightPtr, 8, "task_arg_height");
  llvm::Value* heightPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAheight, heightPtr, 4);

  llvm::Value* TAdepthPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_DEPTH), R.PointerTy(R.Int32Ty));
  llvm::Value* TAdepth = B.CreateAlignedLoad(TAdepthPtr, 8, "task_arg_depth");
  llvm::Value* depthPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAdepth, depthPtr, 4);

  llvm::Value* TArankPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_RANK), R.PointerTy(R.Int32Ty));
  llvm::Value* TArank = B.CreateAlignedLoad(TArankPtr, 8, "task_arg_rank");
  llvm::Value* rankPtr = B.CreateStructGEP(mesh, j);
  B.CreateAlignedStore(TArank, rankPtr, 4);

  B.CreateCall(taskFunc, taskFuncArgs);
}

