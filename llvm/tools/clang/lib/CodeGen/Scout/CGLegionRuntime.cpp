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

#include "Scout/CGLegionRuntime.h"
#include "CodeGenFunction.h"

using namespace std;
using namespace CodeGen;

CGLegionRuntime::CGLegionRuntime(CodeGen::CodeGenModule &CGM) : CGM(CGM){
  llvm::LLVMContext& context = CGM.getLLVMContext();
  
  Int8Ty = llvm::Type::getInt8Ty(context);
  Int32Ty = llvm::Type::getInt32Ty(context);
  Int64Ty = llvm::Type::getInt64Ty(context);
  VoidPtrTy = PointerTy(Int8Ty);
  
  RuntimeTy = VoidPtrTy;
  ContextTy = VoidPtrTy;
  LogicalRegionTy = VoidPtrTy;
  LogicalPartitionTy = VoidPtrTy;
  IndexSpaceTy = VoidPtrTy;
  DomainHandleTy = VoidPtrTy;
  PhysicalRegionsTy = VoidPtrTy;
  Rect1dTy = VoidPtrTy;
  FieldIdTy = Int32Ty;
  IndexLauncherHandleTy = VoidPtrTy;
  TaskArgumentTy = VoidPtrTy;
  ArgumentMapHandleTy = VoidPtrTy;
  ProjectionIdTy = Int32Ty;
  RegionRequirementHndlTy = VoidPtrTy;
  UnimeshHandleTy = VoidPtrTy;
  VariantIdTy = Int64Ty;
  
  SuccessVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0));
  FailureVal = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
  
  NoAccessVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000000));
  ReadOnlyVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000001));
  ReadWriteVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000111));
  WriteOnlyVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000010));
  WriteDiscardVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000010));
  ReduceVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00000100));
  PromotedVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0x00001000));
  
  ExclusiveVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0));
  AtomicVal = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
  SimultaenousVal = llvm::ConstantInt::get(context, llvm::APInt(32, 2));
  RelaxedVal = llvm::ConstantInt::get(context, llvm::APInt(32, 3));
  
  TypeInt32Val = llvm::ConstantInt::get(context, llvm::APInt(32, 0));
  TypeInt64Val = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
  TypeFloatVal = llvm::ConstantInt::get(context, llvm::APInt(32, 2));
  TypeDoubleVal = llvm::ConstantInt::get(context, llvm::APInt(32, 3));
  
  TocProcVal = llvm::ConstantInt::get(context, llvm::APInt(32, 0));
  LocProcVal = llvm::ConstantInt::get(context, llvm::APInt(32, 1));
  UtilProcVal = llvm::ConstantInt::get(context, llvm::APInt(32, 2));
  
  vector<llvm::Type*> fields = {PointerTy(Int8Ty)};
  Rect1dStorageTy = llvm::StructType::get(context, fields);

  fields = {DomainHandleTy, Int64Ty};
  DomainTy = llvm::StructType::get(context, fields);
  
  fields = {Int64Ty, FieldIdTy, IndexSpaceTy, LogicalRegionTy,
    LogicalPartitionTy, DomainTy, Int64Ty, Rect1dTy};
  VectorTy = llvm::StructType::get(context, fields);
  
  fields = {ArgumentMapTy};
  ArgumentMapTy = llvm::StructType::get(context, fields);
  
  fields = {IndexLauncherHandleTy, Int32Ty, DomainTy};
  IndexLauncherTy = llvm::StructType::get(context, fields);
  
  fields = {RegionRequirementHndlTy, LogicalRegionTy,
    ProjectionIdTy, Int32Ty, Int32Ty, LogicalPartitionTy};
  RegionRequirementTy = llvm::StructType::get(context, fields);

  fields = {UnimeshHandleTy, Int64Ty, Int64Ty, Int64Ty, Int64Ty};
  UnimeshTy = llvm::StructType::get(context, fields);
  
  fields = {ContextTy, RuntimeTy, Int32Ty, Int64Ty, PhysicalRegionsTy, VoidPtrTy};
  TaskArgsTy = llvm::StructType::get(context, fields);

  fields = {VoidPtrTy};
  RegTaskDataTy = llvm::StructType::get(context, fields);
}

CGLegionRuntime::~CGLegionRuntime() {}

llvm::Function *CGLegionRuntime::CreateSetupMeshFunction(llvm::Type *MT) {
  std::string funcName = "__scrt_legion_setup_mesh";
  std::vector<llvm::Type*> Params;
  // pointer to mesh
  Params.push_back(llvm::PointerType::get(MT,0));

  // width, height, depth
  Params.push_back(llvm::IntegerType::get(CGM.getLLVMContext(), 64));
  Params.push_back(llvm::IntegerType::get(CGM.getLLVMContext(), 64));
  Params.push_back(llvm::IntegerType::get(CGM.getLLVMContext(), 64));

  return LegionRuntimeFunction(funcName, Params);
}

llvm::Function *CGLegionRuntime::CreateAddFieldFunction(llvm::Type *MT) {
  std::string funcName = "__scrt_legion_add_field";
  std::vector<llvm::Type*> Params;
  // pointer to mesh
  Params.push_back(llvm::PointerType::get(MT,0));

  // name
  Params.push_back(llvm::PointerType::get(llvm::IntegerType::get(CGM.getLLVMContext(), 8), 0));

  // type
  Params.push_back(llvm::IntegerType::get(CGM.getModule().getContext(), 32));

  return LegionRuntimeFunction(funcName, Params);
}

// build a function call to a legion runtime function w/ no arguments
llvm::Function *CGLegionRuntime::LegionRuntimeFunction(std::string funcName, std::vector<llvm::Type*> Params ) {

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

llvm::Function *CGLegionRuntime::LegionRuntimeFunction(string funcName,
                                                       vector<llvm::Type*> Params, llvm::Type* retType) {
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

llvm::Type* CGLegionRuntime::PointerTy(llvm::Type* elementType){
  return llvm::PointerType::get(elementType, 0);
}

llvm::Function* CGLegionRuntime::SizeofCXXRect1dFunc(){
  string name = "lsci_sizeof_cxx_rect_1d";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params;
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int64Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::SubgridBoundsAtFunc(){
  string name = "lsci_subgrid_bounds_at";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {VoidPtrTy, Int64Ty};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(VoidPtrTy, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::VectorDumpFunc(){
  string name = "lsci_vector_dump";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {VoidPtrTy, Int64Ty};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::ArgumentMapCreateFunc(){
  string name = "lsci_argument_map_create";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(ArgumentMapTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::ArgumentMapSetPointFunc(){
  string name = "lsci_argument_map_set_point";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(ArgumentMapTy), Int64Ty, VoidPtrTy, Int64Ty};

  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::IndexLauncherCreateFunc(){
  string name = "lsci_index_launcher_create";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(IndexLauncherTy), Int32Ty, PointerTy(DomainTy),
    PointerTy(TaskArgumentTy), PointerTy(ArgumentMapTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::AddRegionRequirementFunc(){
  string name = "lsci_add_region_requirement";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(IndexLauncherTy), LogicalRegionTy, ProjectionIdTy, Int32Ty,
    Int32Ty, LogicalPartitionTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::AddFieldFunc(){
  string name = "lsci_add_field";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(IndexLauncherTy), Int32Ty, FieldIdTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::ExecuteIndexSpaceFunc(){
  string name = "lsci_execute_index_space";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {RuntimeTy, ContextTy, PointerTy(IndexLauncherTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::VectorCreateFunc(){
  string name = "lsci_vector_create";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(VectorTy), Int64Ty, Int32Ty, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::UnimeshCreateFunc(){
  string name = "lsci_unimesh_create";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(UnimeshTy), Int64Ty, Int64Ty, Int64Ty, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::UnimeshAddFieldFunc(){
  string name = "lsci_unimesh_add_field";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(UnimeshTy), Int32Ty, VoidPtrTy, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::UnimeshPartitionFunc(){
  string name = "lsci_unimesh_partition";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(UnimeshTy), Int64Ty, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::UnimeshGetVecByNameFunc(){
  string name = "lsci_unimesh_get_vec_by_name";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(UnimeshTy), VoidPtrTy, PointerTy(VectorTy), ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::StartFunc(){
  string name = "lsci_start";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {Int32Ty, PointerTy(VoidPtrTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::SetTopLevelTaskIdFunc(){
  string name = "lsci_set_top_level_task_id";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {Int32Ty, PointerTy(VoidPtrTy)};
  
  llvm::FunctionType* ft =
  llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()), params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::RegisterVoidLegionTaskFunc(){
  string name = "lsci_register_void_legion_task";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {Int32Ty, Int32Ty, Int8Ty, Int8Ty, Int8Ty, VariantIdTy, VoidPtrTy, RegTaskDataTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}
