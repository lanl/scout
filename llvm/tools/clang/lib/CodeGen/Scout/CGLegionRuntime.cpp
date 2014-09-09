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
 
  Int1Ty = llvm::Type::getInt1Ty(context); 
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
  StructHandleTy = VoidPtrTy;
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
 
  // need to change all the struct types to create identified struct types if not already created
  // can't use literal struct types
  vector<llvm::Type*> fields;
 
  Rect1dStorageTy = CGM.getModule().getTypeByName("struct.lsci_rect_1d_storage_t");
  if (!Rect1dStorageTy) {
    fields = {PointerTy(Int8Ty)};
    Rect1dStorageTy = llvm::StructType::create(context, fields, "struct.lsci_rect_1d_storage_t");
  }

  DomainTy = CGM.getModule().getTypeByName("struct.lsci_domain_t");
  if (!DomainTy) {
    fields = {DomainHandleTy, Int64Ty};
    DomainTy = llvm::StructType::create(context, fields, "struct.lsci_domain_t");
  }
 
  VectorTy = CGM.getModule().getTypeByName("struct.lsci_vector_t");
  if (!VectorTy) {
    fields = {Int64Ty, FieldIdTy, IndexSpaceTy, LogicalRegionTy,
      LogicalPartitionTy, DomainTy, Int64Ty, Rect1dTy};
    VectorTy = llvm::StructType::create(context, fields, "struct.lsci_vector_t");
  }
  
  ArgumentMapTy = CGM.getModule().getTypeByName("struct.lsci_argument_map_t");
  if (!ArgumentMapTy) {
    fields = {ArgumentMapHandleTy};
    ArgumentMapTy = llvm::StructType::create(context, fields, "struct.lsci_argument_map_t");
  }
 
  IndexLauncherTy =  CGM.getModule().getTypeByName("struct.lsci_index_launcher_t");
  if (!IndexLauncherTy) {
    fields = {IndexLauncherHandleTy, Int32Ty, DomainTy};
    IndexLauncherTy = llvm::StructType::create(context, fields, "struct.lsci_index_launcher_t");
  }
 
  RegionRequirementTy =  CGM.getModule().getTypeByName("struct.lsci_region_requirement_t");
  if (!RegionRequirementTy) {
    fields = {RegionRequirementHndlTy, LogicalRegionTy,
      ProjectionIdTy, Int32Ty, Int32Ty, LogicalPartitionTy};
    RegionRequirementTy = llvm::StructType::create(context, fields, "struct.lsci_region_requirement_t");
  }

  UnimeshTy =  CGM.getModule().getTypeByName("struct.lsci_unimesh_t");
  if (!UnimeshTy) {
    fields = {UnimeshHandleTy, Int64Ty, Int64Ty, Int64Ty, Int64Ty};
    UnimeshTy = llvm::StructType::create(context, fields, "struct.lsci_unimesh_t");
  }

  StructTy =  CGM.getModule().getTypeByName("struct.lsci_struct_t");
  if (!StructTy) {
    fields = {StructHandleTy};
    StructTy = llvm::StructType::create(context, fields, "struct.lsci_struct_t");
  }
  
  TaskArgsTy =  CGM.getModule().getTypeByName("struct.lsci_task_args_t");
  if (!TaskArgsTy) {
    fields = {ContextTy, RuntimeTy, Int32Ty, Int64Ty, PhysicalRegionsTy, VoidPtrTy};
    TaskArgsTy = llvm::StructType::create(context, fields, "struct.lsci_task_args_t");
  }

  RegTaskDataTy = CGM.getModule().getTypeByName("struct.lsci_reg_task_data_t");
  if (!RegTaskDataTy) {
    // lsci_reg_task_data_t contains a pointer to a function that takes a pointer 
    // to lsci_task_args_t and returns void
    vector<llvm::Type*> args = {PointerTy(TaskArgsTy)};
    llvm::FunctionType* funcType = llvm::FunctionType::get(CGM.VoidTy, args, false);
    fields = {funcType};
    RegTaskDataTy = llvm::StructType::create(context, fields, "struct.lsci_reg_task_data_t");
  }
  
  MeshTaskArgsTy = CGM.getModule().getTypeByName("struct.lsci_mesh_task_args_t");
  if (!MeshTaskArgsTy) {
    fields = {Int64Ty, Int64Ty, Int64Ty, Int64Ty, Rect1dStorageTy, Int64Ty};
    MeshTaskArgsTy = llvm::StructType::create(context, fields, "struct.lsci_mesh_task_args_t");
  }
}

CGLegionRuntime::~CGLegionRuntime() {}

llvm::Value* CGLegionRuntime::GetNull(llvm::Type* T){
  return llvm::ConstantPointerNull::get(PointerTy(T));
}

llvm::Value *CGLegionRuntime::GetLegionRuntimeGlobal() {
  return GetLegionGlobal("__scrt_legion_runtime", VoidPtrTy);
}

llvm::Value *CGLegionRuntime::GetLegionContextGlobal() {
  return GetLegionGlobal("__scrt_legion_context", VoidPtrTy);
}

llvm::Function *CGLegionRuntime::CreateSetupMeshFunction(llvm::Type *MT) {
  // std::string funcName = "__scrt_legion_setup_mesh";
  // SC_TODO: once we get the context/runtime setup via a pass then we can
  // switch to using the real lsci function here
  std::string funcName = "lsci_unimesh_create";
  std::vector<llvm::Type*> Params = {PointerTy(UnimeshTy), Int64Ty, Int64Ty, Int64Ty, ContextTy, RuntimeTy};
  return LegionRuntimeFunction(funcName, Params, Int32Ty);
}

llvm::Function *CGLegionRuntime::CreateAddFieldFunction(llvm::Type *MT) {
  // std::string funcName = "__scrt_legion_add_field";
  // SC_TODO: once we get the context/runtime setup via a pass then we can
  // switch to using the real lsci fuction here
  std::string funcName = "lsci_unimesh_add_field";
  std::vector<llvm::Type*> Params = {PointerTy(UnimeshTy), Int32Ty, PointerTy(Int8Ty), ContextTy, RuntimeTy};
  return LegionRuntimeFunction(funcName, Params, Int32Ty);
}


llvm::Value *CGLegionRuntime::GetLegionGlobal(std::string varName, llvm::Type *type) {
  if (!CGM.getModule().getNamedGlobal(varName)) {
    new llvm::GlobalVariable(CGM.getModule(),
        type,
        false,
        llvm::GlobalValue::ExternalLinkage,
        0,
        varName);
  }
  return CGM.getModule().getNamedGlobal(varName);
}

// build a function call to a legion runtime function w/ void return
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

// build a function call to a legion runtime function w/ return value
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


llvm::PointerType* CGLegionRuntime::PointerTy(llvm::Type* elementType){
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

llvm::Function* CGLegionRuntime::StructCreateFunc(){
  string name = "lsci_struct_create";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(StructTy), ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::StructAddFieldFunc(){
  string name = "lsci_struct_add_field";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(StructTy), Int32Ty, Int64Ty, VoidPtrTy, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::StructPartitionFunc(){
  string name = "lsci_struct_partition";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params = {PointerTy(StructTy), Int64Ty, ContextTy, RuntimeTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::StructGetVecByNameFunc(){
  string name = "lsci_struct_get_vec_by_name";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(StructTy), VoidPtrTy, PointerTy(VectorTy), ContextTy, RuntimeTy};
  
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
  
  vector<llvm::Type*> params = {Int32Ty};
  
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
  {Int32Ty, Int32Ty, Int1Ty, Int1Ty, Int1Ty, VariantIdTy, VoidPtrTy, RegTaskDataTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::RegisterVoidLegionTaskAuxFunc(){
  string name = "lsci_register_void_legion_task_aux";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }

  vector<llvm::Type*> args = {PointerTy(TaskArgsTy)};
  llvm::FunctionType* funcType = llvm::FunctionType::get(CGM.VoidTy, args, false);
  
  vector<llvm::Type*> params =
  {Int32Ty, Int32Ty, Int1Ty, Int1Ty, Int1Ty, VariantIdTy, VoidPtrTy, llvm::PointerType::get(funcType, 0)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(Int32Ty, params, false);

  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}
  

llvm::Function* CGLegionRuntime::RawRectPtr1dFunc(){
  string name = "raw_rect_ptr_1d";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PhysicalRegionsTy, Int32Ty, Int64Ty, FieldIdTy, Rect1dTy};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(VoidPtrTy, params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::PrintMeshTaskArgsFunc() {
  string name = "lsci_print_mesh_task_args";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(MeshTaskArgsTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()), params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}

llvm::Function* CGLegionRuntime::PrintTaskArgsLocalArgspFunc() {
  string name = "lsci_print_task_args_local_argsp";
  
  llvm::Function* f = CGM.getModule().getFunction(name);
  
  if(f){
    return f;
  }
  
  vector<llvm::Type*> params =
  {PointerTy(TaskArgsTy)};
  
  llvm::FunctionType* ft = llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()), params, false);
  
  f = llvm::Function::Create(ft,
                             llvm::Function::ExternalLinkage,
                             name,
                             &CGM.getModule());
  
  return f;
}
