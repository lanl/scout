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

#include "Scout/CGLegionCRuntime.h"
#include "CodeGenFunction.h"
//#include "legion/runtime/legion_c.h"

using namespace std;
using namespace clang;
using namespace CodeGen;
using namespace llvm;

CGCLegionCRuntime::CGCLegionCRuntime(CodeGenModule& CGM) : CGM(CGM){
  llvm::LLVMContext& C = CGM.getLLVMContext();
 
  Int1Ty = llvm::Type::getInt1Ty(C);
  Int8Ty = llvm::Type::getInt8Ty(C);
  Int32Ty = llvm::Type::getInt32Ty(C);
  Int64Ty = llvm::Type::getInt64Ty(C);
  VoidTy = llvm::Type::getVoidTy(C);
  VoidPtrTy = PointerTy(Int8Ty);
  
  TypeVec fields;
  
  fields = {VoidPtrTy};
  OpaqueTy = StructType::create(C, fields, "struct.legion_oqaque_t");
  
  RuntimeTy = OpaqueTy;
  ContextTy = OpaqueTy;
  ColoringTy = OpaqueTy;
  DomainColoringTy = OpaqueTy;
  IndexSpaceAllocatorTy = OpaqueTy;
  ArgumentMapTy = OpaqueTy;
  PredicateTy = OpaqueTy;
  FutureTy = OpaqueTy;
  FutureMapTy = OpaqueTy;
  TaskLauncherTy = OpaqueTy;
  IndexLauncherTy = OpaqueTy;
  InlineLauncherTy = OpaqueTy;
  PhysicalRegionTy = OpaqueTy;
  AccessorGenericTy = OpaqueTy;
  AccessorArrayTy = OpaqueTy;
  IndexIteratorTy = OpaqueTy;
  TaskTy = OpaqueTy;
  
  LowlevelIdTy = Int64Ty;
  LowlevelAddressSpaceTy = Int32Ty;
  LowlevelTaskFuncIdTy = Int32Ty;
  LowlevelReductionOpIdTy = Int32Ty;
  
  fields = {Int32Ty};
  PtrTy = StructType::create(C, fields, "struct.legion_ptr_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 1)};
  Point1dTy = StructType::create(C, fields, "struct.legion_point1d_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 2)};
  Point2dTy = StructType::create(C, fields, "struct.legion_point2d_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 3)};
  Point3dTy = StructType::create(C, fields, "struct.legion_point3d_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 1)};
  Rect1dTy = StructType::create(C, fields, "struct.legion_rect1d_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 2)};
  Rect2dTy = StructType::create(C, fields, "struct.legion_rect2d_t");
  
  fields = {llvm::ArrayType::get(Int32Ty, 3)};
  Rect3dTy = StructType::create(C, fields, "struct.legion_rect3d_t");
}

CGCLegionCRuntime::~CGCLegionCRuntime(){}

Value* CGCLegionCRuntime::GetNull(llvm::Type* T){
  return ConstantPointerNull::get(PointerTy(T));
}

llvm::PointerType* CGCLegionCRuntime::PointerTy(llvm::Type* elementType){
  return llvm::PointerType::get(elementType, 0);
}

llvm::Function*
CGCLegionCRuntime::GetFunc(const std::string& funcName,
                           TypeVec& argTypes,
                           llvm::Type* retType){

  llvm::LLVMContext& C = CGM.getLLVMContext();
  
  llvm::Function* func = CGM.getModule().getFunction(funcName);
  
  if(!func){
    llvm::FunctionType* funcType =
    llvm::FunctionType::get(retType == 0 ?
                            llvm::Type::getVoidTy(C) : retType,
                            argTypes, false);
    
    func =
    llvm::Function::Create(funcType,
                           llvm::Function::ExternalLinkage,
                           funcName,
                           &CGM.getModule());
  }
  
  return func;
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromRect1dFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromRect2dFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromRect3dFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromIndexSpaceFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringEnsureColorFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringAddPointFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringAddRangeFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceCreateDomainFunc(){
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionCreateColoringFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionCreateDomainColoringFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldSpaceCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldSpaceDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalRegionCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalRegionDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionGetLogicalSubregionFunc(){
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionGetLogicalSubregionByColorFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorAllocFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorFreeFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorAllocateFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorFreeFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorAllocateLocalFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::PredicateDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::PredicateTrueFunc(){
}

llvm::Function*
CGCLegionCRuntime::PredicateFalseFunc(){
}

llvm::Function*
CGCLegionCRuntime::FutureDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::FutureGetVoidResultFunc(){
}

llvm::Function*
CGCLegionCRuntime::FutureGetResultFunc(){
}

llvm::Function*
CGCLegionCRuntime::FutureIsEmptyFunc(){
}

llvm::Function*
CGCLegionCRuntime::FutureGetUntypedPointerFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskResultCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskResultDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherExecuteFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherAddRegionRequirementLogicalRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherAddFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherExecuteFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherExecuteReductionFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddRegionRequirementLogicalRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddRegionRequirementLogicalPartitionFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherCreateLogicalRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherExecuteFunc(){
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherAddFieldFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeRemapRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeUnmapRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeMapAllRegionFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeUnmapAllRegionsFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionIsMappedFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionWaitUntilValidFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionIsValidFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionGetFieldAccessorGenericFunc(){
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionGetFieldAccessorArrayFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericReadFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericWriteFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr1dFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr2dFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr3dFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayReadFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayWriteFunc(){
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayRefFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorCreateFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorDestroyFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorHasNextFunc(){
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorNextFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskGetArgsFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskGetArglenFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskGetIsIndexSpaceFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskGetLocalArgsFunc(){
}

llvm::Function*
CGCLegionCRuntime::TaskGetLocalArglenFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeStartFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeWaitForShutdownFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeSetTopLevelTaskIdFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeGetInputArgsFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeRegisterTaskVoidFunc(){
}

llvm::Function*
CGCLegionCRuntime::RuntimeRegisterTaskFunc(){
}
