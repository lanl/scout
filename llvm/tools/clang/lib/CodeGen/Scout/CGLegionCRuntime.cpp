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
#include "legion/legion_c.h"

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
  OpaqueTy = StructType::create(C, fields, "struct.legion_opaque_t");
  
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
  DomainMaxRectDimTy = Int32Ty;
  ReductionOpIdTy = Int32Ty;
  AddressSpaceTy = Int32Ty;
  TaskPriorityTy = Int32Ty;
  ColorTy = Int32Ty;
  IndexPartitionTy = Int32Ty;
  FieldIdTy = Int32Ty;
  TraceIdTy = Int32Ty;
  MapperIdTy = Int32Ty;
  ContextIdTy = Int32Ty;
  InstanceIdTy = Int32Ty;
  FieldSpaceIdTy = Int32Ty;
  GenerationIdTy = Int32Ty;
  TypeHandleTy = Int32Ty;
  ProjectionIdTy = Int32Ty;
  RegionTreeIdTy = Int32Ty;
  DistributedIdTy = Int32Ty;
  AddressSpaceIdTy = Int32Ty;
  TunableIdTy = Int32Ty;
  MappingTagIdTy = Int64Ty;
  VariantIdTy = Int64Ty;
  UniqueIdTy = Int64Ty;
  VersionIdTy = Int64Ty;
  TaskIdTy = Int32Ty;
  
  TocProcVal = ConstantInt::get(C, APInt(32, TOC_PROC));
  LocProcVal = ConstantInt::get(C, APInt(32, LOC_PROC));
  UtilProcVal = ConstantInt::get(C, APInt(32, UTIL_PROC));
  ProcGroupVal = ConstantInt::get(C, APInt(32, PROC_GROUP));
  MaxRectDimVal = ConstantInt::get(C, APInt(32, MAX_RECT_DIM));
  
  PrivilegeModeTy = Int32Ty;
  NoAccessVal = ConstantInt::get(C, APInt(32, NO_ACCESS));
  ReadOnlyVal = ConstantInt::get(C, APInt(32, READ_ONLY));
  ReadWriteVal = ConstantInt::get(C, APInt(32, READ_WRITE));
  WriteOnlyVal = ConstantInt::get(C, APInt(32, WRITE_ONLY));
  WriteDiscardVal = ConstantInt::get(C, APInt(32, WRITE_DISCARD));
  ReduceVal = ConstantInt::get(C, APInt(32, REDUCE));
  PromotedVal = ConstantInt::get(C, APInt(32, PROMOTED));
  
  AllocateModeTy = Int32Ty;
  NoMemoryVal = ConstantInt::get(C, APInt(32, NO_MEMORY));
  AllocableVal = ConstantInt::get(C, APInt(32, ALLOCABLE));
  FreeableVal = ConstantInt::get(C, APInt(32, FREEABLE));
  MutableVal = ConstantInt::get(C, APInt(32, MUTABLE));
  RegionCreationVal = ConstantInt::get(C, APInt(32, REGION_CREATION));
  RegionDeletionVal = ConstantInt::get(C, APInt(32, REGION_DELETION));
  AllMemoryVal = ConstantInt::get(C, APInt(32, ALL_MEMORY));
  
  CoherencePropertyTy = Int32Ty;
  ExclusiveVal = ConstantInt::get(C, APInt(32, EXCLUSIVE));
  AtomicVal = ConstantInt::get(C, APInt(32, ATOMIC));
  SimultaneousVal = ConstantInt::get(C, APInt(32, SIMULTANEOUS));
  RelaxedVal = ConstantInt::get(C, APInt(32, RELAXED));
  
  RegionFlagsTy = Int32Ty;
  NoFlagVal = ConstantInt::get(C, APInt(32, NO_FLAG));
  VerifiedFlagVal = ConstantInt::get(C, APInt(32, VERIFIED_FLAG));
  NoAccessFlagVal = ConstantInt::get(C, APInt(32, NO_ACCESS_FLAG));
  
  HandleTy = Int32Ty;
  SingularVal = ConstantInt::get(C, APInt(32, SINGULAR));
  PartProjectionVal = ConstantInt::get(C, APInt(32, PART_PROJECTION));
  RegProjectionVal = ConstantInt::get(C, APInt(32, REG_PROJECTION));
  
  DependenceTy = Int32Ty;
  NoDependenceVal = ConstantInt::get(C, APInt(32, NO_DEPENDENCE));
  TrueDependenceVal = ConstantInt::get(C, APInt(32, TRUE_DEPENDENCE));
  AntiDependenceVal = ConstantInt::get(C, APInt(32, ANTI_DEPENDENCE));
  AtomicDependenceVal = ConstantInt::get(C, APInt(32, ATOMIC_DEPENDENCE));
  SimultaneousDependenceVal =
  ConstantInt::get(C, APInt(32, SIMULTANEOUS_DEPENDENCE));
  PromotedDependenceVal = ConstantInt::get(C, APInt(32, PROMOTED_DEPENDENCE));
  
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
  
  TypeVec params;

  params = {TaskTy, PhysicalRegionTy, Int32Ty, ContextTy, RuntimeTy};
  VoidTaskFuncTy = llvm::FunctionType::get(llvm::Type::getVoidTy(C),
                                           params, false);

  params = {TaskTy, PhysicalRegionTy, Int32Ty, ContextTy, RuntimeTy};
  TaskFuncTy = llvm::FunctionType::get(TaskResultTy,
                                       params, false);
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
                           const TypeVec& argTypes,
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
  return GetFunc("legion_domain_from_rect_1d",
                 {Rect1dTy},
                 DomainTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromRect2dFunc(){
  return GetFunc("legion_domain_from_rect_2d",
                 {Rect2dTy},
                 DomainTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromRect3dFunc(){
  return GetFunc("legion_domain_from_rect_3d",
                 {Rect3dTy},
                 DomainTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::DomainFromIndexSpaceFunc(){
  return GetFunc("legion_domain_from_index_space",
                 {IndexSpaceTy},
                 DomainTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringCreateFunc(){
  return GetFunc("legion_coloring_create",
                 TypeVec(),
                 ColoringTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringDestroyFunc(){
  return GetFunc("legion_coloring_destroy",
                 {ColoringTy});
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringEnsureColorFunc(){
  return GetFunc("legion_coloring_ensure_color",
                 {ColoringTy, ColorTy});
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringAddPointFunc(){
  return GetFunc("legion_coloring_add_point",
                 {ColoringTy, ColorTy, PtrTy});
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::ColoringAddRangeFunc(){
  return GetFunc("legion_coloring_add_range",
                 {ColoringTy, ColorTy, PtrTy, PtrTy});
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceCreateFunc(){
  return GetFunc("legion_index_space_create",
                 {RuntimeTy, ContextTy, Int64Ty},
                 IndexSpaceTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceCreateDomainFunc(){
  return GetFunc("legion_index_space_create_domain",
                 {RuntimeTy, ContextTy, DomainTy},
                 IndexSpaceTy);
}

llvm::Function*
CGCLegionCRuntime::CGCLegionCRuntime::IndexSpaceDestroyFunc(){
  return GetFunc("legion_index_space_destroy",
                 {RuntimeTy, ContextTy, IndexSpaceTy});
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionCreateColoringFunc(){
  return GetFunc("legion_index_partition_create_coloring",
                 {RuntimeTy, ContextTy, IndexSpaceTy,
                   ColoringTy, Int1Ty, Int32Ty},
                 IndexPartitionTy);
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionCreateDomainColoringFunc(){
  return GetFunc("legion_index_partition_create_domain_coloring",
                 {RuntimeTy, ContextTy, IndexSpaceTy,
                   DomainTy, ColoringTy, Int1Ty, Int32Ty},
                 IndexPartitionTy);
}

llvm::Function*
CGCLegionCRuntime::IndexPartitionDestroyFunc(){
  return GetFunc("legion_index_partition_destroy",
                 {RuntimeTy, ContextTy, IndexPartitionTy});
}

llvm::Function*
CGCLegionCRuntime::FieldSpaceCreateFunc(){
  return GetFunc("legion_field_space_create",
                 {RuntimeTy, ContextTy},
                 FieldSpaceTy);
}

llvm::Function*
CGCLegionCRuntime::FieldSpaceDestroyFunc(){
  return GetFunc("legion_field_space_destroy",
                 {RuntimeTy, ContextTy, FieldSpaceTy});
}

llvm::Function*
CGCLegionCRuntime::LogicalRegionCreateFunc(){
  return GetFunc("legion_logical_region_create",
                 {RuntimeTy, ContextTy, IndexSpaceTy, FieldSpaceTy},
                 LogicalRegionTy);
}

llvm::Function*
CGCLegionCRuntime::LogicalRegionDestroyFunc(){
  return GetFunc("legion_logical_region_destroy",
                 {RuntimeTy, ContextTy, LogicalRegionTy});
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionCreateFunc(){
  return GetFunc("legion_logical_partition_create",
                 {RuntimeTy, ContextTy, LogicalPartitionTy},
                 LogicalPartitionTy);
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionDestroyFunc(){
  return GetFunc("legion_logical_partition_destroy",
                 {RuntimeTy, ContextTy, LogicalPartitionTy});
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionGetLogicalSubregionFunc(){
  return GetFunc("legion_logical_partition_get_logical_subregion",
                 {RuntimeTy, ContextTy, LogicalPartitionTy, IndexSpaceTy},
                 LogicalRegionTy);
}

llvm::Function*
CGCLegionCRuntime::LogicalPartitionGetLogicalSubregionByColorFunc(){
  return GetFunc("legion_logical_partition_get_logical_subregion_by_color",
                 {RuntimeTy, ContextTy, LogicalPartitionTy, ColorTy},
                 LogicalRegionTy);
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorCreateFunc(){
  return GetFunc("legion_index_allocator_create",
                 {RuntimeTy, ContextTy, IndexSpaceTy},
                 IndexAllocatorTy);
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorDestroyFunc(){
  return GetFunc("legion_index_allocator_destroy",
                 {IndexAllocatorTy});
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorAllocFunc(){
  return GetFunc("legion_index_allocator_alloc",
                 {IndexAllocatorTy, PtrTy},
                 PtrTy);
}

llvm::Function*
CGCLegionCRuntime::IndexAllocatorFreeFunc(){
  return GetFunc("legion_index_allocator_free",
                 {IndexAllocatorTy, PtrTy, Int32Ty});
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorCreateFunc(){
  return GetFunc("legion_field_allocator_create",
                 {RuntimeTy, ContextTy, FieldSpaceTy},
                 IndexAllocatorTy);
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorDestroyFunc(){
  return GetFunc("legion_field_allocator_destroy",
                 {FieldAllocatorTy});
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorAllocateFieldFunc(){
  return GetFunc("legion_field_allocator_allocate_field",
                 {FieldAllocatorTy, Int64Ty, FieldIdTy},
                 FieldIdTy);
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorFreeFieldFunc(){
  return GetFunc("legion_field_allocator_free_field",
                 {RuntimeTy, ContextTy},
                 DomainTy);
}

llvm::Function*
CGCLegionCRuntime::FieldAllocatorAllocateLocalFieldFunc(){
  return GetFunc("legion_field_allocator_allocate_local_field",
                 {FieldAllocatorTy, Int64Ty, FieldIdTy},
                 FieldIdTy);
}

llvm::Function*
CGCLegionCRuntime::PredicateDestroyFunc(){
  return GetFunc("legion_predicate_destroy",
                 {PredicateTy});
}

llvm::Function*
CGCLegionCRuntime::PredicateTrueFunc(){
  return GetFunc("legion_predicate_true",
                 TypeVec(),
                 PredicateTy);
}

llvm::Function*
CGCLegionCRuntime::PredicateFalseFunc(){
  return GetFunc("legion_predicate_false",
                 TypeVec(),
                 PredicateTy);
}

llvm::Function*
CGCLegionCRuntime::FutureDestroyFunc(){
  return GetFunc("legion_future_destroy",
                 {FutureTy});
}

llvm::Function*
CGCLegionCRuntime::FutureGetVoidResultFunc(){
  return GetFunc("legion_future_get_void_result",
                 {FutureTy});
}

llvm::Function*
CGCLegionCRuntime::FutureGetResultFunc(){
  return GetFunc("legion_future_get_result",
                 {FutureTy},
                 TaskResultTy);
}

llvm::Function*
CGCLegionCRuntime::FutureIsEmptyFunc(){
  return GetFunc("legion_future_is_empty",
                 {FutureTy, Int1Ty},
                 Int1Ty);
}

llvm::Function*
CGCLegionCRuntime::FutureGetUntypedPointerFunc(){
  return GetFunc("legion_future_get_untyped_pointer",
                 {FutureTy},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::TaskResultCreateFunc(){
  return GetFunc("legion_task_result_create",
                 {VoidPtrTy, Int64Ty},
                 TaskResultTy);
}

llvm::Function*
CGCLegionCRuntime::TaskResultDestroyFunc(){
  return GetFunc("legion_task_result_destroy",
                 {TaskResultTy});
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherCreateFunc(){
  return GetFunc("legion_task_launcher_create",
                 {TaskIdTy, TaskArgumentTy, PredicateTy,
                   MapperIdTy, MappingTagIdTy},
                 TaskLauncherTy);
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherDestroyFunc(){
  return GetFunc("legion_task_launcher_destroy",
                 {TaskLauncherTy});
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherExecuteFunc(){
  return GetFunc("legion_task_launcher_execute",
                 {RuntimeTy, ContextTy, TaskLauncherTy},
                 FutureTy);
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherAddRegionRequirementLogicalRegionFunc(){
  return GetFunc("legion_task_launcher_add_region_requirement_logical_region",
                 {TaskLauncherTy, LogicalRegionTy, PrivilegeModeTy,
                 CoherencePropertyTy, LogicalRegionTy, MappingTagIdTy, Int1Ty},
                 Int32Ty);
}

llvm::Function*
CGCLegionCRuntime::TaskLauncherAddFieldFunc(){
  return GetFunc("legion_task_launcher_add_field",
                 {TaskLauncherTy, Int32Ty, FieldIdTy, Int1Ty});
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherCreateFunc(){
  return GetFunc("legion_index_launcher_create",
                 {TaskIdTy, DomainTy, TaskArgumentTy, ArgumentMapTy,
                 PredicateTy, Int1Ty, MapperIdTy, MappingTagIdTy},
                 IndexLauncherTy);
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherDestroyFunc(){
  return GetFunc("legion_index_launcher_destroy",
                 {IndexLauncherTy});
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherExecuteFunc(){
  return GetFunc("legion_index_launcher_execute",
                 {RuntimeTy, ContextTy, IndexLauncherTy},
                 FutureMapTy);
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherExecuteReductionFunc(){
  return GetFunc("legion_index_launcher_execute_reduction",
                 {RuntimeTy, ContextTy, IndexLauncherTy, ReductionOpIdTy},
                 FutureTy);
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddRegionRequirementLogicalRegionFunc(){
  return GetFunc("legion_index_launcher_add_region_requirement_logical_region",
                 {IndexLauncherTy, LogicalRegionTy, ProjectionIdTy,
                 PrivilegeModeTy, CoherencePropertyTy, LogicalRegionTy,
                 MappingTagIdTy, Int1Ty},
                 Int32Ty);
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddRegionRequirementLogicalPartitionFunc(){
  return GetFunc("legion_index_launcher_add_region_requirement_logical_partition",
                 {IndexLauncherTy, LogicalPartitionTy, ProjectionIdTy,
                 ProjectionIdTy, PrivilegeModeTy, CoherencePropertyTy,
                 LogicalRegionTy, MappingTagIdTy, Int1Ty},
                 Int32Ty);
}

llvm::Function*
CGCLegionCRuntime::IndexLauncherAddFieldFunc(){
  return GetFunc("legion_index_launcher_add_field",
                 {IndexLauncherTy, Int32Ty, FieldIdTy, Int1Ty});
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherCreateLogicalRegionFunc(){
  return GetFunc("legion_inline_launcher_create_logical_region",
                 {LogicalRegionTy, PrivilegeModeTy, CoherencePropertyTy,
                   LogicalRegionTy, MappingTagIdTy, Int1Ty,
                   MapperIdTy, MappingTagIdTy},
                 InlineLauncherTy);
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherDestroyFunc(){
  return GetFunc("legion_inline_launcher_destroy",
                 {InlineLauncherTy});
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherExecuteFunc(){
  return GetFunc("legion_inline_launcher_execute",
                 {RuntimeTy, ContextTy, InlineLauncherTy},
                 PhysicalRegionTy);
}

llvm::Function*
CGCLegionCRuntime::InlineLauncherAddFieldFunc(){
  return GetFunc("legion_inline_launcher_add_field",
                 {InlineLauncherTy, FieldIdTy, Int1Ty});
}

llvm::Function*
CGCLegionCRuntime::RuntimeRemapRegionFunc(){
  return GetFunc("legion_runtime_remap_region",
                 {RuntimeTy, ContextTy, PhysicalRegionTy});
}

llvm::Function*
CGCLegionCRuntime::RuntimeUnmapRegionFunc(){
  return GetFunc("legion_runtime_unmap_region",
                 {RuntimeTy, ContextTy, PhysicalRegionTy});
}

llvm::Function*
CGCLegionCRuntime::RuntimeMapAllRegionFunc(){
  return GetFunc("legion_runtime_map_all_region",
                 {RuntimeTy, ContextTy});
}

llvm::Function*
CGCLegionCRuntime::RuntimeUnmapAllRegionsFunc(){
  return GetFunc("legion_runtime_unmap_all_regions",
                 {RuntimeTy, ContextTy});
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionDestroyFunc(){
  return GetFunc("legion_physical_region_destroy",
                 {PhysicalRegionTy});
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionIsMappedFunc(){
  return GetFunc("legion_physical_region_is_mapped",
                 {PhysicalRegionTy},
                 Int1Ty);
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionWaitUntilValidFunc(){
  return GetFunc("legion_physical_region_wait_until_valid",
                 {PhysicalRegionTy});
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionIsValidFunc(){
  return GetFunc("legion_physical_region_is_valid",
                 {PhysicalRegionTy},
                 Int1Ty);
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionGetFieldAccessorGenericFunc(){
  return GetFunc("legion_physical_region_get_field_accessor_generic",
                 {PhysicalRegionTy, FieldIdTy},
                 AccessorGenericTy);
}

llvm::Function*
CGCLegionCRuntime::PhysicalRegionGetFieldAccessorArrayFunc(){
  return GetFunc("legion_physical_region_get_field_accessor_array",
                 {PhysicalRegionTy, FieldIdTy},
                 AccessorArrayTy);
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericDestroyFunc(){
  return GetFunc("legion_accessor_generic_destroy",
                 {AccessorGenericTy});
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericReadFunc(){
  return GetFunc("legion_accessor_generic_read",
                 {AccessorGenericTy, PtrTy, VoidPtrTy, Int64Ty});
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericWriteFunc(){
  return GetFunc("legion_accessor_generic_write",
                 {AccessorGenericTy, PtrTy, VoidPtrTy, Int64Ty});
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr1dFunc(){
  return GetFunc("legion_accessor_generic_raw_rect_ptr_1d",
                 {AccessorGenericTy, Rect1dTy, PointerTy(Rect1dTy),
                   PointerTy(ByteOffsetTy)},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr2dFunc(){
  return GetFunc("legion_accessor_generic_raw_rect_ptr_2d",
                 {AccessorGenericTy, Rect2dTy, PointerTy(Rect2dTy),
                   PointerTy(ByteOffsetTy)},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::AccessorGenericRawRectPtr3dFunc(){
  return GetFunc("legion_accessor_generic_raw_rect_ptr_3d",
                 {AccessorGenericTy, Rect3dTy, PointerTy(Rect3dTy),
                   PointerTy(ByteOffsetTy)},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayDestroyFunc(){
  return GetFunc("legion_accessor_array_destroy",
                 {AccessorArrayTy});
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayReadFunc(){
  return GetFunc("legion_accessor_array_read",
                 {AccessorArrayTy, PtrTy, VoidPtrTy, Int64Ty});
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayWriteFunc(){
  return GetFunc("legion_accessor_array_write",
                 {AccessorArrayTy, PtrTy, VoidPtrTy, Int64Ty});
}

llvm::Function*
CGCLegionCRuntime::AccessorArrayRefFunc(){
  return GetFunc("legion_accessor_array_ref",
                 {AccessorArrayTy, PtrTy},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorCreateFunc(){
  return GetFunc("legion_index_iterator_create",
                 {IndexSpaceTy},
                 IndexIteratorTy);
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorDestroyFunc(){
  return GetFunc("legion_index_iterator_destroy",
                 {IndexIteratorTy});
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorHasNextFunc(){
  return GetFunc("legion_index_iterator_has_next",
                 {IndexIteratorTy},
                 Int1Ty);
}

llvm::Function*
CGCLegionCRuntime::IndexIteratorNextFunc(){
  return GetFunc("legion_index_iterator_next",
                 {IndexIteratorTy},
                 PtrTy);
}

llvm::Function*
CGCLegionCRuntime::TaskGetArgsFunc(){
  return GetFunc("legion_task_get_args",
                 {TaskTy},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::TaskGetArglenFunc(){
  return GetFunc("legion_task_get_arglen",
                 {TaskTy},
                 Int64Ty);
}

llvm::Function*
CGCLegionCRuntime::TaskGetIsIndexSpaceFunc(){
  return GetFunc("legion_task_get_is_index_space",
                 {TaskTy},
                 Int1Ty);
}

llvm::Function*
CGCLegionCRuntime::TaskGetLocalArgsFunc(){
  return GetFunc("legion_task_get_local_args",
                 {TaskTy},
                 VoidPtrTy);
}

llvm::Function*
CGCLegionCRuntime::TaskGetLocalArglenFunc(){
  return GetFunc("legion_task_get_local_arglen",
                 {TaskTy},
                 Int64Ty);
}

llvm::Function*
CGCLegionCRuntime::RuntimeStartFunc(){
  return GetFunc("legion_runtime_start",
                 {Int32Ty, PointerTy(PointerTy(VoidPtrTy)), Int1Ty},
                 Int32Ty);
}

llvm::Function*
CGCLegionCRuntime::RuntimeWaitForShutdownFunc(){
  return GetFunc("legion_runtime_wait_for_shutdown",
                 TypeVec());
}

llvm::Function*
CGCLegionCRuntime::RuntimeSetTopLevelTaskIdFunc(){
  return GetFunc("legion_runtime_set_top_level_task_id",
                 {TaskIdTy});
}

llvm::Function*
CGCLegionCRuntime::RuntimeGetInputArgsFunc(){
  return GetFunc("legion_runtime_get_input_args",
                 TypeVec(),
                 InputArgsTy);
}

llvm::Function*
CGCLegionCRuntime::RuntimeRegisterTaskVoidFunc(){
  return GetFunc("legion_runtime_register_task_void",
                 {TaskIdTy, Int32Ty, Int1Ty, Int1Ty, VariantIdTy,
                 TaskConfigOptionsTy, VoidPtrTy, VoidTaskFuncTy},
                 TaskIdTy);
}

llvm::Function*
CGCLegionCRuntime::RuntimeRegisterTaskFunc(){
  return GetFunc("legion_runtime_register_task",
                 {TaskIdTy, Int32Ty, Int1Ty, Int1Ty, VariantIdTy,
                   TaskConfigOptionsTy, VoidPtrTy, TaskFuncTy},
                 TaskIdTy);
}
