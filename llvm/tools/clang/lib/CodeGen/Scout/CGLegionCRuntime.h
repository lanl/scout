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

#ifndef CLANG_CODEGEN_LEGION_C_RUNTIME_H
#define CLANG_CODEGEN_LEGION_C_RUNTIME_H

#include "CodeGenModule.h"

namespace llvm {
  class Function;
}

namespace clang {
namespace CodeGen {
  class CodeGenModule;

  class CGLegionCRuntime{
  public:
    typedef std::vector<llvm::Type*> TypeVec;
    
    CGLegionCRuntime(CodeGen::CodeGenModule &CGM);
    
    ~CGLegionCRuntime();
    
    llvm::PointerType* PointerTy(llvm::Type* elementType);
    
    llvm::Value* GetNull(llvm::Type* T);
    
    llvm::Function* GetFunc(const std::string& funcName,
                            const TypeVec& argTypes,
                            llvm::Type* retType=0);
    
    llvm::PointerType* VoidPtrTy;
    llvm::IntegerType* Int1Ty;
    llvm::IntegerType* Int8Ty;
    llvm::IntegerType* Int32Ty;
    llvm::IntegerType* Int64Ty;
    llvm::Type* VoidTy;
    llvm::Type* StringTy;
    
    llvm::StructType* OpaqueTy;
    
    llvm::Type* RuntimeTy;
    llvm::Type* ContextTy;
    llvm::Type* ColoringTy;
    llvm::Type* DomainColoringTy;
    llvm::Type* IndexSpaceAllocatorTy;
    llvm::Type* ArgumentMapTy;
    llvm::Type* PredicateTy;
    llvm::Type* FutureTy;
    llvm::Type* FutureMapTy;
    llvm::Type* TaskLauncherTy;
    llvm::Type* IndexLauncherTy;
    llvm::Type* InlineLauncherTy;
    llvm::Type* PhysicalRegionTy;
    llvm::Type* AccessorGenericTy;
    llvm::Type* AccessorArrayTy;
    llvm::Type* IndexIteratorTy;
    llvm::Type* TaskTy;
    
    llvm::StructType* PtrTy;
    llvm::Type* Point1dTy;
    llvm::Type* Point2dTy;
    llvm::Type* Point3dTy;
    llvm::Type* Rect1dTy;
    llvm::Type* Rect2dTy;
    llvm::Type* Rect3dTy;
    
    llvm::IntegerType* LowlevelIdTy;
    llvm::IntegerType* ProcessorTy;
    llvm::IntegerType* DomainMaxRectDimTy;
    llvm::IntegerType* ReductionOpIdTy;
    llvm::IntegerType* AddressSpaceTy;
    llvm::IntegerType* TaskPriorityTy;
    llvm::IntegerType* ColorTy;
    llvm::IntegerType* IndexPartitionTy;
    llvm::IntegerType* FieldIdTy;
    llvm::IntegerType* TraceIdTy;
    llvm::IntegerType* MapperIdTy;
    llvm::IntegerType* ContextIdTy;
    llvm::IntegerType* InstanceIdTy;
    llvm::IntegerType* FieldSpaceIdTy;
    llvm::IntegerType* GenerationIdTy;
    llvm::IntegerType* TypeHandleTy;
    llvm::IntegerType* ProjectionIdTy;
    llvm::IntegerType* RegionTreeIdTy;
    llvm::IntegerType* DistributedIdTy;
    llvm::IntegerType* AddressSpaceIdTy;
    llvm::IntegerType* TunableIdTy;
    llvm::IntegerType* MappingTagIdTy;
    llvm::IntegerType* VariantIdTy;
    llvm::IntegerType* UniqueIdTy;
    llvm::IntegerType* VersionIdTy;
    llvm::IntegerType* TaskIdTy;
    
    llvm::Type* PrivilegeModeTy;
    llvm::Value* NoAccessVal;
    llvm::Value* ReadOnlyVal;
    llvm::Value* ReadWriteVal;
    llvm::Value* WriteOnlyVal;
    llvm::Value* WriteDiscardVal;
    llvm::Value* ReduceVal;
    llvm::Value* PromotedVal;
    
    llvm::Type* AllocateModeTy;
    llvm::Value* NoMemoryVal;
    llvm::Value* AllocableVal;
    llvm::Value* FreeableVal;
    llvm::Value* MutableVal;
    llvm::Value* RegionCreationVal;
    llvm::Value* RegionDeletionVal;
    llvm::Value* AllMemoryVal;
    
    llvm::Type* CoherencePropertyTy;
    llvm::Value* ExclusiveVal;
    llvm::Value* AtomicVal;
    llvm::Value* SimultaneousVal;
    llvm::Value* RelaxedVal;
    
    llvm::Type* RegionFlagsTy;
    llvm::Value* NoFlagVal;
    llvm::Value* VerifiedFlagVal;
    llvm::Value* NoAccessFlagVal;
    
    llvm::Type* HandleTy;
    llvm::Value* SingularVal;
    llvm::Value* PartProjectionVal;
    llvm::Value* RegProjectionVal;
    
    llvm::Type* DependenceTy;
    llvm::Value* NoDependenceVal;
    llvm::Value* TrueDependenceVal;
    llvm::Value* AntiDependenceVal;
    llvm::Value* AtomicDependenceVal;
    llvm::Value* SimultaneousDependenceVal;
    llvm::Value* PromotedDependenceVal;
    
    llvm::StructType* DomainTy;
    llvm::StructType* IndexSpaceTy;
    llvm::StructType* IndexAllocatorTy;
    llvm::StructType* FieldSpaceTy;
    llvm::StructType* FieldAllocatorTy;
    llvm::StructType* LogicalRegionTy;
    llvm::StructType* LogicalPartitionTy;
    llvm::StructType* TaskArgumentTy;
    llvm::StructType* TaskResultTy;
    llvm::StructType* ByteOffsetTy;
    llvm::StructType* InputArgsTy;
    llvm::StructType* TaskConfigOptionsTy;
    
    llvm::FunctionType* VoidTaskFuncTy;
    llvm::FunctionType* TaskFuncTy;
    
    llvm::Value* TocProcVal;
    llvm::Value* LocProcVal;
    llvm::Value* UtilProcVal;
    llvm::Value* ProcGroupVal;
    llvm::Value* MaxRectDimVal;

    // ======== Scout specific types =================

    llvm::Type* ScUniformMeshTy;
    llvm::Type* ScUniformMeshLauncherTy;

    llvm::Type* ScFieldKindTy;
    llvm::Value* ScInt32Val;
    llvm::Value* ScInt64Val;
    llvm::Value* ScFloatVal;
    llvm::Value* ScDoubleVal;

    llvm::Type* ScElementKindTy;
    llvm::Value* ScCellVal;
    llvm::Value* ScVertexVal;
    llvm::Value* ScEdgeVal;
    llvm::Value* ScFaceVal;

    // ==============================================

    llvm::Function* DomainFromRect1dFunc();
    llvm::Function* DomainFromRect2dFunc();
    llvm::Function* DomainFromRect3dFunc();
    llvm::Function* DomainFromIndexSpaceFunc();

    llvm::Function* ColoringCreateFunc();
    llvm::Function* ColoringDestroyFunc();
    llvm::Function* ColoringEnsureColorFunc();
    llvm::Function* ColoringAddPointFunc();
    llvm::Function* ColoringAddRangeFunc();
    
    llvm::Function* IndexSpaceCreateFunc();
    llvm::Function* IndexSpaceCreateDomainFunc();
    llvm::Function* IndexSpaceDestroyFunc();
    
    llvm::Function* IndexPartitionCreateColoringFunc();
    llvm::Function* IndexPartitionCreateDomainColoringFunc();
    llvm::Function* IndexPartitionDestroyFunc();
    
    llvm::Function* FieldSpaceCreateFunc();
    llvm::Function* FieldSpaceDestroyFunc();

    llvm::Function* LogicalRegionCreateFunc();
    llvm::Function* LogicalRegionDestroyFunc();

    llvm::Function* LogicalPartitionCreateFunc();
    llvm::Function* LogicalPartitionDestroyFunc();
    llvm::Function* LogicalPartitionGetLogicalSubregionFunc();
    llvm::Function* LogicalPartitionGetLogicalSubregionByColorFunc();
    
    llvm::Function* IndexAllocatorCreateFunc();
    llvm::Function* IndexAllocatorDestroyFunc();
    llvm::Function* IndexAllocatorAllocFunc();
    llvm::Function* IndexAllocatorFreeFunc();

    llvm::Function* FieldAllocatorCreateFunc();
    llvm::Function* FieldAllocatorDestroyFunc();
    llvm::Function* FieldAllocatorAllocateFieldFunc();
    llvm::Function* FieldAllocatorFreeFieldFunc();
    llvm::Function* FieldAllocatorAllocateLocalFieldFunc();
    
    llvm::Function* PredicateDestroyFunc();
    llvm::Function* PredicateTrueFunc();
    llvm::Function* PredicateFalseFunc();
    
    llvm::Function* FutureDestroyFunc();
    llvm::Function* FutureGetVoidResultFunc();
    llvm::Function* FutureGetResultFunc();
    llvm::Function* FutureIsEmptyFunc();
    llvm::Function* FutureGetUntypedPointerFunc();
    
    llvm::Function* TaskResultCreateFunc();
    llvm::Function* TaskResultDestroyFunc();
    
    llvm::Function* TaskLauncherCreateFunc();
    llvm::Function* TaskLauncherDestroyFunc();
    llvm::Function* TaskLauncherExecuteFunc();
    llvm::Function* TaskLauncherAddRegionRequirementLogicalRegionFunc();
    llvm::Function* TaskLauncherAddFieldFunc();
    
    llvm::Function* IndexLauncherCreateFunc();
    llvm::Function* IndexLauncherDestroyFunc();
    llvm::Function* IndexLauncherExecuteFunc();
    llvm::Function* IndexLauncherExecuteReductionFunc();
    llvm::Function* IndexLauncherAddRegionRequirementLogicalRegionFunc();
    llvm::Function* IndexLauncherAddRegionRequirementLogicalPartitionFunc();
    llvm::Function* IndexLauncherAddFieldFunc();
    
    llvm::Function* InlineLauncherCreateLogicalRegionFunc();
    llvm::Function* InlineLauncherDestroyFunc();
    llvm::Function* InlineLauncherExecuteFunc();
    llvm::Function* InlineLauncherAddFieldFunc();
    
    llvm::Function* RuntimeRemapRegionFunc();
    llvm::Function* RuntimeUnmapRegionFunc();
    llvm::Function* RuntimeMapAllRegionFunc();
    llvm::Function* RuntimeUnmapAllRegionsFunc();
    
    llvm::Function* PhysicalRegionDestroyFunc();
    llvm::Function* PhysicalRegionIsMappedFunc();
    llvm::Function* PhysicalRegionWaitUntilValidFunc();
    llvm::Function* PhysicalRegionIsValidFunc();
    llvm::Function* PhysicalRegionGetFieldAccessorGenericFunc();
    llvm::Function* PhysicalRegionGetFieldAccessorArrayFunc();
    
    llvm::Function* AccessorGenericDestroyFunc();
    llvm::Function* AccessorGenericReadFunc();
    llvm::Function* AccessorGenericWriteFunc();
    llvm::Function* AccessorGenericRawRectPtr1dFunc();
    llvm::Function* AccessorGenericRawRectPtr2dFunc();
    llvm::Function* AccessorGenericRawRectPtr3dFunc();
    
    llvm::Function* AccessorArrayDestroyFunc();
    llvm::Function* AccessorArrayReadFunc();
    llvm::Function* AccessorArrayWriteFunc();
    llvm::Function* AccessorArrayRefFunc();
    
    llvm::Function* IndexIteratorCreateFunc();
    llvm::Function* IndexIteratorDestroyFunc();
    llvm::Function* IndexIteratorHasNextFunc();
    llvm::Function* IndexIteratorNextFunc();
    
    llvm::Function* TaskGetArgsFunc();
    llvm::Function* TaskGetArglenFunc();
    llvm::Function* TaskGetIsIndexSpaceFunc();
    llvm::Function* TaskGetLocalArgsFunc();
    llvm::Function* TaskGetLocalArglenFunc();
    
    llvm::Function* RuntimeStartFunc();
    llvm::Function* RuntimeWaitForShutdownFunc();
    llvm::Function* RuntimeSetTopLevelTaskIdFunc();
    llvm::Function* RuntimeGetInputArgsFunc();
    llvm::Function* RuntimeRegisterTaskVoidFunc();
    llvm::Function* RuntimeRegisterTaskFunc();

    // ======== Scout specific functions =================
    
    llvm::Function* ScInitFunc();
    llvm::Function* ScStartFunc();
    llvm::Function* ScRegisterTaskFunc();
    llvm::Function* ScUniformMeshCreateFunc();
    llvm::Function* ScUniformMeshAddFieldFunc();
    llvm::Function* ScUniformMeshInitFunc();
    llvm::Function* ScUniformMeshReconstructFunc();
    llvm::Function* ScUniformMeshCreateLauncherFunc();
    llvm::Function* ScUniformMeshLauncherAddFieldFunc();
    llvm::Function* ScUniformMeshLauncherExecuteFunc();

    // ===================================================

    Address GetLegionRuntimeGlobal();
    Address GetLegionContextGlobal();
    Address GetLegionGlobal(std::string varName, llvm::Type* type);
    
  private:
    CodeGen::CodeGenModule& CGM;
  };
}
}
#endif // CLANG_CODEGEN_LEGION_C_RUNTIME_H
