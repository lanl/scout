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

#ifndef CLANG_CODEGEN_LEGIONRUNTIME_H
#define CLANG_CODEGEN_LEGIONRUNTIME_H

#include "CodeGenModule.h"

namespace llvm {
  class Function;
}

namespace clang {
namespace CodeGen {
  class CodeGenModule;

  class CGLegionRuntime {
  public:
    CGLegionRuntime(CodeGen::CodeGenModule &CGM);
    virtual ~CGLegionRuntime();
    
    llvm::PointerType* PointerTy(llvm::Type* elementType);
    
    llvm::Value *GetLegionRuntimeGlobal();
    llvm::Value *GetLegionContextGlobal();
    llvm::Function *CreateSetupMeshFunction(llvm::Type *MT);
    llvm::Function *CreateAddFieldFunction(llvm::Type *MT);
    
    llvm::Function* SizeofCXXRect1dFunc();
    llvm::Function* SubgridBoundsAtFunc();
    llvm::Function* SubgridBoundsAtSetFunc();
    llvm::Function* VectorDumpFunc();
    llvm::Function* ArgumentMapCreateFunc();
    llvm::Function* ArgumentMapSetPointFunc();
    llvm::Function* IndexLauncherCreateFunc();
    llvm::Function* AddRegionRequirementFunc();
    llvm::Function* AddFieldFunc();
    llvm::Function* ExecuteIndexSpaceFunc();
    llvm::Function* VectorCreateFunc();
    llvm::Function* VectorFreeFunc();
    
    llvm::Function* UnimeshCreateFunc();
    llvm::Function* UnimeshFreeFunc();
    llvm::Function* UnimeshAddFieldFunc();
    llvm::Function* UnimeshPartitionFunc();
    llvm::Function* UnimeshGetVecByNameFunc();

    llvm::Function* StructCreateFunc();
    llvm::Function* StructAddFieldFunc();
    llvm::Function* StructPartitionFunc();
    llvm::Function* StructGetVecByNameFunc();
    
    llvm::Function* StartFunc();
    llvm::Function* SetTopLevelTaskIdFunc();
    llvm::Function* RegisterVoidLegionTaskFunc();
    llvm::Function* RegisterVoidLegionTaskAuxFunc();
    llvm::Function* RawRectPtr1dFunc();
    llvm::Function* PrintMeshTaskArgsFunc();
    llvm::Function* PrintTaskArgsLocalArgspFunc();
    
    llvm::Value* GetNull(llvm::Type* T);
    
    llvm::PointerType* VoidPtrTy;
    llvm::Type* Int1Ty;
    llvm::Type* Int8Ty;
    llvm::Type* Int32Ty;
    llvm::Type* Int64Ty;
    llvm::Type* VoidTy;
    
    llvm::Type* RuntimeTy;
    llvm::Type* ContextTy;
    llvm::Type* LogicalRegionTy;
    llvm::Type* LogicalPartitionTy;
    llvm::Type* IndexSpaceTy;
    llvm::Type* DomainHandleTy;
    llvm::Type* PhysicalRegionsTy;
    llvm::Type* Rect1dTy;
    llvm::Type* FieldIdTy;
    llvm::Type* IndexLauncherHandleTy;
    llvm::Type* TaskTy;
    llvm::Type* ArgumentMapHandleTy;
    llvm::Type* ProjectionIdTy;
    llvm::Type* RegionRequirementHndlTy;
    llvm::Type* UnimeshHandleTy;
    llvm::Type* StructHandleTy;
    llvm::Type* VariantIdTy;
    
    llvm::StructType* Rect1dStorageTy;
    llvm::StructType* DomainTy;
    llvm::StructType* VectorTy;
    llvm::StructType* ArgumentMapTy;
    llvm::StructType* IndexLauncherTy;
    llvm::StructType* RegionRequirementTy;
    llvm::StructType* UnimeshTy;
    llvm::StructType* StructTy;
    llvm::StructType* TaskArgsTy;
    llvm::StructType* RegTaskDataTy;
    llvm::StructType* MeshTaskArgsTy;
    
    llvm::Value* SuccessVal;
    llvm::Value* FailureVal;
    llvm::Value* NoAccessVal;
    llvm::Value* ReadOnlyVal;
    llvm::Value* ReadWriteVal;
    llvm::Value* WriteOnlyVal;
    llvm::Value* WriteDiscardVal;
    llvm::Value* ReduceVal;
    llvm::Value* PromotedVal;
    llvm::Value* ExclusiveVal;
    llvm::Value* AtomicVal;
    llvm::Value* SimultaenousVal;
    llvm::Value* RelaxedVal;
    llvm::Value* TypeInt32Val;
    llvm::Value* TypeInt64Val;
    llvm::Value* TypeFloatVal;
    llvm::Value* TypeDoubleVal;
    llvm::Value* TocProcVal;
    llvm::Value* LocProcVal;
    llvm::Value* UtilProcVal;
    llvm::Value* DomainHandleVal;
    llvm::Value* DomainVolumeVal;
    llvm::Value* VectorLRVal;
    llvm::Value* VectorFIDVal;
    llvm::Value* VectorIndexSpaceVal;
    llvm::Value* VectorLogicalRegionVal;
    llvm::Value* VectorLogicalPartitionVal;
    llvm::Value* VectorLaunchDomainVal;
    llvm::Value* VectorSubgridBoundsLenVal;
    llvm::Value* VectorSubgridBoundsVal;
    
  protected:
    CodeGen::CodeGenModule &CGM;
    llvm::Value *GetLegionGlobal(std::string varName, llvm::Type *type);
    llvm::Function *LegionRuntimeFunction(std::string funcName, std::vector<llvm::Type*> Params);
    llvm::Function *LegionRuntimeFunction(std::string funcName, std::vector<llvm::Type*> Params,
                                          llvm::Type* retType);
  };
}
}
#endif
