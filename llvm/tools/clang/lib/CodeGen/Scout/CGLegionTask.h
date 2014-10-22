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

#ifndef CLANG_CODEGEN_LEGIONTASK_H
#define CLANG_CODEGEN_LEGIONTASK_H

#include "CGBuilder.h"
#include "CodeGenModule.h"
#include "Scout/CGLegionRuntime.h"

namespace llvm {
  class Function;
  class Value;
  class Type;
  class StructType;
}

namespace clang {
  class FunctionDecl;
  class MeshDecl;

  namespace CodeGen {
    class TaskDeclVisitor;
    class CodeGenFunction;
    class CGLegionTask {

      typedef std::vector<llvm::Value*> ValueVec;
      typedef std::vector<llvm::Type*> TypeVec;

      public:

      CGLegionTask(const FunctionDecl* FD, llvm::Function* taskFunc, CodeGenModule& codeGenModule, CGBuilderTy& builder, 
        CodeGenFunction* codeGenFunction);
      ~CGLegionTask(){}; 

      void EmitLegionTask();

      private:

      CodeGenModule& CGM;
      CGBuilderTy& B;
      CGLegionRuntime& R;
      CodeGenFunction* CGF;

      const FunctionDecl* funcDecl;
      MeshDecl* meshDecl;
      llvm::Function* taskFunc;
      llvm::Function* legionTaskInitFunc;
      llvm::Function* legionTaskFunc;
      llvm::StructType* meshType;
      std::vector<llvm::Value*> fields;
      llvm::Value* firstField;

      size_t taskId;
      llvm::Value* legionContext;
      llvm::Value* legionRuntime;
      llvm::Value* meshTaskArgs;


      // members used only for creating legion task init function
      llvm::Value* indexLauncher;
      llvm::Value* argMap;
      TaskDeclVisitor* taskDeclVisitor;
      size_t meshPos;       // position of the mesh in the argument list of LegionTaskInitFunc()
      llvm::Value* meshPtr;  // pointer to the scout mesh struct argument LegionTaskInitFunc()

      // used only for creating legion task function
      llvm::Value* taskArgs;
      llvm::Value* task;
      llvm::Value* regions;
      std::vector<llvm::Value*> taskFuncArgs;
      llvm::Value* mesh;     // legion-allocated scout mesh struct to be passed to forall

      // member functions for creating the legion task init function
      void EmitLegionTaskInitFunction();
      void EmitLegionTaskInitFunctionStart();
      void EmitUnimeshGetVecByNameFuncCalls();
      void EmitArgumentMapCreateFuncCall();
      llvm::Value* CreateMeshTaskArgs();
      void EmitSubgridBoundsAtSetFuncCall(llvm::Value* index);
      void EmitArgumentMapSetPointFuncCall(llvm::Value* index);
      void EmitIndexLauncherCreateFuncCall();
      void EmitAddMeshRegionReqAndFieldFuncCalls();
      void EmitAddVectorRegionReqAndFieldFuncCalls();
      void EmitExecuteIndexSpaceFuncCall();

      // member function for creating the legion task function
      void EmitLegionTaskFunction();
      void EmitLegionTaskFunctionStart();
      void EmitGetIndexSpaceDomainFuncCall();
      void EmitScoutMesh();
      void EmitMeshRawRectPtr1dFuncCalls();
      void EmitVectorRawRectPtr1dFuncCalls();
      void EmitTaskFuncCall();

    };
  }
}
#endif
