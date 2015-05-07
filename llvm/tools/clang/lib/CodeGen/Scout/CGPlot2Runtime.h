/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
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

#ifndef CLANG_CODEGEN_PLOT2_RUNTIME_H
#define CLANG_CODEGEN_PLOT2_RUNTIME_H

#include "CodeGenModule.h"

namespace llvm {
  class Function;
}

namespace clang {
namespace CodeGen {
  class CodeGenModule;

  class CGPlot2Runtime{
  public:
    typedef std::vector<llvm::Type*> TypeVec;
    
    CGPlot2Runtime(CodeGen::CodeGenModule &CGM);
    
    ~CGPlot2Runtime();
    
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
    llvm::Type* FloatTy;
    llvm::Type* DoubleTy;
    llvm::Type* VoidTy;
    llvm::Type* StringTy;
    llvm::FunctionType* PlotFuncI32Ty;
    llvm::FunctionType* PlotFuncI64Ty;
    llvm::FunctionType* PlotFuncFloatTy;
    llvm::FunctionType* PlotFuncDoubleTy;

    llvm::FunctionType* PlotFuncI32VecTy;
    llvm::FunctionType* PlotFuncI64VecTy;
    llvm::FunctionType* PlotFuncFloatVecTy;
    llvm::FunctionType* PlotFuncDoubleVecTy;
    
    llvm::Value* ElementInt32Val;
    llvm::Value* ElementInt64Val;
    llvm::Value* ElementFloatVal;
    llvm::Value* ElementDoubleVal;

    llvm::Function* CreateFrameFunc();
    llvm::Function* CreateMeshFrameFunc();
    
    llvm::Function* FrameAddVarFunc();
    llvm::Function* FrameAddArrayVarFunc();
    
    llvm::Function* FrameCaptureI32Func();
    llvm::Function* FrameCaptureI64Func();
    llvm::Function* FrameCaptureFloatFunc();
    llvm::Function* FrameCaptureDoubleFunc();

    llvm::Function* PlotGetFunc();
    llvm::Function* PlotInitFunc();
    llvm::Function* PlotReadyFunc();
    llvm::Function* PlotGetI32Func();
    llvm::Function* PlotGetI64Func();
    llvm::Function* PlotGetFloatFunc();
    llvm::Function* PlotGetDoubleFunc();
    llvm::Function* PlotAddLinesFunc();
    llvm::Function* PlotAddLineFunc();
    llvm::Function* PlotAddPointsFunc();
    llvm::Function* PlotAddAreaFunc();
    llvm::Function* PlotAddIntervalFunc();
    llvm::Function* PlotAddPieFunc();
    llvm::Function* PlotAddBinsFunc();
    llvm::Function* PlotAddProportionFunc();
    llvm::Function* PlotProportionAddVarFunc();
    llvm::Function* PlotAddAxisFunc();
    llvm::Function* PlotRenderFunc();
    llvm::Function* PlotAddVarI32Func();
    llvm::Function* PlotAddVarI64Func();
    llvm::Function* PlotAddVarFloatFunc();
    llvm::Function* PlotAddVarDoubleFunc();
    llvm::Function* PlotAddAggregateFunc();
    llvm::Function* AggregateAddVarFunc();
    
    llvm::Function* PlotAddVarI32VecFunc();
    llvm::Function* PlotAddVarI64VecFunc();
    llvm::Function* PlotAddVarFloatVecFunc();
    llvm::Function* PlotAddVarDoubleVecFunc();
    
    llvm::Function* PlotSetAntialiasedFunc();
    
  private:
    CodeGen::CodeGenModule& CGM;
  };
}
}
#endif // CLANG_CODEGEN_PLOT2_RUNTIME_H
