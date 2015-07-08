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

#include "Scout/CGPlotRuntime.h"
#include "CodeGenFunction.h"

using namespace std;
using namespace clang;
using namespace CodeGen;
using namespace llvm;

CGPlotRuntime::CGPlotRuntime(CodeGenModule& CGM) : CGM(CGM){
  llvm::LLVMContext& C = CGM.getLLVMContext();
 
  Int1Ty = llvm::IntegerType::getInt1Ty(C);
  Int8Ty = llvm::IntegerType::getInt8Ty(C);
  Int32Ty = llvm::IntegerType::getInt32Ty(C);
  Int64Ty = llvm::IntegerType::getInt64Ty(C);
  FloatTy = llvm::Type::getFloatTy(C);
  DoubleTy = llvm::Type::getDoubleTy(C);
  VoidTy = llvm::Type::getVoidTy(C);
  VoidPtrTy = PointerTy(Int8Ty);
  StringTy = PointerTy(Int8Ty);
  
  TypeVec params = {VoidPtrTy, Int64Ty};
  PlotFuncI32Ty = llvm::FunctionType::get(Int32Ty, params, false);
  PlotFuncI64Ty = llvm::FunctionType::get(Int64Ty, params, false);
  PlotFuncFloatTy = llvm::FunctionType::get(FloatTy, params, false);
  PlotFuncDoubleTy = llvm::FunctionType::get(DoubleTy, params, false);
  
  params = {VoidPtrTy, Int64Ty, PointerTy(Int32Ty)};
  PlotFuncI32VecTy = llvm::FunctionType::get(VoidTy, params, false);
  
  params = {VoidPtrTy, Int64Ty, PointerTy(Int64Ty)};
  PlotFuncI64VecTy = llvm::FunctionType::get(VoidTy, params, false);
  
  params = {VoidPtrTy, Int64Ty, PointerTy(FloatTy)};
  PlotFuncFloatVecTy = llvm::FunctionType::get(VoidTy, params, false);
  
  params = {VoidPtrTy, Int64Ty, PointerTy(DoubleTy)};
  PlotFuncDoubleVecTy = llvm::FunctionType::get(VoidTy, params, false);
  
  ElementInt32Val = ConstantInt::get(C, APInt(32, 0));
  ElementInt64Val = ConstantInt::get(C, APInt(32, 1));
  ElementFloatVal = ConstantInt::get(C, APInt(32, 2));
  ElementDoubleVal = ConstantInt::get(C, APInt(32, 3));
  ElementStringVal = ConstantInt::get(C, APInt(32, 4));
}

CGPlotRuntime::~CGPlotRuntime(){}

Value* CGPlotRuntime::GetNull(llvm::Type* T){
  return ConstantPointerNull::get(PointerTy(T));
}

llvm::PointerType* CGPlotRuntime::PointerTy(llvm::Type* elementType){
  return llvm::PointerType::get(elementType, 0);
}

llvm::Function*
CGPlotRuntime::GetFunc(const std::string& funcName,
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
CGPlotRuntime::CreateFrameFunc(){
  return GetFunc("__scrt_create_frame", TypeVec(), VoidPtrTy);
}

llvm::Function*
CGPlotRuntime::CreateMeshFrameFunc(){
  return GetFunc("__scrt_create_mesh_frame", {Int32Ty, Int32Ty, Int32Ty},
                 VoidPtrTy);
}

llvm::Function*
CGPlotRuntime::FrameAddVarFunc(){
  return GetFunc("__scrt_frame_add_var", {VoidPtrTy, Int32Ty, Int32Ty});
}

llvm::Function*
CGPlotRuntime::FrameAddArrayVarFunc(){
  return GetFunc("__scrt_frame_add_array_var",
                 {VoidPtrTy, Int32Ty, Int32Ty, VoidPtrTy, Int64Ty});
}

llvm::Function* CGPlotRuntime::FrameCaptureI32Func(){
  return GetFunc("__scrt_frame_capture_i32", {VoidPtrTy, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::FrameCaptureI64Func(){
  return GetFunc("__scrt_frame_capture_i64", {VoidPtrTy, Int32Ty, Int64Ty});
}

llvm::Function* CGPlotRuntime::FrameCaptureFloatFunc(){
  return GetFunc("__scrt_frame_capture_float", {VoidPtrTy, Int32Ty, FloatTy});
}

llvm::Function* CGPlotRuntime::FrameCaptureDoubleFunc(){
  return GetFunc("__scrt_frame_capture_double", {VoidPtrTy, Int32Ty, DoubleTy});
}

llvm::Function* CGPlotRuntime::FrameCaptureStringFunc(){
  return GetFunc("__scrt_frame_capture_string", {VoidPtrTy, Int32Ty, StringTy});
}

llvm::Function* CGPlotRuntime::PlotGetFunc(){
  return GetFunc("__scrt_plot_get", {Int64Ty}, VoidPtrTy);
}

llvm::Function* CGPlotRuntime::PlotInitFunc(){
  return GetFunc("__scrt_plot_init", {VoidPtrTy, VoidPtrTy, VoidPtrTy});
}

llvm::Function* CGPlotRuntime::PlotReadyFunc(){
  return GetFunc("__scrt_plot_ready", {VoidPtrTy}, Int1Ty);
}

llvm::Function* CGPlotRuntime::PlotGetI32Func(){
  return GetFunc("__scrt_plot_get_i32", {VoidPtrTy, Int32Ty, Int64Ty}, Int32Ty);
}

llvm::Function* CGPlotRuntime::PlotGetI64Func(){
  return GetFunc("__scrt_plot_get_i64", {VoidPtrTy, Int32Ty, Int64Ty}, Int64Ty);
}

llvm::Function* CGPlotRuntime::PlotGetFloatFunc(){
  return GetFunc("__scrt_plot_get_float", {VoidPtrTy, Int32Ty, Int64Ty}, FloatTy);
}

llvm::Function* CGPlotRuntime::PlotGetDoubleFunc(){
  return GetFunc("__scrt_plot_get_double", {VoidPtrTy, Int32Ty, Int64Ty}, DoubleTy);
}

llvm::Function* CGPlotRuntime::PlotAddLinesFunc(){
  return GetFunc("__scrt_plot_add_lines",
                 {VoidPtrTy, Int32Ty, Int32Ty, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddPointsFunc(){
  return GetFunc("__scrt_plot_add_points",
                 {VoidPtrTy, Int32Ty, Int32Ty, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddLineFunc(){
  return GetFunc("__scrt_plot_add_line",
                 {VoidPtrTy, Int32Ty, Int32Ty, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddAreaFunc(){
  return GetFunc("__scrt_plot_add_area",
                 {VoidPtrTy, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddIntervalFunc(){
  return GetFunc("__scrt_plot_add_interval",
                 {VoidPtrTy, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddPieFunc(){
  return GetFunc("__scrt_plot_add_pie",
                 {VoidPtrTy, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddBinsFunc(){
  return GetFunc("__scrt_plot_add_bins",
                 {VoidPtrTy, Int32Ty, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddProportionFunc(){
  return GetFunc("__scrt_plot_add_proportion",
                 {VoidPtrTy, Int32Ty}, VoidPtrTy);
}

llvm::Function* CGPlotRuntime::PlotProportionAddVarFunc(){
  return GetFunc("__scrt_plot_proportion_add_var",
                 {VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddAxisFunc(){
  return GetFunc("__scrt_plot_add_axis",
                 {VoidPtrTy, Int32Ty, StringTy, Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotRenderFunc(){
  return GetFunc("__scrt_plot_render", {VoidPtrTy});
}

llvm::Function* CGPlotRuntime::PlotRefreshFunc(){
  return GetFunc("__scrt_plot_refresh", {VoidPtrTy});
}

llvm::Function* CGPlotRuntime::PlotAddVarI32Func(){
  return GetFunc("__scrt_plot_add_var_i32",
  {VoidPtrTy, Int32Ty, VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarI64Func(){
  return GetFunc("__scrt_plot_add_var_i64",
  {VoidPtrTy, Int32Ty, VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarFloatFunc(){
  return GetFunc("__scrt_plot_add_var_float",
  {VoidPtrTy, Int32Ty, VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarDoubleFunc(){
  return GetFunc("__scrt_plot_add_var_double",
  {VoidPtrTy, Int32Ty, VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarI32VecFunc(){
  return GetFunc("__scrt_plot_add_var_i32_vec",
                 {VoidPtrTy, Int32Ty, VoidPtrTy,
                   Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarI64VecFunc(){
  return GetFunc("__scrt_plot_add_var_i64_vec",
                 {VoidPtrTy, Int32Ty, VoidPtrTy,
                   Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarFloatVecFunc(){
  return GetFunc("__scrt_plot_add_var_float_vec",
                 {VoidPtrTy, Int32Ty, VoidPtrTy,
                   Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddVarDoubleVecFunc(){
  return GetFunc("__scrt_plot_add_var_double_vec",
                 {VoidPtrTy, Int32Ty, VoidPtrTy,
                   Int32Ty, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotAddAggregateFunc(){
  return GetFunc("__scrt_plot_add_aggregate",
                 {VoidPtrTy, Int64Ty, Int32Ty, Int32Ty}, VoidPtrTy);
}

llvm::Function* CGPlotRuntime::AggregateAddVarFunc(){
  return GetFunc("__scrt_aggregate_add_var",
                 {VoidPtrTy, Int32Ty});
}

llvm::Function* CGPlotRuntime::PlotSetAntialiasedFunc(){
  return GetFunc("__scrt_plot_set_antialiased",
                 {VoidPtrTy, Int1Ty});
}

llvm::Function* CGPlotRuntime::PlotSetOutputFunc(){
  return GetFunc("__scrt_plot_set_output",
                 {VoidPtrTy, StringTy});
}

llvm::Function* CGPlotRuntime::PlotSetRangeFunc(){
  return GetFunc("__scrt_plot_set_range",
                 {VoidPtrTy, Int1Ty, DoubleTy, DoubleTy});
}

Function* CGPlotRuntime::PlotAddVarFunc(){
  return GetFunc("__scrt_plot_add_var",
                 {VoidPtrTy, Int32Ty, Int32Ty});
}

Function* CGPlotRuntime::PlotCaptureI32Func(){
  return GetFunc("__scrt_plot_capture_i32",
                 {VoidPtrTy, Int32Ty, Int32Ty});
}

Function* CGPlotRuntime::PlotCaptureI64Func(){
  return GetFunc("__scrt_plot_capture_i64",
                 {VoidPtrTy, Int32Ty, Int64Ty});
}

Function* CGPlotRuntime::PlotCaptureFloatFunc(){
  return GetFunc("__scrt_plot_capture_float",
                 {VoidPtrTy, Int32Ty, FloatTy});
}

Function* CGPlotRuntime::PlotCaptureDoubleFunc(){
  return GetFunc("__scrt_plot_capture_double",
                 {VoidPtrTy, Int32Ty, DoubleTy});
}
