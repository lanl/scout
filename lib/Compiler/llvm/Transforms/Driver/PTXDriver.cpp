/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */

#include "llvm/Transforms/Scout/Driver/PTXDriver.h"

#include <iostream>

using namespace llvm;

PTXDriver::PTXDriver(Module &module, IRBuilder<> &builder, bool debug)
  : Driver(module, builder, debug), _blocksInY(NULL)
{
  
}

llvm::Value *PTXDriver::insertGetThreadIdx(int dim) {
  assert((dim == X || dim == Y || dim == Z) &&
         "ThreadIdx.* must specify a dimension x, y, or z!");

  std::string f = 
    std::string("llvm.nvvm.read.ptx.sreg.tid.").append(1, dim);

  Function* function = getModule().getFunction(f);
  if(!function){
    llvm::Type* i32 = llvm::Type::getInt32Ty(getModule().getContext());

    llvm::FunctionType* ft =
      llvm::FunctionType::get(i32, false);
    
    function =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                             f, &getModule());
    
  }

  return getBuilder().CreateCall(function);
}

llvm::Value *PTXDriver::insertGetBlockDim(int dim) {
  assert((dim == X || dim == Y || dim == Z) &&
         "BlockDim.* must specify a dimension x, y, or z!");

  std::string f = 
    std::string("llvm.nvvm.read.ptx.sreg.ntid.").append(1, dim);

  Function* function = getModule().getFunction(f);
  if(!function){
    llvm::Type* i32 = llvm::Type::getInt32Ty(getModule().getContext());

    llvm::FunctionType* ft =
      llvm::FunctionType::get(i32, false);
    
    function =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                             f, &getModule());
  }

  return getBuilder().CreateCall(function);  
}

llvm::Value *PTXDriver::insertGetBlockIdx(int dim) {
  assert((dim == X || dim == Y) &&
         "BlockIdx.* must specify a dimension x, or y!");

  std::string f = 
    std::string("llvm.nvvm.read.ptx.sreg.ctaid.").append(1, dim);

  Function* function = getModule().getFunction(f);
  if(!function){
    llvm::Type* i32 = llvm::Type::getInt32Ty(getModule().getContext());
    
    llvm::FunctionType* ft =
      llvm::FunctionType::get(i32, false);
    
    function =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                             f, &getModule());
  }

  return getBuilder().CreateCall(function);
}

llvm::Value *PTXDriver::insertGetGridDim(int dim) {
  assert((dim == X || dim == Y) &&
         "GridDim.* must specify dimension x or y!");

  std::string f = 
    std::string("llvm.nvvm.read.ptx.sreg.nctaid.").append(1, dim);

  Function* function = getModule().getFunction(f);
  if(!function){
    llvm::Type* i32 = llvm::Type::getInt32Ty(getModule().getContext());

    llvm::FunctionType* ft =
      llvm::FunctionType::get(i32, false);
    
    function =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                             f, &getModule());
  }

  return getBuilder().CreateCall(function);
}

void PTXDriver::setBlocksInY(const unsigned blocksInY) {
  Type *i32Ty = Type::getInt32Ty(getGlobalContext());
  _blocksInY = llvm::ConstantInt::get(i32Ty, blocksInY);
}

llvm::Value *PTXDriver::insertGetGlobalThreadIdx(int dim) {
  llvm::Value *blockIdx;
  llvm::Value *blockIdx32;
  Type *i32Ty = Type::getInt32Ty(_module.getContext());
  //Type *i64Ty = Type::getInt64Ty(_module.getContext());
  if(_blocksInY) {
    blockIdx   = insertGetBlockIdx(Y);
    blockIdx32 = _builder.CreateZExt(blockIdx, i32Ty);
    if(dim == Y)     blockIdx32 = _builder.CreateSDiv(blockIdx32, _blocksInY);
    else/*dim == Z*/ blockIdx32 = _builder.CreateSRem(blockIdx32, _blocksInY);
  } else {
    blockIdx   = insertGetBlockIdx(dim);
    blockIdx32 = _builder.CreateZExt(blockIdx, i32Ty);
  }

  llvm::Value *blockDim   = insertGetBlockDim(dim);
  llvm::Value *blockDim32 = _builder.CreateZExt(blockDim, i32Ty);
  llvm::Value *blockStart = _builder.CreateMul(blockIdx32, blockDim32);

  llvm::Value *threadIdx   = insertGetThreadIdx(dim);
  llvm::Value *threadIdx32 = _builder.CreateZExt(threadIdx, i32Ty);

  llvm::Value *globalThreadIdx = _builder.CreateAdd(threadIdx32, blockStart, "Tid");
  return globalThreadIdx;
}
