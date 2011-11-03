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

using namespace llvm;

static void createSpecialReg(Module &module, Type *type,
                             const StringRef name, Constant *cons) {
  if(!module.getNamedGlobal(name)) {
    new GlobalVariable(module,
                       type,
                       false,
                       GlobalValue::ExternalLinkage,
                       cons,
                       name);
  }
}

PTXDriver::PTXDriver(Module &module, IRBuilder<> &builder, bool debug)
  : Driver(module, builder, debug), _blocksInY(NULL)
{
  Type *i16Ty = llvm::Type::getInt16Ty(llvm::getGlobalContext());
  Constant *zero = llvm::ConstantInt::get(i16Ty, 0);
  Constant *one  = llvm::ConstantInt::get(i16Ty, 1);
  createSpecialReg(module, i16Ty, "__ptx_sreg_tid_x", zero);
  createSpecialReg(module, i16Ty, "__ptx_sreg_tid_y", zero);
  createSpecialReg(module, i16Ty, "__ptx_sreg_tid_z", zero);
  createSpecialReg(module, i16Ty, "__ptx_sreg_ntid_x", one);
  createSpecialReg(module, i16Ty, "__ptx_sreg_ntid_y", one);
  createSpecialReg(module, i16Ty, "__ptx_sreg_ntid_z", one);
  createSpecialReg(module, i16Ty, "__ptx_sreg_ctaid_x", zero);
  createSpecialReg(module, i16Ty, "__ptx_sreg_ctaid_y", zero);
  createSpecialReg(module, i16Ty, "__ptx_sreg_nctaid_x", one);
  createSpecialReg(module, i16Ty, "__ptx_sreg_nctaid_y", one);
}

llvm::Value *PTXDriver::insertGetThreadIdx(int dim) {
  assert(dim == X || dim == Y || dim == Z &&
         "ThreadIdx.* must specify a dimension x, y, or z!");
  std::string reg = std::string("__ptx_sreg_tid_").append(1, dim);
  return insertGet(reg.c_str());
}

llvm::Value *PTXDriver::insertGetBlockDim(int dim) {
  assert(dim == X || dim == Y || dim == Z &&
         "BlockDim.* must specify a dimension x, y, or z!");
  std::string reg = std::string("__ptx_sreg_ntid_").append(1, dim);
  return insertGet(reg.c_str());
}

llvm::Value *PTXDriver::insertGetBlockIdx(int dim) {
  assert(dim == X || dim == Y &&
         "BlockIdx.* must specify a dimension x, or y!");
  std::string reg = std::string("__ptx_sreg_ctaid_").append(1, dim);
  return insertGet(reg.c_str());
}

llvm::Value *PTXDriver::insertGetGridDim(int dim) {
  assert(dim == X || dim == Y &&
         "GridDim.* must specify dimension x or y!");
  std::string reg = std::string("__ptx_sreg_nctaid_").append(1, dim);
  return insertGet(reg.c_str());
}

void PTXDriver::setBlocksInY(const unsigned blocksInY) {
  Type *i32Ty = Type::getInt32Ty(getGlobalContext());
  _blocksInY = llvm::ConstantInt::get(i32Ty, blocksInY);
}

llvm::Value *PTXDriver::insertGetGlobalThreadIdx(int dim) {
  llvm::Value *blockIdx;
  llvm::Value *blockIdx32;
  Type *i32Ty = Type::getInt32Ty(getGlobalContext());
  Type *i64Ty = Type::getInt64Ty(getGlobalContext());
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
  return _builder.CreateZExt(globalThreadIdx, i64Ty);
}
