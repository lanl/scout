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

#ifndef _SC_PTX_DRIVER_H_
#define _SC_PTX_DRIVER_H_

#include "Driver.h"

class PTXDriver: public Driver {
 public:

  PTXDriver(llvm::Module &module, llvm::IRBuilder<> &builder, bool debug = false);

  enum Axis { X = 120, Y, Z };

  void setBlocksInY(const unsigned blocksInY);

  llvm::Value *insertGetThreadIdx(int dim);
  llvm::Value *insertGetBlockDim(int dim);
  llvm::Value *insertGetBlockIdx(int dim);
  llvm::Value *insertGetGridDim(int dim);
  llvm::Value *insertGetGlobalThreadDim(int dim);
  llvm::Value *insertGetGlobalThreadIdx(int dim);
 private:
  llvm::Constant *_blocksInY;
};

#endif
