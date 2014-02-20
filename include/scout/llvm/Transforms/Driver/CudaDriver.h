/*
 *  
 *###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
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
 */ 

#ifndef _SC_CUDA_DRIVER_H_
#define _SC_CUDA_DRIVER_H_

#include "Driver.h"

class CudaDriver: public Driver {
  
 public:
  
  struct Memcpy {
    llvm::Value *host;
    llvm::Value *device;
    llvm::Value *size;

  Memcpy(llvm::Value *_host, llvm::Value *_device, llvm::Value *_size)
  :  host(_host), device(_device), size(_size) { }
  };

  CudaDriver(llvm::Module &module, llvm::IRBuilder<> &builder, bool debug = false);

  void setCUDA_ARRAY_DESCRIPTORTy(llvm::Module &module);
  void setCUDA_ARRAY3D_DESCRIPTORTy(llvm::Module &module);
  void setCUdevpropTy(llvm::Module &module);
  void setCUDA_MEMCPY2DTy(llvm::Module &module);
  void setCUDA_MEMCPY3DTy(llvm::Module &module);

  llvm::Type *getCUdeviceTy();
  llvm::Type *getCUcontextTy();
  llvm::Type *getCUmoduleTy();
  llvm::Type *getCUfunctionTy();
  llvm::Type *getCUdeviceptrTy();

  void setFnArgAttributes(llvm::SmallVector< llvm::ConstantInt *, 3 > args);
  void setMeshFieldNames(llvm::SmallVector< llvm::Value *, 3 > args);  
  void setDimensions(llvm::SmallVector< llvm::ConstantInt *, 3 > args);

  int getLinearizedMeshSize();

  bool isMeshMember(unsigned i);

  llvm::Value* meshFieldName(unsigned i);

  void create(llvm::Function *func, llvm::GlobalValue *ptxAsm,
              llvm::Value *meshName, llvm::MDNode* mdn);
  void initialize();
  void finalize();
  void destroy();

  llvm::Value *getDimension(int dim);
  bool setGridAndBlockSizes();

  void setGridSize(llvm::SmallVector< llvm::Constant *, 3 > &size);
  void setBlockSize(llvm::SmallVector< llvm::Constant *, 3 > &size);

  llvm::Value *insertCheckedCall(llvm::StringRef name, llvm::Value **begin, llvm::Value **end);
  llvm::Value *insertCheckedCall(llvm::StringRef name, llvm::ArrayRef< llvm::Value * > args);

  // Initialization
  llvm::Value *insertInit();

  // Device Management
  llvm::Value *insertDeviceComputeCapability(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertDeviceGet(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertDeviceGetAttribute(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertDeviceGetCount(llvm::Value *a);
  llvm::Value *insertDeviceGetName(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertDeviceGetProperties(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertDeviceTotalMem(llvm::Value *a, llvm::Value *b);

  // Version Management
  llvm::Value *insertDriverGetVersion(llvm::Value *a);

  // Context Management
  llvm::Value *insertCtxAttach(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertCtxCreate(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertCtxDestroy(llvm::Value *a);
  llvm::Value *insertCtxDetach(llvm::Value *a);
  llvm::Value *insertCtxGetDevice(llvm::Value *a);
  llvm::Value *insertCtxPopCurrent(llvm::Value *a);
  llvm::Value *insertCtxPushCurrent(llvm::Value *a);
  llvm::Value *insertCtxSynchronize();

  // Module Management
  llvm::Value *insertModuleGetFunction(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertModuleGetGlobal(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertModuleGetTexRef(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertModuleLoad(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertModuleLoadData(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertModuleLoadDataEx(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertModuleLoadFatBinary(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertModuleUnload(llvm::Value *a);

  // Stream Management
  llvm::Value *insertStreamCreate(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertStreamDestroy(llvm::Value *a);
  llvm::Value *insertStreamQuery(llvm::Value *a);
  llvm::Value *insertStreamSynchronize(llvm::Value *a);

  // Event Management
  llvm::Value *insertEventCreate(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertEventDestroy(llvm::Value *a);
  llvm::Value *insertEventElapsedTime(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertEventQuery(llvm::Value *a);
  llvm::Value *insertEventRecord(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertEventSynchronize(llvm::Value *a);

  // Execution Control
  llvm::Value *insertFuncGetAttribute(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertFuncSetBlockShape(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertFuncSetSharedSize(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertLaunch(llvm::Value *a);
  llvm::Value *insertLaunchGrid(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertLaunchGridAsync(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertParamSetf(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertParamSeti(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertParamSetSize(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertParamSetTexRef(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertParamSetv(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);

  // Memory Management
  llvm::Value *insertArray3DCreate(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertArray3DGetDescriptor(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertArrayCreate(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertArrayDestroy(llvm::Value *a);
  llvm::Value *insertArrayGetDescriptor(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemAlloc(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemAllocHost(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemAllocPitch(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemcpy2D(llvm::Value *a);
  llvm::Value *insertMemcpy2DAsync(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemcpy2DUnaligned(llvm::Value *a);
  llvm::Value *insertMemcpy3D(llvm::Value *a);
  llvm::Value *insertMemcpy3DAsync(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemcpyAtoA(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemcpyAtoD(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemcpyAtoH(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemcpyAtoHAsync(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemcpyDtoA(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemcpyDtoD(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemcpyDtoH(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemcpyDtoHAsync(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemcpyHtoA(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemcpyHtoAAsync(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemcpyHtoD(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemcpyHtoDAsync(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertMemFree(llvm::Value *a);
  llvm::Value *insertMemFreeHost(llvm::Value *a);
  llvm::Value *insertMemGetAddressRange(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemGetInfo(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemHostAlloc(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemHostGetDevicePointer(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemHostGetFlags(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertMemsetD16(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemsetD2D16(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemsetD2D32(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemsetD2D8(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d, llvm::Value *e);
  llvm::Value *insertMemsetD32(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertMemsetD8(llvm::Value *a, llvm::Value *b, llvm::Value *c);

  // Texture Reference Management
  llvm::Value *insertTexRefCreate(llvm::Value *a);
  llvm::Value *insertTexRefDestroy(llvm::Value *a);
  llvm::Value *insertTexRefGetAddress(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefGetAddressMode(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertTexRefGetArray(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefGetFilterMode(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefGetFlags(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefGetFormat(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertTexRefSetAddress(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertTexRefSetAddress2D(llvm::Value *a, llvm::Value *b, llvm::Value *c, llvm::Value *d);
  llvm::Value *insertTexRefSetAddressMode(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertTexRefSetArray(llvm::Value *a, llvm::Value *b, llvm::Value *c);
  llvm::Value *insertTexRefSetFilterMode(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefSetFlags(llvm::Value *a, llvm::Value *b);
  llvm::Value *insertTexRefSetFormat(llvm::Value *a, llvm::Value *b, llvm::Value *c);

  // Scout GPU Runtime
  llvm::Value *insertScoutGetModule(llvm::Value *a, llvm::Value *b);

 private:
  llvm::Type *CUaddress_modeTy;
  llvm::Type *CUarrayTy;
  llvm::Type *CUarray_formatTy;
  llvm::Type *CUDA_ARRAY_DESCRIPTORTy;
  llvm::Type *CUDA_ARRAY3D_DESCRIPTORTy;
  llvm::Type *CUcontextTy;
  llvm::Type *CUdeviceTy;
  llvm::Type *CUdeviceptrTy;
  llvm::Type *CUdevice_attributeTy;
  llvm::Type *CUdevpropTy;
  llvm::Type *CUeventTy;
  llvm::Type *CUfilter_modeTy;
  llvm::Type *CUfunctionTy;
  llvm::Type *CUfunction_attributeTy;
  llvm::Type *CUjit_optionTy;
  llvm::Type *CUDA_MEMCPY2DTy;
  llvm::Type *CUDA_MEMCPY3DTy;
  llvm::Type *CUmoduleTy;
  llvm::Type *CUresultTy;
  llvm::Type *CUstreamTy;
  llvm::Type *CUtexrefTy;

  llvm::GlobalVariable *cuDevice;
  llvm::GlobalVariable *cuContext;
  llvm::GlobalVariable *cuModule;

  llvm::SmallVector< llvm::Constant *, 3 > _gridSize;
  llvm::SmallVector< llvm::Constant *, 3 > _blockSize;

  llvm::SmallVector< llvm::ConstantInt *, 3 > fnArgAttrs;
  llvm::SmallVector< llvm::ConstantInt *, 3 > dimensions;
  llvm::SmallVector< llvm::Value *, 3 > meshFieldNames;
/* suppress unused field warning for now.
  llvm::Value *startX;
  llvm::Value *startY;
  llvm::Value *startZ;

  llvm::Value *stopX;
  llvm::Value *stopY;
  llvm::Value *stopZ;
*/
};

#endif
