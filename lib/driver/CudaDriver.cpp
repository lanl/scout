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

#include "driver/CudaDriver.h"
#include "driver/types.h"

using namespace llvm;

CudaDriver::CudaDriver(Module &module, IRBuilder<> &builder, bool debug)
  : Driver(module, builder, debug),
    CUaddress_modeTy(i32Ty),
    CUarrayTy(getOrInsertType(module, "struct.CUarray_st")),
    CUarray_formatTy(i32Ty),
    CUcontextTy(getOrInsertType(module, "struct.CUctx_st")),
    CUdeviceTy(i32Ty),
    CUdeviceptrTy(IntegerType::get(getGlobalContext(), sizeof(uintptr_t) * 8)),
    CUdevice_attributeTy(i32Ty),
    CUeventTy(getOrInsertType(module, "struct.CUevent_st")),
    CUfilter_modeTy(i32Ty),
    CUfunctionTy(getOrInsertType(module, "struct.CUfunc_st")),
    CUfunction_attributeTy(i32Ty),
    CUjit_optionTy(i32PtrTy),
    CUmoduleTy(getOrInsertType(module, "struct.CUmod_st")),
    CUresultTy(i32Ty),
    CUstreamTy(getOrInsertType(module, "struct.CUstream_st")),
    CUtexrefTy(getOrInsertType(module, "struct.CUtexref_st")),
    _gridSize(SmallVector< Constant *, 3 >(3, ConstantInt::get(i32Ty, 1))),
    _blockSize(SmallVector< Constant *, 3 >(3, ConstantInt::get(i32Ty, 1)))
{
  setCUDA_ARRAY_DESCRIPTORTy(module);
  setCUDA_ARRAY3D_DESCRIPTORTy(module);
  setCUdevpropTy(module);
  setCUDA_MEMCPY2DTy(module);
  setCUDA_MEMCPY3DTy(module);

#include <driver/CudaDriverDeclarations.h>
}

void CudaDriver::setGridSize(SmallVector< Constant *, 3 > &size) {
  _gridSize = size;
}

void CudaDriver::setBlockSize(SmallVector< Constant *, 3 > &size) {
  _blockSize = size;
}

void CudaDriver::setCUDA_ARRAY_DESCRIPTORTy(Module &module) {
  //%struct.CUDA_ARRAY_DESCRIPTOR = type { i32, i32, i32, i32 }
  std::vector< Type * > params(4, i32Ty);
  Type *structType = StructType::get(getGlobalContext(), params);
  CUDA_ARRAY_DESCRIPTORTy = getOrInsertType(module,
                                            "struct.CUDA_ARRAY_DESCRIPTOR",
                                            structType);
}

void CudaDriver::setCUDA_ARRAY3D_DESCRIPTORTy(Module &module) {
  //%struct.CUDA_ARRAY3D_DESCRIPTOR = type { i32, i32, i32, i32, i32, i32 }
  std::vector< Type * > params(6, i32Ty);
  Type *structType = StructType::get(getGlobalContext(), params);
  CUDA_ARRAY3D_DESCRIPTORTy = getOrInsertType(module,
                                          "struct.CUDA_ARRAY3D_DESCRIPTOR",
                                          structType);
}

void CudaDriver::setCUdevpropTy(Module &module) {
  //%struct.CUdevprop = type { i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32 }
  std::vector< Type * > params(10, i32Ty);
  params[1] = params[2] = ArrayType::get(i32Ty, 3);
  Type *structType = StructType::get(getGlobalContext(), params);
  CUdevpropTy = getOrInsertType(module, "struct.CUdevprop", structType);
}

void CudaDriver::setCUDA_MEMCPY2DTy(Module &module) {
  //%struct.CUDA_MEMCPY2D = type { i32, i32, i32, i8*,
  //                               i32, %struct.CUarray_st*, i32,
  //                               i32, i32, i32, i8*,
  //                               i32, %struct.CUarray_st*, i32,
  //                               i32, i32 }
  std::vector< Type * > params(16, i32Ty);
  params[3] = params[10] = i8PtrTy;
  params[5] = params[12] = getPtrTy(CUarrayTy);
  Type *structType = StructType::get(getGlobalContext(), params);
  CUDA_MEMCPY2DTy = getOrInsertType(module, "struct.CUDA_MEMCPY2D", structType);
}

void CudaDriver::setCUDA_MEMCPY3DTy(Module &module) {
  //%struct.CUDA_MEMCPY3D = type { i32, i32, i32, i32, i32,
  //                               i8*, i32, %struct.CUarray_st*, i8*, i32, i32,
  //                               i32, i32, i32, i32, i32,
  //                               i8*, i32, %struct.CUarray_st*, i8*,
  //                               i32, i32, i32, i32, i32 }
  std::vector< Type * > params(25, i32Ty);
  params[3] = params[8] = params[16]= params[19] = i8PtrTy;
  params[7] = params[18] = getPtrTy(CUarrayTy);
  Type *structType = StructType::get(getGlobalContext(), params);
  CUDA_MEMCPY3DTy = getOrInsertType(module, "struct.CUDA_MEMCPY3D", structType);
}

Type *CudaDriver::getCUdeviceTy() {
  return CUdeviceTy;
}

Type *CudaDriver::getCUcontextTy() {
  return CUcontextTy;
}

Type *CudaDriver::getCUmoduleTy() {
  return CUmoduleTy;
}

Type *CudaDriver::getCUfunctionTy() {
  return CUfunctionTy;
}

Type *CudaDriver::getCUdeviceptrTy() {
  return CUdeviceptrTy;
}

Value *CudaDriver::insertCheckedCall(StringRef name,
                                Value **begin,
                                Value **end) {
  if(_debug) {
    Value *a = _module.getGlobalVariable(name);
    if(a == NULL) a = getBuilder().CreateGlobalStringPtr(name.data(), name);
    Value *b = insertCall(name, begin, end);
    Value *args[] = { a, b };
    return insertCall("CheckCudaError", args, args + 2);
  } else {
    return insertCall(name, begin, end);
  }
}

void CudaDriver::create(Function *func, GlobalValue *ptxAsm) {
  setInsertPoint(&func->getEntryBlock());

  // Load module, add memcpy's, and launch kernel.
  insertModuleLoadData(cuModule,
                       _builder.CreateConstInBoundsGEP2_32(ptxAsm, 0, 0));

  // Create variable of type CUFunctionTy.
  Value *cuFunction = _builder.CreateAlloca(getPtrTy(getCUfunctionTy()),
                                             0,
                                             "cuFunction");

  // Get function handle.
  Value *kernelName = _builder.CreateGlobalStringPtr(func->getName().str().c_str(), func->getName());
  insertModuleGetFunction(cuFunction, _builder.CreateLoad(cuModule), kernelName);

  int offset = 0;
  std::vector< Memcpy > memcpyList;

  typedef llvm::Function::arg_iterator FuncArgIterator;
  FuncArgIterator arg = func->arg_begin(), end = func->arg_end();
  for( ; arg != end; ++arg) {
    Type *type = arg->getType();
    if(type->isPointerTy()) {
      Value *size = ConstantInt::get(i64Ty, getSizeInBytes(type));
      Value *d_arg = _builder.CreateAlloca(getCUdeviceptrTy(), 0, "d_" + arg->getName());

      // Allocate memory for variable on GPU.
      insertMemAlloc(d_arg, size);

      // Copy variable from CPU to GPU.
      insertMemcpyHtoD(_builder.CreateLoad(d_arg),
                       _builder.CreateBitCast(arg, i8PtrTy),
                       size);

      // Add variable to list of memcpy's.
      memcpyList.push_back(Memcpy(arg, d_arg, size));

      // Set pointer variable as parameter to kernel.
      insertParamSetv(_builder.CreateLoad(cuFunction),
                      ConstantInt::get(i32Ty, offset),
                      _builder.CreateBitCast(d_arg, i8PtrTy),
                      ConstantInt::get(i32Ty, 8));
      offset += 8;
    } else {
      assert(false && "Unsupported type for cuParamSet*.");
    }
  }

  insertParamSetSize(_builder.CreateLoad(cuFunction),
                     ConstantInt::get(i32Ty, offset));

  // Set block-dimensions for kernel.
  insertFuncSetBlockShape(_builder.CreateLoad(cuFunction),
                          _blockSize[0],
                          _blockSize[1],
                          _blockSize[2]);

  // Launch kernel.
  insertLaunchGrid(_builder.CreateLoad(cuFunction),
                   _gridSize[0],
                   _gridSize[1]);

  for(unsigned i = 0, e = memcpyList.size(); i < e; ++i) {
    // Copy results from GPU to CPU.
    insertMemcpyDtoH(_builder.CreateBitCast(memcpyList[i].host, i8PtrTy),
                     _builder.CreateLoad(memcpyList[i].device),
                     memcpyList[i].size);

    // Free GPU memory.
    insertMemFree(_builder.CreateLoad(memcpyList[i].device));
  }
}

void CudaDriver::initialize() {
  StringRef name = "cudaCreate";
  if(Function *func = _module.getFunction(name)) return;

  std::vector< Type * > types;
  Function *init = Function::Create(FunctionType::get(voidTy, types, false),
                                    GlobalValue::ExternalLinkage,
                                    name,
                                    &_module);

  cuDevice  = new GlobalVariable(_module, getCUdeviceTy(),
                                 false, GlobalValue::PrivateLinkage,
                                 ConstantInt::get(i32Ty, 0), "cuDevice");

  cuContext = new GlobalVariable(_module,
                                 getPtrTy(getCUcontextTy()),
                                 false,
                                 GlobalValue::PrivateLinkage,
                                 Constant::getNullValue(PointerType::get(getCUcontextTy(), 0)),
                                 "cuContext");

  cuModule = new GlobalVariable(_module,
                                getPtrTy(getCUmoduleTy()),
                                false,
                                GlobalValue::PrivateLinkage,
                                Constant::getNullValue(PointerType::get(getCUmoduleTy(), 0)),
                                "cuModule");

  BasicBlock *entryBB = BasicBlock::Create(_module.getContext(), "entry", init);
  _builder.SetInsertPoint(entryBB);

  // Initialize CUDA.
  insertInit();

  // Get handle for device 0.
  insertDeviceGet(cuDevice, ConstantInt::get(i32Ty, 0));

  // Create context.
  insertCtxCreate(cuContext, ConstantInt::get(i32Ty, 0),
                  _builder.CreateLoad(cuDevice));

  _builder.CreateRetVoid();

  // Insert Cuda initialization at the start of main().
  Function *func = _module.getFunction("main");
  assert(func != NULL && "Could not find main()!\n");
  BasicBlock *bb = func->begin();
  _builder.SetInsertPoint(bb, bb->begin());
  _builder.CreateCall(init);

  // Generate finalization for CUDA.
  finalize();
}

void CudaDriver::finalize() {
  StringRef name = "cudaDestroy";
  if(_module.getFunction(name)) return;

  std::vector< Type * > types;
  Function *fin = Function::Create(FunctionType::get(voidTy, types, false),
                                   GlobalValue::ExternalLinkage,
                                   name,
                                   &_module);

  BasicBlock *entryBB = BasicBlock::Create(_module.getContext(), "entry", fin);
  _builder.SetInsertPoint(entryBB);

  // Destroy context.
  insertCtxDestroy(_builder.CreateLoad(cuContext));

  _builder.CreateRetVoid();

  // Insert Cuda finalization at the end of main().
  Function *func = _module.getFunction("main");
  assert(func != NULL && "Could not find main()!\n");

  typedef llvm::Function::iterator BasicBlockIterator;
  BasicBlockIterator bb = func->begin(), bb_end = func->end();
  for( ; bb != bb_end; ++bb) {
    if(isa< ReturnInst >(bb->getTerminator())) break;
  }
  _builder.SetInsertPoint(bb, bb->getTerminator());
  _builder.CreateCall(fin);
}

void CudaDriver::destroy() {
  if(_module.getFunction("cudaCreate") == NULL) return;

  Function *fin = _module.getFunction("cudaDestroy");
  Instruction *insertBefore = _builder.GetInsertBlock()->getTerminator();
  CallInst::Create(fin, "", insertBefore);
}

Value *CudaDriver::insertInit() {
  // The single parameter to cuInit() must be 0.
  Value *args[] = { ConstantInt::get(i32Ty, 0) };
  return insertCheckedCall("cuInit", args, args + 1);
}

Value *CudaDriver::insertDeviceComputeCapability(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceComputeCapability", args, args + 2);
}

Value *CudaDriver::insertDeviceGet(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGet", args, args + 2);
}

Value *CudaDriver::insertDeviceGetAttribute(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuDeviceGetAttribute", args, args + 3);
}

Value *CudaDriver::insertDeviceGetCount(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuDeviceGetCount", args, args + 1);
}

Value *CudaDriver::insertDeviceGetName(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGetName", args, args + 2);
}

Value *CudaDriver::insertDeviceGetProperties(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGetProperties", args, args + 2);
}

Value *CudaDriver::insertDeviceTotalMem(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceTotalMem", args, args + 2);
}

Value *CudaDriver::insertDriverGetVersion(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuDriverGetVersion", args, args + 1);
}

Value *CudaDriver::insertCtxAttach(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuXtxAttach", args, args + 2);
}

Value *CudaDriver::insertCtxCreate(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuCtxCreate_v2", args, args + 3);
}

Value *CudaDriver::insertCtxDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxDestroy", args, args + 1);
}

Value *CudaDriver::insertCtxDetach(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxDetach", args, args + 1);
}

Value *CudaDriver::insertCtxGetDevice(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxGetDevice", args, args + 1);
}

Value *CudaDriver::insertCtxPopCurrent(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxPopCurrent", args, args + 1);
}

Value *CudaDriver::insertCtxPushCurrent(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxPushCurrent", args, args + 1);
}

Value *CudaDriver::insertCtxSynchronize() {
  Value *args[] = { };
  return insertCheckedCall("cuCtxSynchronize", args, args);
}

Value *CudaDriver::insertModuleGetFunction(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuModuleGetFunction", args, args + 3);
}

Value *CudaDriver::insertModuleGetGlobal(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuModuleGetGlobal", args, args + 4);
}

Value *CudaDriver::insertModuleGetTexRef(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuModuleGetTexRef", args, args + 3);
}

Value *CudaDriver::insertModuleLoad(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoad", args, args + 2);
}

Value *CudaDriver::insertModuleLoadData(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoadData", args, args + 2);
}

Value *CudaDriver::insertModuleLoadDataEx(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuModuleLoadDataEx", args, args + 5);
}

Value *CudaDriver::insertModuleLoadFatBinary(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoadFatBinary", args, args + 2);
}

Value *CudaDriver::insertModuleUnload(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuModuleUnload", args, args + 1);
}

Value *CudaDriver::insertStreamCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuStreamCreate", args, args + 2);
}

Value *CudaDriver::insertStreamDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamDestroy", args, args + 1);
}

Value *CudaDriver::insertStreamQuery(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamQuery", args, args + 1);
}

Value *CudaDriver::insertStreamSynchronize(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamSynchronize", args, args + 1);
}

Value *CudaDriver::insertEventCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuEventCreate", args, args + 2);
}

Value *CudaDriver::insertEventDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventDestroy", args, args + 1);
}

Value *CudaDriver::insertEventElapsedTime(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuEventElapsedTime", args, args + 3);
}

Value *CudaDriver::insertEventQuery(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventQuery", args, args + 1);
}

Value *CudaDriver::insertEventRecord(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuEventRecord", args, args + 2);
}

Value *CudaDriver::insertEventSynchronize(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventSynchronize", args, args + 1);
}

Value *CudaDriver::insertFuncGetAttribute(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuFuncGetAttribute", args, args + 3);
}

Value *CudaDriver::insertFuncSetBlockShape(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuFuncSetBlockShape", args, args + 4);
}

Value *CudaDriver::insertFuncSetSharedSize(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuFuncSetSharedSize", args, args + 2);
}

Value *CudaDriver::insertLaunch(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuLaunch", args, args + 1);
}

Value *CudaDriver::insertLaunchGrid(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuLaunchGrid", args, args + 3);
}

Value *CudaDriver::insertLaunchGridAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuLaunchGridAsync", args, args + 4);
}

Value *CudaDriver::insertParamSetf(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSetf", args, args + 3);
}

Value *CudaDriver::insertParamSeti(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSeti", args, args + 3);
}

Value *CudaDriver::insertParamSetSize(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuParamSetSize", args, args + 2);
}

Value *CudaDriver::insertParamSetTexRef(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSetTexRef", args, args + 3);
}

Value *CudaDriver::insertParamSetv(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuParamSetv", args, args + 4);
}

Value *CudaDriver::insertArray3DCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArray3DCreate", args, args + 2);
}

Value *CudaDriver::insertArray3DGetDescriptor(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArray3DGetDescriptor", args, args + 2);
}

Value *CudaDriver::insertArrayCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArrayCreate", args, args + 2);
}

Value *CudaDriver::insertArrayDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuArrayDestroy", args, args + 1);
}

Value *CudaDriver::insertArrayGetDescriptor(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArrayGetDescriptor", args, args + 2);
}

Value *CudaDriver::insertMemAlloc(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemAlloc_v2", args, args + 2);
}

Value *CudaDriver::insertMemAllocHost(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemAllocHost", args, args + 2);
}

Value *CudaDriver::insertMemAllocPitch(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemAllocPitch", args, args + 5);
}

Value *CudaDriver::insertMemcpy2D(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy2D", args, args + 1);
}

Value *CudaDriver::insertMemcpy2DAsync(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemcpy2DAsync", args, args + 2);
}

Value *CudaDriver::insertMemcpy2DUnaligned(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy2DUnaligned", args, args + 1);
}

Value *CudaDriver::insertMemcpy3D(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy3D", args, args + 1);
}

Value *CudaDriver::insertMemcpy3DAsync(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemcpy3DAsync", args, args + 2);
}

Value *CudaDriver::insertMemcpyAtoA(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyAtoA", args, args + 5);
}

Value *CudaDriver::insertMemcpyAtoD(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyAtoD", args, args + 4);
}

Value *CudaDriver::insertMemcpyAtoH(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyAtoH", args, args + 4);
}

Value *CudaDriver::insertMemcpyAtoHAsync(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyAtoHAsync", args, args + 5);
}

Value *CudaDriver::insertMemcpyDtoA(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyDtoA", args, args + 4);
}

Value *CudaDriver::insertMemcpyDtoD(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyDtoD", args, args + 3);
}

Value *CudaDriver::insertMemcpyDtoH(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyDtoH_v2", args, args + 3);
}

Value *CudaDriver::insertMemcpyDtoHAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyDtoHAsync", args, args + 4);
}

Value *CudaDriver::insertMemcpyHtoA(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyHtoA", args, args + 4);
}

Value *CudaDriver::insertMemcpyHtoAAsync(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyHtoAAsync", args, args + 5);
}

Value *CudaDriver::insertMemcpyHtoD(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyHtoD_v2", args, args + 3);
}

Value *CudaDriver::insertMemcpyHtoDAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyHtoDAsync", args, args + 4);
}

Value *CudaDriver::insertMemFree(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemFree_v2", args, args + 1);
}

Value *CudaDriver::insertMemFreeHost(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemFreeHost", args, args + 1);
}

Value *CudaDriver::insertMemGetAddressRange(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemGetAddressRange", args, args + 3);
}

Value *CudaDriver::insertMemGetInfo(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemGetInfo", args, args + 2);
}

Value *CudaDriver::insertMemHostAlloc(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemHostAlloc", args, args + 3);
}

Value *CudaDriver::insertMemHostGetDevicePointer(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemHostGetDevicePointer", args, args + 3);
}

Value *CudaDriver::insertMemHostGetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemHostGetFlags", args, args + 2);
}

Value *CudaDriver::insertMemsetD16(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD16", args, args + 3);
}

Value *CudaDriver::insertMemsetD2D16(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D16", args, args + 5);
}

Value *CudaDriver::insertMemsetD2D32(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D32", args, args + 5);
}

Value *CudaDriver::insertMemsetD2D8(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D8", args, args + 5);
}

Value *CudaDriver::insertMemsetD32(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD32", args, args + 3);
}

Value *CudaDriver::insertMemsetD8(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD8", args, args + 3);
}

Value *CudaDriver::insertTexRefCreate(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuTexRefCreate", args, args + 1);
}

Value *CudaDriver::insertTexRefDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuTexRefDestroy", args, args + 1);
}

Value *CudaDriver::insertTexRefGetAddress(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetAddress", args, args + 2);
}

Value *CudaDriver::insertTexRefGetAddressMode(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefGetAddressMode", args, args + 3);
}

Value *CudaDriver::insertTexRefGetArray(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetArray", args, args + 2);
}

Value *CudaDriver::insertTexRefGetFilterMode(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetFilterMode", args, args + 2);
}

Value *CudaDriver::insertTexRefGetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetFlags", args, args + 2);
}

Value *CudaDriver::insertTexRefGetFormat(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefGetFormat", args, args + 3);
}

Value *CudaDriver::insertTexRefSetAddress(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuTexRefSetAddress", args, args + 4);
}

Value *CudaDriver::insertTexRefSetAddress2D(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuTexRefSetAddress2D", args, args + 4);
}

Value *CudaDriver::insertTexRefSetAddressMode(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetAddressMode", args, args + 3);
}

Value *CudaDriver::insertTexRefSetArray(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetArray", args, args + 3);
}

Value *CudaDriver::insertTexRefSetFilterMode(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefSetFilterMode", args, args + 2);
}

Value *CudaDriver::insertTexRefSetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefSetFlags", args, args + 2);
}

Value *CudaDriver::insertTexRefSetFormat(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetFormat", args, args + 3);
}
