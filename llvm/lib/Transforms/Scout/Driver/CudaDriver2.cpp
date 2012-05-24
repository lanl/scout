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

#include "llvm/Transforms/Scout/Driver/CudaDriver2.h"

#include <iostream>

using namespace llvm;

CudaDriver2::CudaDriver2(Module &module, IRBuilder<> &builder, bool debug)
  : Driver(module, builder, debug),
    CUaddress_modeTy(i32Ty),
    CUarrayTy(getOrInsertType(module, "struct.CUarray_st")),
    CUarray_formatTy(i32Ty),
    CUcontextTy(getOrInsertType(module, "struct.CUctx_st")),
    CUdeviceTy(i32Ty),
    CUdeviceptrTy(IntegerType::get(module.getContext(), sizeof(uintptr_t) * 8)),
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
    _blockSize(SmallVector< Constant *, 3 >(3, ConstantInt::get(i32Ty, 1))),
    fnArgAttrs(SmallVector< ConstantInt *, 3 >()),
    meshFieldNames(SmallVector< Value *, 3 >()),
    dimensions(SmallVector< ConstantInt *, 3 >())
{
  setCUDA_ARRAY_DESCRIPTORTy(module);
  setCUDA_ARRAY3D_DESCRIPTORTy(module);
  setCUdevpropTy(module);
  setCUDA_MEMCPY2DTy(module);
  setCUDA_MEMCPY3DTy(module);

#include "llvm/Transforms/Scout/Driver/CudaDriverDeclarations.h"
}

void CudaDriver2::setGridSize(SmallVector< Constant *, 3 > &size) {
  _gridSize = size;
}

void CudaDriver2::setBlockSize(SmallVector< Constant *, 3 > &size) {
  _blockSize = size;
}

void CudaDriver2::setCUDA_ARRAY_DESCRIPTORTy(Module &module) {
  //%struct.CUDA_ARRAY_DESCRIPTOR = type { i32, i32, i32, i32 }
  std::vector< Type * > params(4, i32Ty);
  Type *structType = StructType::get(module.getContext(), params);
  CUDA_ARRAY_DESCRIPTORTy = getOrInsertType(module,
                                            "struct.CUDA_ARRAY_DESCRIPTOR",
                                            structType);
}

void CudaDriver2::setCUDA_ARRAY3D_DESCRIPTORTy(Module &module) {
  //%struct.CUDA_ARRAY3D_DESCRIPTOR = type { i32, i32, i32, i32, i32, i32 }
  std::vector< Type * > params(6, i32Ty);
  Type *structType = StructType::get(module.getContext(), params);
  CUDA_ARRAY3D_DESCRIPTORTy = getOrInsertType(module,
                                          "struct.CUDA_ARRAY3D_DESCRIPTOR",
                                          structType);
}

void CudaDriver2::setCUdevpropTy(Module &module) {
  //%struct.CUdevprop = type { i32, [3 x i32], [3 x i32], i32, i32, i32, i32, i32, i32, i32 }
  std::vector< Type * > params(10, i32Ty);
  params[1] = params[2] = ArrayType::get(i32Ty, 3);
  Type *structType = StructType::get(module.getContext(), params);
  CUdevpropTy = getOrInsertType(module, "struct.CUdevprop", structType);
}

void CudaDriver2::setCUDA_MEMCPY2DTy(Module &module) {
  //%struct.CUDA_MEMCPY2D = type { i32, i32, i32, i8*,
  //                               i32, %struct.CUarray_st*, i32,
  //                               i32, i32, i32, i8*,
  //                               i32, %struct.CUarray_st*, i32,
  //                               i32, i32 }
  std::vector< Type * > params(16, i32Ty);
  params[3] = params[10] = i8PtrTy;
  params[5] = params[12] = getPtrTy(CUarrayTy);
  Type *structType = StructType::get(module.getContext(), params);
  CUDA_MEMCPY2DTy = getOrInsertType(module, "struct.CUDA_MEMCPY2D", structType);
}

void CudaDriver2::setCUDA_MEMCPY3DTy(Module &module) {
  //%struct.CUDA_MEMCPY3D = type { i32, i32, i32, i32, i32,
  //                               i8*, i32, %struct.CUarray_st*, i8*, i32, i32,
  //                               i32, i32, i32, i32, i32,
  //                               i8*, i32, %struct.CUarray_st*, i8*,
  //                               i32, i32, i32, i32, i32 }
  std::vector< Type * > params(25, i32Ty);
  params[3] = params[8] = params[16]= params[19] = i8PtrTy;
  params[7] = params[18] = getPtrTy(CUarrayTy);
  Type *structType = StructType::get(module.getContext(), params);
  CUDA_MEMCPY3DTy = getOrInsertType(module, "struct.CUDA_MEMCPY3D", structType);
}

Type *CudaDriver2::getCUdeviceTy() {
  return CUdeviceTy;
}

Type *CudaDriver2::getCUcontextTy() {
  return CUcontextTy;
}

Type *CudaDriver2::getCUmoduleTy() {
  return CUmoduleTy;
}

Type *CudaDriver2::getCUfunctionTy() {
  return CUfunctionTy;
}

Type *CudaDriver2::getCUdeviceptrTy() {
  return CUdeviceptrTy;
}

Value *CudaDriver2::insertCheckedCall(StringRef name,
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

Value *CudaDriver2::insertCheckedCall(StringRef name,
                                     ArrayRef< Value * > args) {
 if(_debug) {
    Value *a = _module.getGlobalVariable(name);
    if(a == NULL) a = getBuilder().CreateGlobalStringPtr(name.data(), name);
    Value *b = insertCall(name, args);
    Value *args[] = { a, b };
    return insertCall("CheckCudaError", ArrayRef< Value * >(args));
  } else {
   return insertCall(name, args);
  }
}

bool CudaDriver2::setGridAndBlockSizes() {
  _blockSize[0] = llvm::ConstantInt::get(i32Ty, 16);
  _blockSize[1] = llvm::ConstantInt::get(i32Ty, 16);

  if(getLinearizedMeshSize() < 256) {
    return false;
  }

  int gridDims = getLinearizedMeshSize() / 256;

  if(gridDims < 65535)
    if(getLinearizedMeshSize() % 256 == 0) {
      _gridSize[0] = llvm::ConstantInt::get(i32Ty, gridDims);
      return true;
    } else {
      _gridSize[0] = llvm::ConstantInt::get(i32Ty, gridDims + 1);
      return false;
    }

  int half = gridDims / 2;
  if(gridDims % 2 == 0) {
    _gridSize[0] = _gridSize[1] = llvm::ConstantInt::get(i32Ty, half);
    return true;
  } else {
    _gridSize[0] = llvm::ConstantInt::get(i32Ty, half);
    _gridSize[1] = llvm::ConstantInt::get(i32Ty, half + 1);
    return false;
  }
}

void CudaDriver2::create(Function *func,
			GlobalValue *ptxAsm,
			Value* meshName,
			MDNode* mdn) {
  setInsertPoint(&func->getEntryBlock());

  // Calculate the total number of mesh elements.
  int meshSize = getLinearizedMeshSize();

  Value* image = _builder.CreateConstInBoundsGEP2_32(ptxAsm, 0, 0);

  // Load module, add memcpy's, and launch kernel.
  llvm::Value *cuModule = _builder.CreateAlloca(getPtrTy(getCUmoduleTy()), 0, "cuModule");

  //insertModuleLoadData(cuModule, image);

  insertScoutGetModule(cuModule, image);

  Value *colors;
  if(func->getName().startswith("renderall")) {
    if(!_module.getNamedGlobal("__sc_device_renderall_uniform_colors")) {
      colors = new GlobalVariable(_module,
                                  getCUdeviceptrTy(),
                                  false,
                                  GlobalValue::ExternalLinkage,
                                  0,
                                  "__sc_device_renderall_uniform_colors");

    }
  }

  // Create variable of type CUFunctionTy.
  Value *cuFunction = _builder.CreateAlloca(getPtrTy(getCUfunctionTy()),
                                             0,
                                             "cuFunction");

  // Get function handle.
  Value *kernelName = _builder.CreateGlobalStringPtr(func->getName().str().c_str(), func->getName());
  insertModuleGetFunction(cuFunction, _builder.CreateLoad(cuModule), kernelName);

  int offset = 0;
  llvm::SmallVector< Memcpy, 3 > memcpyList;

  typedef llvm::Function::arg_iterator FuncArgIterator;
  FuncArgIterator arg = func->arg_begin(), end = func->arg_end();
  for(unsigned i = 0; arg != end; ++arg, ++i) {
    Type *type = arg->getType();

    if(type->isPointerTy()) {

      Value *d_arg;
      if(arg->getName().startswith("colors")){
        d_arg = colors;
      }
      else {
        int numElements = getSizeInBytes(type);

        if(isMeshMember(i))
          numElements *= meshSize;

        Value *size = ConstantInt::get(i64Ty, numElements);
        d_arg = _builder.CreateAlloca(getCUdeviceptrTy(), 0, "d_" + arg->getName());
	
	Value* name =
	  _builder.CreateGlobalStringPtr(arg->getName());

	Value* args[] = {meshName, meshFieldName(i)}; 
	Value* dp = insertCall("__sc_get_gpu_device_ptr", args, args+2);
	  
	Value* np = _builder.CreateIsNull(dp);
	  
	BasicBlock* tb = BasicBlock::Create(_module.getContext(), "then");
	BasicBlock* eb = BasicBlock::Create(_module.getContext(), "else");
	BasicBlock* mb = BasicBlock::Create(_module.getContext(), "merge");
	  
	func->getBasicBlockList().push_back(tb);
	func->getBasicBlockList().push_back(eb);
	func->getBasicBlockList().push_back(mb);
	  
	_builder.CreateCondBr(np, tb, eb);
	_builder.SetInsertPoint(tb);

	// Allocate memory for variable on GPU.
	insertMemAlloc(d_arg, size);

	Value* ld = _builder.CreateLoad(d_arg);
	  
	// Copy variable from CPU to GPU.
	insertMemcpyHtoD(ld,
			 _builder.CreateBitCast(arg, i8PtrTy),
			 size);

        memcpyList.push_back(Memcpy(arg, d_arg, size));
	  
	Value* args2[] = {meshName, meshFieldName(i), ld}; 
	insertCall("__sc_put_gpu_device_ptr", args2, args2+3);
	  
	_builder.CreateBr(mb);
	  
	_builder.SetInsertPoint(eb);

	_builder.CreateStore(dp, d_arg);

	if(!isMeshMember(i)){
	  unsigned numOperands = mdn->getNumOperands();
	  bool found = false;
	  for(unsigned i = 0; i < numOperands; ++i){
	    Value* v = mdn->getOperand(i);
	    std::string s = v->getName().str();
	    std::string a = arg->getName().str();

	    if(s == a){
	      found = true;
	      break;
	    }
	  }
	  
	  if(found){
	    Value* ld = _builder.CreateLoad(d_arg);
	    insertMemcpyHtoD(ld,
			     _builder.CreateBitCast(arg, i8PtrTy),
			     size);
	  }
	}
	
	_builder.CreateBr(mb);
	_builder.SetInsertPoint(mb);
      }

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

  if(_debug) insertCtxSynchronize();

  // normally, we do not want to copy memory back to the host
  // but, for debugging purposes, it can be helpful to uncomment
  // the following
  /*
  for(unsigned i = 0, e = memcpyList.size(); i < e; ++i) {
    // Copy results from GPU to CPU.
    insertMemcpyDtoH(_builder.CreateBitCast(memcpyList[i].host, i8PtrTy),
                     _builder.CreateLoad(memcpyList[i].device),
                     memcpyList[i].size);

    // Free GPU memory.
    //insertMemFree(_builder.CreateLoad(memcpyList[i].device));
  }
  */

  // Unload cuda module.
  //insertModuleUnload(_builder.CreateLoad(cuModule));

  _builder.CreateRetVoid();
}

void CudaDriver2::initialize() {
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

void CudaDriver2::finalize() {
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

void CudaDriver2::destroy() {
  if(_module.getFunction("cudaCreate") == NULL) return;

  Function *fin = _module.getFunction("cudaDestroy");
  Instruction *insertBefore = _builder.GetInsertBlock()->getTerminator();
  CallInst::Create(fin, "", insertBefore);
}

Value *CudaDriver2::insertInit() {
  // The single parameter to cuInit() must be 0.
  Value *args[] = { ConstantInt::get(i32Ty, 0) };
  return insertCheckedCall("cuInit", ArrayRef< Value * >(args));
}

void CudaDriver2::setFnArgAttributes(SmallVector< llvm::ConstantInt *, 3 > args) {
  fnArgAttrs = args;
}

void CudaDriver2::setMeshFieldNames(SmallVector< llvm::Value *, 3 > args) {
  meshFieldNames = args;
}

llvm::Value *CudaDriver2::getDimension(int dim) {
  return dimensions[dim];
}

void CudaDriver2::setDimensions(SmallVector< llvm::ConstantInt *, 3 > args) {
  dimensions = args;
}

int CudaDriver2::getLinearizedMeshSize() {
  int sum = 1;
  for(unsigned i = 1, e = dimensions.size(); i < e; i+=2) {
    sum *= dimensions[i]->getSExtValue();
  }
  return sum;
}

bool CudaDriver2::isMeshMember(unsigned i) {
  return fnArgAttrs[i]->getSExtValue();
}

Value* CudaDriver2::meshFieldName(unsigned i) {
  return meshFieldNames[i];
}

Value *CudaDriver2::insertDeviceComputeCapability(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceComputeCapability", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceGet(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGet", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceGetAttribute(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuDeviceGetAttribute", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceGetCount(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuDeviceGetCount", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceGetName(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGetName", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceGetProperties(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceGetProperties", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDeviceTotalMem(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuDeviceTotalMem", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertDriverGetVersion(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuDriverGetVersion", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxAttach(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuXtxAttach", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxCreate(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuCtxCreate_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxDestroy", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxDetach(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxDetach", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxGetDevice(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxGetDevice", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxPopCurrent(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxPopCurrent", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxPushCurrent(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuCtxPushCurrent", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertCtxSynchronize() {
  return insertCheckedCall("cuCtxSynchronize", ArrayRef< Value * >());
}

Value *CudaDriver2::insertModuleGetFunction(Value *a, Value *b, Value *c) {
  Value *args[3] = { a, b, c };
  return insertCheckedCall("cuModuleGetFunction", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleGetGlobal(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuModuleGetGlobal_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleGetTexRef(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuModuleGetTexRef", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleLoad(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoad", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleLoadData(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoadData", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertScoutGetModule(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("__sc_get_gpu_module", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleLoadDataEx(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuModuleLoadDataEx", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleLoadFatBinary(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuModuleLoadFatBinary", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertModuleUnload(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuModuleUnload", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertStreamCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuStreamCreate", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertStreamDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamDestroy", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertStreamQuery(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamQuery", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertStreamSynchronize(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuStreamSynchronize", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuEventCreate", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventDestroy", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventElapsedTime(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuEventElapsedTime", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventQuery(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventQuery", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventRecord(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuEventRecord", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertEventSynchronize(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuEventSynchronize", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertFuncGetAttribute(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuFuncGetAttribute", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertFuncSetBlockShape(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuFuncSetBlockShape", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertFuncSetSharedSize(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuFuncSetSharedSize", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertLaunch(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuLaunch", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertLaunchGrid(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuLaunchGrid", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertLaunchGridAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuLaunchGridAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertParamSetf(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSetf", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertParamSeti(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSeti", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertParamSetSize(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuParamSetSize", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertParamSetTexRef(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuParamSetTexRef", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertParamSetv(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuParamSetv", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertArray3DCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArray3DCreate", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertArray3DGetDescriptor(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArray3DGetDescriptor", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertArrayCreate(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArrayCreate", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertArrayDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuArrayDestroy", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertArrayGetDescriptor(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuArrayGetDescriptor", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemAlloc(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemAlloc_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemAllocHost(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemAllocHost", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemAllocPitch(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemAllocPitch", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpy2D(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy2D", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpy2DAsync(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemcpy2DAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpy2DUnaligned(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy2DUnaligned", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpy3D(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemcpy3D", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpy3DAsync(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemcpy3DAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyAtoA(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyAtoA", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyAtoD(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyAtoD", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyAtoH(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyAtoH", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyAtoHAsync(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyAtoHAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyDtoA(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyDtoA", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyDtoD(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyDtoD", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyDtoH(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyDtoH_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyDtoHAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyDtoHAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyHtoA(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyHtoA", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyHtoAAsync(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemcpyHtoAAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyHtoD(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemcpyHtoD_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemcpyHtoDAsync(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuMemcpyHtoDAsync", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemFree(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemFree_v2", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemFreeHost(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuMemFreeHost", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemGetAddressRange(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemGetAddressRange", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemGetInfo(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemGetInfo", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemHostAlloc(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemHostAlloc", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemHostGetDevicePointer(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemHostGetDevicePointer", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemHostGetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuMemHostGetFlags", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD16(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD16", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD2D16(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D16", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD2D32(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D32", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD2D8(Value *a, Value *b, Value *c, Value *d, Value *e) {
  Value *args[] = { a, b, c, d, e };
  return insertCheckedCall("cuMemsetD2D8", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD32(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD32", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertMemsetD8(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuMemsetD8", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefCreate(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuTexRefCreate", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefDestroy(Value *a) {
  Value *args[] = { a };
  return insertCheckedCall("cuTexRefDestroy", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetAddress(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetAddress", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetAddressMode(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefGetAddressMode", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetArray(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetArray", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetFilterMode(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetFilterMode", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefGetFlags", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefGetFormat(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefGetFormat", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetAddress(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuTexRefSetAddress", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetAddress2D(Value *a, Value *b, Value *c, Value *d) {
  Value *args[] = { a, b, c, d };
  return insertCheckedCall("cuTexRefSetAddress2D", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetAddressMode(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetAddressMode", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetArray(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetArray", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetFilterMode(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefSetFilterMode", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetFlags(Value *a, Value *b) {
  Value *args[] = { a, b };
  return insertCheckedCall("cuTexRefSetFlags", ArrayRef< Value * >(args));
}

Value *CudaDriver2::insertTexRefSetFormat(Value *a, Value *b, Value *c) {
  Value *args[] = { a, b, c };
  return insertCheckedCall("cuTexRefSetFormat", ArrayRef< Value * >(args));
}
