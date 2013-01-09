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

Type *CUaddress_modePtrTy = getPtrTy(CUaddress_modeTy);

Type *CUarrayPtrTy = getPtrTy(CUarrayTy);
Type *CUarrayDblPtrTy = getPtrTy(CUarrayPtrTy);

Type *CUarray_formatPtrTy = getPtrTy(CUarray_formatTy);

Type *CUDA_ARRAY_DESCRIPTORPtrTy = getPtrTy(CUDA_ARRAY_DESCRIPTORTy);

Type *CUDA_ARRAY3D_DESCRIPTORPtrTy = getPtrTy(CUDA_ARRAY3D_DESCRIPTORTy);

Type *CUcontextPtrTy = getPtrTy(CUcontextTy);
Type *CUcontextDblPtrTy = getPtrTy(CUcontextPtrTy);

Type *CUdevicePtrTy = getPtrTy(CUdeviceTy);

Type *CUdeviceptrPtrTy = getPtrTy(CUdeviceptrTy);

Type *CUdevpropPtrTy = getPtrTy(CUdevpropTy);

Type *CUeventPtrTy = getPtrTy(CUeventTy);
Type *CUeventDblPtrTy = getPtrTy(CUeventPtrTy);

Type *CUfilter_modePtrTy = getPtrTy(CUfilter_modeTy);

Type *CUfunctionPtrTy = getPtrTy(CUfunctionTy);
Type *CUfunctionDblPtrTy = getPtrTy(CUfunctionPtrTy);

Type *CUmodulePtrTy = getPtrTy(CUmoduleTy);
Type *CUmoduleDblPtrTy = getPtrTy(CUmodulePtrTy);

Type *CUDA_MEMCPY2DPtrTy = getPtrTy(CUDA_MEMCPY2DTy);

Type *CUDA_MEMCPY3DPtrTy = getPtrTy(CUDA_MEMCPY3DTy);

Type *CUstreamPtrTy = getPtrTy(CUstreamTy);
Type *CUstreamDblPtrTy = getPtrTy(CUstreamPtrTy);

Type *CUtexrefPtrTy = getPtrTy(CUtexrefTy);
Type *CUtexrefDblPtrTy = getPtrTy(CUtexrefPtrTy);

// Initialization
declareFunction(CUresultTy, "cuInit", i32Ty);

// Device Management
declareFunction(CUresultTy, "cuDeviceComputeCapability", i32PtrTy, i32PtrTy, CUdeviceTy);
declareFunction(CUresultTy, "cuDeviceGet", CUdevicePtrTy, i32Ty);
declareFunction(CUresultTy, "cuDeviceGetAttribute", i32PtrTy, CUdevice_attributeTy, CUdeviceTy);
declareFunction(CUresultTy, "cuDeviceGetCount", i32PtrTy);
declareFunction(CUresultTy, "cuDeviceGetName", i8PtrTy, i32Ty, CUdeviceTy);
declareFunction(CUresultTy, "cuDeviceGetProperties", CUdevpropPtrTy, CUdeviceTy);
declareFunction(CUresultTy, "cuDeviceTotalMem", i32PtrTy, CUdeviceTy);

// Version Management
declareFunction(CUresultTy, "cuDriverGetVersion", i32PtrTy);

// Context Management
declareFunction(CUresultTy, "cuCtxAttach", CUcontextDblPtrTy, i32Ty);
declareFunction(CUresultTy, "cuCtxCreate_v2", CUcontextDblPtrTy, i32Ty, CUdeviceTy);
declareFunction(CUresultTy, "cuCtxDestroy", CUcontextPtrTy);
declareFunction(CUresultTy, "cuCtxDetach", CUcontextPtrTy);
declareFunction(CUresultTy, "cuCtxGetDevice", CUdevicePtrTy);
declareFunction(CUresultTy, "cuCtxPopCurrent", CUcontextDblPtrTy);
declareFunction(CUresultTy, "cuCtxPushCurrent", CUcontextPtrTy);
declareFunction(CUresultTy, "cuCtxSynchronize");

// Module Management
declareFunction(CUresultTy, "cuModuleGetFunction", CUfunctionDblPtrTy, CUmodulePtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleGetGlobal_v2", CUdeviceptrTy, i32PtrTy, CUmodulePtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleGetTexRef", CUtexrefPtrTy, CUmodulePtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleLoad", CUmoduleDblPtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleLoadData", CUmoduleDblPtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleLoadDataEx", CUmodulePtrTy, i8PtrTy, i32Ty, CUjit_optionTy, i8DblPtrTy);
declareFunction(CUresultTy, "cuModuleLoadFatBinary", CUmoduleDblPtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuModuleUnload", CUmodulePtrTy);

// Stream Management
declareFunction(CUresultTy, "cuStreamCreate", CUstreamDblPtrTy, i32Ty);
declareFunction(CUresultTy, "cuStreamDestroy", CUstreamPtrTy);
declareFunction(CUresultTy, "cuStreamQuery", CUstreamPtrTy);
declareFunction(CUresultTy, "cuStreamSynchronize", CUstreamPtrTy);

// Event Management
declareFunction(CUresultTy, "cuEventCreate", CUeventDblPtrTy, i32Ty);
declareFunction(CUresultTy, "cuEventDestroy", CUeventPtrTy);
declareFunction(CUresultTy, "cuEventElapsedTime", fltTy, CUeventPtrTy, CUeventPtrTy);
declareFunction(CUresultTy, "cuEventQuery", CUeventPtrTy);
declareFunction(CUresultTy, "cuEventRecord", CUeventPtrTy, CUstreamPtrTy);
declareFunction(CUresultTy, "cuEventSynchronize", CUeventPtrTy);

// Execution Control
declareFunction(CUresultTy, "cuFuncGetAttribute", i32PtrTy, CUfunction_attributeTy, CUfunctionPtrTy);
declareFunction(CUresultTy, "cuFuncSetBlockShape", CUfunctionPtrTy, i32Ty, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuFuncSetSharedSize", CUfunctionPtrTy, i32Ty);
declareFunction(CUresultTy, "cuLaunch", CUfunctionPtrTy);
declareFunction(CUresultTy, "cuLaunchGrid", CUfunctionPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuLaunchGridAsync", CUfunctionPtrTy, i32Ty, i32Ty, CUstreamPtrTy);
declareFunction(CUresultTy, "cuParamSetf", CUfunctionPtrTy, i32Ty, fltTy);
declareFunction(CUresultTy, "cuParamSeti", CUfunctionPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuParamSetSize", CUfunctionPtrTy, i32Ty);
declareFunction(CUresultTy, "cuParamSetTexRef", CUfunctionPtrTy, i32Ty, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuParamSetv", CUfunctionPtrTy, i32Ty, i8PtrTy, i32Ty);

// Memory Management
declareFunction(CUresultTy, "cuArray3DCreate", CUarrayDblPtrTy, CUDA_ARRAY3D_DESCRIPTORPtrTy);
declareFunction(CUresultTy, "cuArray3DGetDescriptor", CUDA_ARRAY3D_DESCRIPTORPtrTy, CUarrayPtrTy);
declareFunction(CUresultTy, "cuArrayCreate", CUarrayDblPtrTy, CUDA_ARRAY_DESCRIPTORPtrTy);
declareFunction(CUresultTy, "cuArrayDestroy", CUarrayPtrTy);
declareFunction(CUresultTy, "cuArrayGetDescriptor", CUDA_ARRAY_DESCRIPTORPtrTy, CUarrayPtrTy);
declareFunction(CUresultTy, "cuMemAlloc_v2", CUdeviceptrPtrTy, i64Ty);
declareFunction(CUresultTy, "cuMemAllocHost", i32PtrTy, i32Ty);
declareFunction(CUresultTy, "cuMemAllocPitch", CUdeviceptrPtrTy, i32PtrTy, i32Ty, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemcpy2D", CUDA_MEMCPY2DPtrTy);
declareFunction(CUresultTy, "cuMemcpy2DAsync", CUDA_MEMCPY2DPtrTy, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemcpy2DUnaligned", CUDA_MEMCPY2DPtrTy);
declareFunction(CUresultTy, "cuMemcpy3D", CUDA_MEMCPY3DPtrTy);
declareFunction(CUresultTy, "cuMemcpy3DAsync", CUDA_MEMCPY3DPtrTy, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemcpyAtoA", CUarrayPtrTy, i32Ty, CUarrayPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemcpyAtoD", CUdeviceptrTy, CUarrayPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemcpyAtoH", i8PtrTy, CUarrayPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemcpyAtoHAsync", i8PtrTy, CUarrayPtrTy, i32Ty, i32Ty, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemcpyDtoA", CUarrayPtrTy, i32Ty, CUdeviceptrTy, i32Ty);
declareFunction(CUresultTy, "cuMemcpyDtoD", CUdeviceptrTy, CUdeviceptrTy, i32Ty);
declareFunction(CUresultTy, "cuMemcpyDtoH_v2", i8PtrTy, CUdeviceptrTy, i64Ty);
declareFunction(CUresultTy, "cuMemcpyDtoHAsync", i8PtrTy, CUdeviceptrTy, i32Ty, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemcpyHtoA", CUarrayPtrTy, i32Ty, i8PtrTy, i32Ty);
declareFunction(CUresultTy, "cuMemcpyHtoAAsync", CUarrayPtrTy, i32Ty, i8PtrTy, i32Ty, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemcpyHtoD_v2", CUdeviceptrTy, i8PtrTy, i64Ty);
declareFunction(CUresultTy, "cuMemcpyHtoDAsync", CUdeviceptrTy, i8PtrTy, i32Ty, CUstreamPtrTy);
declareFunction(CUresultTy, "cuMemFree_v2", CUdeviceptrTy);
declareFunction(CUresultTy, "cuMemFreeHost", i8PtrTy);
declareFunction(CUresultTy, "cuMemGetAddressRange", CUdeviceptrPtrTy, i32PtrTy, CUdeviceptrTy);
declareFunction(CUresultTy, "cuMemGetInfo", i32PtrTy, i32PtrTy);
declareFunction(CUresultTy, "cuMemHostAlloc", i8DblPtrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemHostGetDevicePoier", CUdeviceptrPtrTy, i8PtrTy, i32Ty);
declareFunction(CUresultTy, "cuMemHostGetFlags", i32PtrTy, i8PtrTy);
declareFunction(CUresultTy, "cuMemsetD16", CUdeviceptrTy, i16Ty, i32Ty);
declareFunction(CUresultTy, "cuMemsetD2D16", CUdeviceptrTy, i32Ty, i16Ty, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemsetD2D32", CUdeviceptrTy, i32Ty, i16Ty, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemsetD2D8", CUdeviceptrTy, i32Ty, i16Ty, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemsetD32", CUdeviceptrTy, i32Ty, i32Ty);
declareFunction(CUresultTy, "cuMemsetD8", CUdeviceptrTy, i8Ty, i32Ty);

// Texture Reference Management
declareFunction(CUresultTy, "cuTexRefCreate", CUtexrefDblPtrTy);
declareFunction(CUresultTy, "cuTexRefDestroy", CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefGetAddress", CUdeviceptrPtrTy, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefGetAddressMode", CUaddress_modePtrTy, CUtexrefPtrTy, i32Ty);
declareFunction(CUresultTy, "cuTexRefGetArray", CUarrayPtrTy, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefGetFilterMode", CUfilter_modePtrTy, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefGetFlags", i32PtrTy, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefGetFormat", CUarray_formatPtrTy, i32PtrTy, CUtexrefPtrTy);
declareFunction(CUresultTy, "cuTexRefSetAddress", i32PtrTy, CUtexrefPtrTy, CUdeviceptrTy, i32Ty);
declareFunction(CUresultTy, "cuTexRefSetAddress2D", CUtexrefPtrTy, CUDA_ARRAY_DESCRIPTORPtrTy, CUdeviceptrTy, i32Ty);
declareFunction(CUresultTy, "cuTexRefSetAddressMode", CUtexrefPtrTy, i32Ty, CUaddress_modeTy);
declareFunction(CUresultTy, "cuTexRefSetArray", CUtexrefPtrTy, CUarrayPtrTy, i32Ty);
declareFunction(CUresultTy, "cuTexRefSetFilterMode", CUtexrefPtrTy, CUfilter_modeTy);
declareFunction(CUresultTy, "cuTexRefSetFlags", CUtexrefPtrTy, i32Ty);
declareFunction(CUresultTy, "cuTexRefSetFormat", CUtexrefPtrTy, CUarray_formatTy, i32Ty);

// Scout-CUDA error checking
declareFunction(voidTy, "CheckCudaError", i8PtrTy, CUresultTy);

// Scout GPU runtime calls
declareFunction(CUresultTy, "__sc_get_cuda_module", CUmoduleDblPtrTy, i8PtrTy);
declareFunction(CUdeviceptrTy, "__sc_get_cuda_device_ptr", i8PtrTy, i8PtrTy);
declareFunction(voidTy, "__sc_put_cuda_device_ptr", i8PtrTy, i8PtrTy, CUdeviceptrTy);
