#ifndef SCOUT_CUDA_H_
#define SCOUT_CUDA_H_

#ifdef SC_ENABLE_OPENGL 
#include "scout/Runtime/opengl/opengl.h"
#endif

#include <cuda.h>
#include <cudaGL.h>

extern bool __sc_cuda;

extern CUdevice __sc_cuda_device;
extern CUcontext __sc_cuda_device_context;
extern CUgraphicsResource __sc_cuda_device_resource;
extern CUstream __sc_cuda_device_stream;

void __sc_init_cuda();

void __sc_register_cuda_pbo(GLuint pbo, unsigned int flags);

#endif // SCOUT_CUDA_H_
