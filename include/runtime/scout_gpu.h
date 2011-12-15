#ifndef SCOUT_GPU_H_
#define SCOUT_GPU_H_

#include "runtime/opengl/opengl.h"

#include <cuda.h>
#include <cudaGL.h>

extern bool __sc_gpu;

extern CUdevice __sc_device;
extern CUcontext __sc_device_context;
extern CUgraphicsResource __sc_device_resource;
extern CUstream __sc_device_stream;

void __sc_init_cuda();

void __sc_register_gpu_pbo(GLuint pbo, unsigned int flags);

#endif // SCOUT_GPU_H_
