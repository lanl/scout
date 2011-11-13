#ifndef SCOUT_GPU_H_
#define SCOUT_GPU_H_

#include "runtime/opengl/opengl.h"

#include <cuda.h>
#include <cudaGL.h>

extern bool _scout_gpu;

extern CUdevice _scout_device;
extern CUcontext _scout_device_context;
extern CUgraphicsResource _scout_device_resource;
extern CUstream _scout_device_stream;
extern CUdeviceptr _scout_device_pixels;

void __sc_init_cuda();

#endif // SCOUT_GPU_H_
