#include <cassert>

#include "runtime/opengl/opengl.h"

#include "runtime/scout_gpu.h"

using namespace std;

bool __sc_gpu = false;

CUdevice __sc_device;
CUcontext __sc_device_context;
CUgraphicsResource __sc_device_resource;
CUstream __sc_device_stream;

void __sc_init_cuda() {
  __sc_gpu = true;

  // Initialize CUDA Driver API.
  assert(cuInit(0) == CUDA_SUCCESS);

  // Acquire a GPU device.
  assert(cuDeviceGet(&__sc_device, 0) == CUDA_SUCCESS);

  // Create a CUDA context for interoperability with OpenGL.
  assert(cuGLCtxCreate(&__sc_device_context, 0, __sc_device) ==
	 CUDA_SUCCESS);
}

void __sc_register_gpu_pbo(GLuint pbo, unsigned int flags){
  assert(cuGraphicsGLRegisterBuffer(&__sc_device_resource, pbo, flags) ==
	 CUDA_SUCCESS);
}
