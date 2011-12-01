#include <cassert>

#include "runtime/opengl/opengl.h"

#include "runtime/scout_gpu.h"

using namespace std;

bool _scout_gpu = false;

CUdevice _scout_device;
CUcontext _scout_device_context;
CUgraphicsResource _scout_device_resource;
CUstream _scout_device_stream;
CUdeviceptr _scout_device_pixels;

void __sc_init_cuda() {
  _scout_gpu = true;

  // Initialize CUDA Driver API.
  assert(cuInit(0) == CUDA_SUCCESS);

  // Acquire a GPU device.
  assert(cuDeviceGet(&_scout_device, 0) == CUDA_SUCCESS);

  // Create a CUDA context for interoperability with OpenGL.
  assert(cuGLCtxCreate(&_scout_device_context, 0, _scout_device) ==
	 CUDA_SUCCESS);
}