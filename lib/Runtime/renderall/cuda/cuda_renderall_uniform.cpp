#include <iostream>
#include <cassert>
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

#include <cuda.h>
#include <cudaGL.h>
#include "scout/Runtime/cuda/CudaDevice.h" //has __sc_cuda_device_resource 
#include "scout/Runtime/renderall/renderall_uniform_.h"

namespace scout{
  void renderall_uniform_rt_::register_pbo(GLuint pbo) {
    if(__sc_cuda) {
      assert(cuGraphicsGLRegisterBuffer(&__sc_cuda_device_resource,
          pbo,
          CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) == CUDA_SUCCESS);
    } else {
     std::cout << "no cuda" << std::endl;
    }
  }
}

