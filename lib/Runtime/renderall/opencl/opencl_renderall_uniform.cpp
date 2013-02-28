#include <cassert>
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

#include "scout/Runtime/opencl/scout_opencl.h"
#include "scout/Runtime/renderall/renderall_uniform_.h"

namespace scout{
  void renderall_uniform_rt_::register_pbo(GLuint pbo){
    cl_int ret;
    if(__sc_opencl) {
      __sc_opencl_device_renderall_uniform_colors =
          clCreateFromGLBuffer(__sc_opencl_context,
                           CL_MEM_WRITE_ONLY,
                           pbo,
                           &ret);
      assert(ret == CL_SUCCESS);
    }
  }
}

