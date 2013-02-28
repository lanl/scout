/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 * -----
 *
 */

#ifndef SCOUT_RENDERALL_UNIFORM__H_
#define SCOUT_RENDERALL_UNIFORM__H_

#include "scout/Runtime/renderall/renderall_uniform.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"


namespace scout{

  class renderall_uniform_rt_{
    public:
      renderall_uniform_rt_(renderall_uniform_rt* o);

      ~renderall_uniform_rt_();

      void init();

      void begin();

      void end();

      void map_cuda_resources();

      void unmap_cuda_resources();

      void map_opencl_resources();

      void unmap_opencl_resources();

      void register_pbo(GLuint pbo);

      void register_cuda_pbo(GLuint pbo);

      void register_opencl_pbo(GLuint pbo);

      void exec();

    private:
      renderall_uniform_rt* o_;
      glQuadRenderableVA* _renderable;
  };
} // end namespace scout

#endif
