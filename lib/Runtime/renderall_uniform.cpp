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


#include <iostream>
#include "scout/Runtime/renderall_uniform.h"
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

// scout includes
#include "scout/Config/defs.h"

#ifdef SC_ENABLE_CUDA
#include <cuda.h>
#include <cudaGL.h>
#include "scout/Runtime/cuda/CudaDevice.h"
#endif

#ifdef SC_ENABLE_OPENCL
#include "scout/Runtime/opencl/scout_opencl.h"
#endif

using namespace std;
using namespace scout;

// ------  LLVM - globals accessed by LLVM / CUDA driver

float4* __sc_renderall_uniform_colors;

#ifdef SC_ENABLE_CUDA
CUdeviceptr __sc_cuda_device_renderall_uniform_colors;
#else
static bool __sc_cuda = false;
#endif

#ifdef SC_ENABLE_OPENCL
cl_mem __sc_opencl_device_renderall_uniform_colors;
#else
static bool __sc_opencl = false;
#endif

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

namespace scout{

  class renderall_uniform_rt_{
    public:
      renderall_uniform_rt_(renderall_uniform_rt* o)
        : o_(o){


          if(!__sc_glsdl){
            __sc_init_sdl(__sc_initial_width, __sc_initial_height);
          }

          init();
        }

      ~renderall_uniform_rt_(){
        if (_renderable != NULL) delete _renderable;
      }

      void init(){
        _renderable = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0),
          glfloat3(o_->width(), o_->height(), 0.0));

#if defined(SC_ENABLE_CUDA) || defined(SC_ENABLE_OPENCL)
        if(__sc_cuda){
          register_cuda_pbo(_renderable->get_buffer_object_id());
        }
        else if(__sc_opencl){
          register_opencl_pbo(_renderable->get_buffer_object_id());
        }
#endif

        _renderable->initialize(NULL);

        // show empty buffer
        __sc_glsdl->swapBuffers();
      }

      void begin(){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#if defined(SC_ENABLE_CUDA) || defined(SC_ENABLE_OPENCL)
        if(__sc_cuda){
          map_cuda_resources();
        }
        else if(__sc_opencl){
          map_opencl_resources();
        }
        else{
         __sc_renderall_uniform_colors =_renderable->map_colors();
        }
#else
       __sc_renderall_uniform_colors =_renderable->map_colors(); 
#endif
      }

      void end(){
#if defined(SC_ENABLE_CUDA) || defined(SC_ENABLE_OPENCL)
        if(__sc_cuda){
          unmap_cuda_resources();
        }
        else if(__sc_opencl){
          unmap_opencl_resources();
        }
        else{
          _renderable->unmap_colors();
        }
#else
        _renderable->unmap_colors(); 
#endif

        exec();

        // show what we just drew
        __sc_glsdl->swapBuffers();

        bool done = __sc_glsdl->processEvent();

        if (done) exit(0);

      }

      void map_cuda_resources(){
#ifdef SC_ENABLE_CUDA
        // map one graphics resource for access by CUDA
        assert(cuGraphicsMapResources(1, &__sc_cuda_device_resource, 0) == CUDA_SUCCESS);

        size_t bytes;
        // return a pointer by which the mapped graphics resource may be accessed.
        assert(cuGraphicsResourceGetMappedPointer(&__sc_cuda_device_renderall_uniform_colors, &bytes, __sc_cuda_device_resource) == CUDA_SUCCESS);
#endif
      }

      void unmap_cuda_resources(){
#ifdef SC_ENABLE_CUDA
        assert(cuGraphicsUnmapResources(1, &__sc_cuda_device_resource, 0) == CUDA_SUCCESS);

        _renderable->alloc_texture();
#endif
      }

      void map_opencl_resources(){
#ifdef SC_ENABLE_OPENCL
        glFinish();
        clEnqueueAcquireGLObjects(__sc_opencl_command_queue, 
                                  1, 
                                  &__sc_opencl_device_renderall_uniform_colors,
                                  0,
                                  NULL,
                                  NULL);
#endif
      }

      void unmap_opencl_resources(){
#ifdef SC_ENABLE_OPENCL
        clEnqueueReleaseGLObjects(__sc_opencl_command_queue,
                                  1,
                                  &__sc_opencl_device_renderall_uniform_colors,
                                  0,
                                  NULL,
                                  NULL);
        clFinish(__sc_opencl_command_queue);
#endif
      }

      // register pbo for access by CUDA, return handle 
      void register_cuda_pbo(GLuint pbo){
#ifdef SC_ENABLE_CUDA
        assert(cuGraphicsGLRegisterBuffer(&__sc_cuda_device_resource, 
                                          pbo, 
                                          CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) == CUDA_SUCCESS);
#endif
      }

      // register pbo for access by CUDA, return handle 
      void register_opencl_pbo(GLuint pbo){
#ifdef SC_ENABLE_OPENCL
        cl_int ret;
        __sc_opencl_device_renderall_uniform_colors = 
          clCreateFromGLBuffer(__sc_opencl_context,
                               CL_MEM_WRITE_ONLY,
                               pbo,
                               &ret);
        assert(ret == CL_SUCCESS);
#endif
      }

      void exec(){
        _renderable->draw(NULL);
      }

    private:
      renderall_uniform_rt* o_;
      glQuadRenderableVA* _renderable;
  };

} // end namespace scout

renderall_uniform_rt::renderall_uniform_rt(size_t width,
    size_t height,
    size_t depth)
: renderall_base_rt(width, height, depth){

  x_ = new renderall_uniform_rt_(this);

}

renderall_uniform_rt::~renderall_uniform_rt(){
  delete x_;
}

void renderall_uniform_rt::begin(){
  x_->begin();
}

void renderall_uniform_rt::end(){
  x_->end();
}

void __sc_begin_uniform_renderall(size_t width,
    size_t height,
    size_t depth){
  if(!__sc_renderall){
    __sc_renderall = new renderall_uniform_rt(width, height, depth);
  }

  __sc_renderall->begin();

}
