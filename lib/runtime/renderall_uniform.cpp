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
#include "runtime/renderall_uniform.h"
#include "runtime/scout_gpu.h"
#include "runtime/base_types.h"
#include "runtime/opengl/glSDL.h"
#include "runtime/opengl/glQuadRenderableVA.h"

using namespace std;
using namespace scout;

// ------  LLVM - globals accessed by LLVM / CUDA driver

float4* __sc_renderall_uniform_colors;
CUdeviceptr __sc_device_renderall_uniform_colors;

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height);

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

        if(__sc_gpu){
          //register_gpu_pbo(pbo_->id();
          register_gpu_pbo(_renderable->get_colors(),
              CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
        }

        _renderable->initialize(NULL);

        // show empty buffer
        __sc_glsdl->swapBuffers();
      }

      void begin(){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if(__sc_gpu){
          map_gpu_resources();
        }
        else{
         __sc_renderall_uniform_colors =_renderable->map_colors();
        }
      }

      void end(){
        if(__sc_gpu){
          unmap_gpu_resources();
        }
        else{
          _renderable->unmap_colors();
        }

        exec();

        // show what we just drew
        __sc_glsdl->swapBuffers();

        bool done = __sc_glsdl->processEvent();

        if (done) exit(0);

      }

      void map_gpu_resources(){
        // map one graphics resource for access by CUDA
        assert(cuGraphicsMapResources(1, &__sc_device_resource, 0) == CUDA_SUCCESS);

        size_t bytes;
        // return a pointer by which the mapped graphics resource may be accessed.
        assert(cuGraphicsResourceGetMappedPointer(&__sc_device_renderall_uniform_colors, &bytes, __sc_device_resource) == CUDA_SUCCESS);
      }

      void unmap_gpu_resources(){
        assert(cuGraphicsUnmapResources(1, &__sc_device_resource, 0) == CUDA_SUCCESS);

        _renderable->alloc_texture();
      }

      // register pbo for access by CUDA, return handle 
      void register_gpu_pbo(GLuint pbo, unsigned int flags){
        assert(cuGraphicsGLRegisterBuffer(&__sc_device_resource, pbo, flags) ==
            CUDA_SUCCESS);
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
