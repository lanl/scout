/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */
#include <iostream>
#include <stdlib.h>

#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/glyph_renderall.h"
#include "scout/Runtime/opengl/glGlyphRenderable.h"
#include "scout/Runtime/opengl/glyph_vertex.h"
#include "scout/Runtime/types.h"

#ifdef SC_ENABLE_CUDA
#include "scout/Runtime/cuda/scout_cuda.h"
#endif

// ------  LLVM - globals accessed by LLVM / CUDA driver

glyph_vertex* __sc_glyph_renderall_vertex_data;

#ifdef SC_ENABLE_CUDA
CUdeviceptr __sc_device_glyph_renderall_vertex_data;
#endif

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

namespace scout 
{

  using namespace std;

  glyph_renderall::glyph_renderall(size_t width, size_t height, size_t depth, 
      size_t npoints, glCamera* camera)
    : renderall_base_rt(width, height, depth), _camera(camera)
  {
      if(!__sc_glsdl){
        __sc_init_sdl(__sc_initial_width, __sc_initial_height, camera);
      }

    _renderable = new glGlyphRenderable(npoints);

#ifdef SC_ENABLE_CUDA
    if(__sc_cuda){
      // register buffer object for access by CUDA, return handle 
      assert(cuGraphicsGLRegisterBuffer(&__sc_cuda_device_resource, 
            _renderable->get_buffer_object_id(), 
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ==
          CUDA_SUCCESS);
    }
#endif // SC_ENABLE_CUDA
    
    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
        cerr << "Warning: no camera so can't view anything!" << endl;
    }

    _renderable->initialize(camera);

    // show empty buffer
    __sc_glsdl->swapBuffers();
  }


  glyph_renderall::~glyph_renderall()
  {
    delete _renderable;
  }


  void glyph_renderall::begin()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef SC_ENABLE_CUDA
    if(__sc_cuda){
      map_gpu_resources();
    }
    else{
      __sc_glyph_renderall_vertex_data =_renderable->map_vertex_data();
    }
#else
    __sc_glyph_renderall_vertex_data =_renderable->map_vertex_data();
#endif // SC_ENABLE_CUDA
  }


  void glyph_renderall::end()
  {
#ifdef SC_ENABLE_CUDA
    if(__sc_cuda){
      unmap_gpu_resources();
    }
    else{
      _renderable->unmap_vertex_data();
    }
#else
      _renderable->unmap_vertex_data();
#endif // SC_ENABLE_CUDA

    exec();

    // show what we just drew
    __sc_glsdl->swapBuffers();

    bool done = __sc_glsdl->processEvent();

    // fix this
    if (done) exit(0);
  }


  // should this be a member function?
  void glyph_renderall::map_gpu_resources()
  {
#ifdef SC_ENABLE_CUDA
    // map one graphics resource for access by CUDA
    assert(cuGraphicsMapResources(1, &__sc_cuda_device_resource, 0) == CUDA_SUCCESS);

    size_t bytes;
    // return a pointer by which the mapped graphics resource may be accessed.
    assert(cuGraphicsResourceGetMappedPointer(
          &__sc_device_glyph_renderall_vertex_data, &bytes, 
          __sc_cuda_device_resource) == CUDA_SUCCESS);
#endif // SC_ENABLE_CUDA
  }


  // should this be a member function?
  void glyph_renderall::unmap_gpu_resources()
  {
#ifdef SC_ENABLE_CUDA
    assert(cuGraphicsUnmapResources(1, &__sc_cuda_device_resource, 0) 
        == CUDA_SUCCESS);
#endif // SC_ENABLE_CUDA
  }


  void glyph_renderall::exec(){
    __sc_glsdl->update();
    _renderable->draw(_camera);
  }

}


void __sc_init_glyph_renderall(size_t width, size_t height, size_t depth, 
    size_t npoints, glCamera* camera)
{
  if(!__sc_renderall){
    __sc_renderall = new glyph_renderall(width, height, depth, npoints, camera);
  }
}
