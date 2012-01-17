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

#include "runtime/opengl/glSDL.h"
#include "runtime/glyph_renderall.h"
#include "runtime/opengl/glGlyphRenderable.h"
#include "runtime/opengl/glyph_vertex.h"
#include "runtime/types.h"
#include "runtime/scout_gpu.h"

// ------  LLVM - globals accessed by LLVM / CUDA driver

glyph_vertex* __sc_glyph_renderall_vertex_data;
CUdeviceptr __sc_device_glyph_renderall_vertex_data;

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
        __sc_init_sdl(width, height, camera);
      }

    _renderable = new glGlyphRenderable(npoints);

    if(__sc_gpu){
      // register buffer object for access by CUDA, return handle 
      assert(cuGraphicsGLRegisterBuffer(&__sc_device_resource, 
            _renderable->get_buffer_object_id(), 
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) ==
          CUDA_SUCCESS);
    }
    
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
    if(__sc_gpu){
      map_gpu_resources();
    }
    else{
      __sc_glyph_renderall_vertex_data =_renderable->map_vertex_data();
    }
  }


  void glyph_renderall::end()
  {
    if(__sc_gpu){
      unmap_gpu_resources();
    }
    else{
      _renderable->unmap_vertex_data();
    }

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
    // map one graphics resource for access by CUDA
    assert(cuGraphicsMapResources(1, &__sc_device_resource, 0) == CUDA_SUCCESS);

    size_t bytes;
    // return a pointer by which the mapped graphics resource may be accessed.
    assert(cuGraphicsResourceGetMappedPointer(
          &__sc_device_glyph_renderall_vertex_data, &bytes, 
          __sc_device_resource) == CUDA_SUCCESS);
  }


  // should this be a member function?
  void glyph_renderall::unmap_gpu_resources()
  {
    assert(cuGraphicsUnmapResources(1, &__sc_device_resource, 0) 
        == CUDA_SUCCESS);
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
