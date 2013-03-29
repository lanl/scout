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
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/renderall/renderall_surface.h"
#include "scout/Runtime/opengl/glSDL.h"

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

#ifdef SC_ENABLE_CUDA
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#endif

using namespace std;
using namespace scout;

// ------  LLVM - globals accessed by LLVM / CUDA driver


// CUDA and OPENCL parts not done yet, just borrowed code from uniform_renderall

#ifdef SC_ENABLE_CUDA
CUdeviceptr __sc_cuda_device_render_surface_colors;
#else
static bool __sc_cuda = false;
#endif

#ifdef SC_ENABLE_OPENCL
cl_mem __sc_opencl_device_render_surface_colors;
#else
static bool __sc_opencl = false;
#endif

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

renderall_surface_rt::renderall_surface_rt(size_t width, size_t height, size_t depth,
    float* vertices, float* normals, float* colors, int num_vertices, glCamera* camera) 
:renderall_base_rt(width, height, depth), _vertices(vertices),
  _normals(normals), _colors(colors), _num_vertices(num_vertices), _camera(camera)
{
  if(!__sc_glsdl){
    __sc_init_sdl(__sc_initial_width, __sc_initial_height);
  }

  _localcamera = false;

  // we need a camera or nothing will happen! 
  if (_camera ==  NULL)
  {
    cerr << "Warning: creating default camera" << endl;

    _camera = new glCamera();
    _localcamera = true;

    _camera->near = 70.0;
    _camera->far = 500.0;
    _camera->fov  = 40.0;
    const glfloat3 pos = glfloat3(350.0, -100.0, 650.0);
    const glfloat3 lookat = glfloat3(350.0, 200.0, 25.0);
    const glfloat3 up = glfloat3(-1.0, 0.0, 0.0);

    _camera->setPosition(pos);
    _camera->setLookAt(lookat);
    _camera->setUp(up);
    _camera->resize(__sc_initial_width, __sc_initial_height);

  }

  _renderable = new glSurfaceRenderable(width, height, depth, _vertices, _normals,
      colors, _num_vertices, _camera);

  // show empty buffer
  __sc_glsdl->swapBuffers();
}

renderall_surface_rt::~renderall_surface_rt(){
  if (_renderable != NULL) delete _renderable;
  if (_localcamera && (_camera != NULL)) delete _camera;
}

void renderall_surface_rt::begin(){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void renderall_surface_rt::end(){

  exec();

  // show what we just drew
  __sc_glsdl->swapBuffers();

  bool done = __sc_glsdl->processEvent();

  if (done) exit(0);

}

void renderall_surface_rt::exec(){
  _renderable->draw(_camera);
}


void __sc_begin_renderall_surface(size_t width, size_t height, size_t depth,
    float* vertices, float* normals, float* colors, size_t num_vertices, glCamera* camera){

  __sc_renderall = new renderall_surface_rt(width, height, depth, vertices, normals, 
      (float*)colors, num_vertices, camera);

}

