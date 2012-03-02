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
#include "runtime/volume_renderall.h"
#include "runtime/opengl/glVolumeRenderable.h"
#include "runtime/types.h"
#include "runtime/scout_gpu.h"

// ------  LLVM - globals accessed by LLVM / CUDA driver

void* __sc_volume_renderall_data;
CUdeviceptr __sc_device_volume_renderall_data;

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

namespace scout 
{

  using namespace std;

  volume_renderall::volume_renderall(
      int npx, int npy, int npz, int nx, int ny, int nz, 
      double* x, double* y, double* z, 
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_t* trans_func,
      int id, int root, MPI_Comm gcomm)
    : renderall_base_rt(nx, ny, nz), _camera(camera),
    _id(id), _root(root)
  {
      if((_id == root) && (!__sc_glsdl)){
        __sc_init_sdl(win_width, win_height);
      }

    _renderable = new glVolumeRenderable(npx, npy, npz, nx, ny, nz, 
      x, y, z, win_width, win_height, camera, trans_func, id, root, gcomm);

    
    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
        cerr << "Warning: no camera so can't view anything!" << endl;
    }

    _renderable->initialize(camera);

    // show empty buffer
    if (_id == _root) __sc_glsdl->swapBuffers();
  }


  volume_renderall::~volume_renderall()
  {
    delete _renderable;
  }


  void volume_renderall::begin()
  {
    if (_id == _root) glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  }


  void volume_renderall::end()
  {
    _renderable->setVolumeData(__sc_volume_renderall_data);

    exec();

    // show what we just drew
    if (_id == _root) {
      __sc_glsdl->swapBuffers();

      bool done = __sc_glsdl->processEvent();

      // fix this
      if (done) exit(0);
    }
  }


  void volume_renderall::exec(){
    if (_id == _root)__sc_glsdl->update();
    _renderable->draw(_camera);
  }

}


void __sc_init_volume_renderall(int npx, int npy, int npz, 
    int nx, int ny, int nz, double* x, double* y, double* z, 
    size_t win_width, size_t win_height, 
    glCamera* camera, trans_func_t* trans_func,
    int id, int root, MPI_Comm gcomm)
{
  if(!__sc_renderall){
    __sc_renderall = new volume_renderall(npx, npy, npz, nx, ny, nz, 
        x, y, z, win_width, win_height, camera, trans_func, id, root, gcomm);
  }
}
