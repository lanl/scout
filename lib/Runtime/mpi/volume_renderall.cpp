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
#include "scout/Runtime/volume_renderall.h"
#include "scout/Runtime/opengl/glVolumeRenderable.h"
#include "scout/Runtime/types.h"

#ifdef SC_ENABLE_CUDA
#include "scout/Runtime/cuda/CudaDevice.h"
#endif

// ------  LLVM - globals accessed by LLVM / CUDA driver

void* __sc_volume_renderall_data;

#ifdef SC_ENABLE_CUDA
CUdeviceptr __sc_device_volume_renderall_data;
#endif

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
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_ab_t trans_func,
      int id, int root, MPI_Comm gcomm, bool stop_mpi_after)
    : renderall_base_rt(nx, ny, nz), _camera(camera),
    _id(id), _root(root), _stop_mpi_after(stop_mpi_after)
  {
    if((_id == root) && (!__sc_glsdl)){
      __sc_init_sdl(win_width, win_height);
    }

    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
      cerr << "Warning: creating default camera" << endl;

      _camera = new glCamera();

      // for combustion test
      _camera->near = 70.0;
      _camera->far = 500.0;
      _camera->fov  = 40.0;
      const glfloat3 pos = glfloat3(350.0, -100.0, 650.0);
      const glfloat3 lookat = glfloat3(350.0, 200.0, 25.0);
      const glfloat3 up = glfloat3(-1.0, 0.0, 0.0);

      
      // for volren test
      /*
      _camera->near = 70.0;
      _camera->far = 100.0;
      _camera->fov  = 40.0;
      const glfloat3 pos = glfloat3(-300.0, -300.0, -300.0);
      const glfloat3 lookat = glfloat3(0.0, 0.0, 0.0);
      const glfloat3 up = glfloat3(0.0, 0.0, -1.0);
      */

      _camera->setPosition(pos);
      _camera->setLookAt(lookat);
      _camera->setUp(up);
      _camera->resize(win_width, win_height);
     
     // default 
     //_camera = new glCamera();
    }

    genGrid(id, npx, npy, npz, &nx, &ny, &nz);

    _renderable = new glVolumeRenderable(npx, npy, npz, nx, ny, nz, 
        _x, _y, _z, win_width, win_height, _camera, trans_func, id, root, gcomm);


    _renderable->initialize(_camera);

    // show empty buffer
    if (_id == _root) __sc_glsdl->swapBuffers();
  }

  void volume_renderall::genGrid(int id, int npx, int npy, int npz, 
      int* pnx, int* pny, int* pnz)
  {
    int domain_grid_size[3];

    domain_grid_size[0] = *pnx;
    domain_grid_size[1] = *pny;
    domain_grid_size[2] = *pnz;

    int nx = domain_grid_size[0] / npx;
    int ny = domain_grid_size[1] / npy;
    int nz = domain_grid_size[2] / npz;

    int mypz = id /(npx * npy);
    int mypx = (id - mypz * npx * npy) % npx;
    int mypy = (id - mypz * npx * npy) / npx;

    uint64_t start[3];

    start[0] = mypx * nx;
    start[1] = mypy * ny;
    start[2] = mypz * nz;

    // set grid with evenly spaced 1-unit ticks on all axes

    _x = (double *)calloc(nx, sizeof(double));
    _y = (double *)calloc(ny, sizeof(double));
    _z = (double *)calloc(nz, sizeof(double));

    int i;

    for (i = 0; i < nx; i++) {
      _x[i] = start[0] + i;
    }

    for (i = 0; i < ny; i++) {
      _y[i] = start[1] + i;
    }

    for (i = 0; i < nz; i++) {
      _z[i] = start[2] + i;
    }

    *pnx = nx;
    *pny = ny;
    *pnz = nz;
  }

  volume_renderall::~volume_renderall()
  {
    if (_x) free((void*)_x);
    if (_y) free((void*)_y);
    if (_z) free((void*)_z);

    delete _renderable;
  }


  void volume_renderall::addVolume(void* dataptr, unsigned volumenum){
    _renderable->addVolume(dataptr, volumenum);
  }


  void volume_renderall::begin()
  {
    // TO DO:  clear previous data in _renderable's block first

    if (_id == _root) glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  }


  void volume_renderall::end()
  {

    exec();

    // show what we just drew
    if (_id == _root) {
      __sc_glsdl->swapBuffers();

      bool done = __sc_glsdl->processEvent();

      // fix this
      if (done) {
        if (_stop_mpi_after) {
          MPI_Finalize();
        }
        exit(0);
      }
    }
  }


  void volume_renderall::exec(){
    if (_id == _root)__sc_glsdl->update();
    _renderable->draw(_camera);
  }

}


void __sc_init_volume_renderall(
    int nx, int ny, int nz, 
    size_t win_width, size_t win_height, 
    glCamera* camera, trans_func_ab_t trans_func)
{
  if(!__sc_renderall){
    bool stop_mpi_after = false;
   int flag;
    MPI_Initialized(&flag);
    // hardwire process dims to 1 1 1 for now
    if (!flag) {
      stop_mpi_after = true;
      int argc = 3;
      char argv[3][10]; 
      strcpy(argv[0], "1");
      strcpy(argv[1], "1");
      strcpy(argv[2], "1");
      MPI_Init(&argc, (char***)&argv);
    }

    MPI_Comm gcomm = MPI_COMM_WORLD;
    int id;
    MPI_Comm_rank(gcomm, &id);

    __sc_renderall = new volume_renderall(1, 1, 1, nx, ny, nz, 
        win_width, win_height, camera, trans_func, id, 0, gcomm, stop_mpi_after);
  }
}

void __sc_add_volume(float* dataptr, unsigned volumenum)
{
  __sc_renderall->addVolume((void*)dataptr, volumenum);
}


