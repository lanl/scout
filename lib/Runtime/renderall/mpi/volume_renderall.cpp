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
#include "scout/Runtime/renderall/mpi/volume_renderall.h"
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
      int nx, int ny, int nz,  // number mesh cells in each dimension for local mesh
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_ab_t trans_func,
      int root, MPI_Comm gcomm, bool stop_mpi_after)
    : renderall_base_rt(nx, ny, nz), _camera(camera),
     _root(root), _gcomm(gcomm), _stop_mpi_after(stop_mpi_after)
  {

    int myid;
    MPI_Comm_rank(_gcomm, &myid);
    _id = myid;

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

    int procdims[3], periodic[3], mycoord[3];
    MPI_Cart_get(_gcomm, 3, procdims, periodic, mycoord);

    genGrid();

    _renderable = new glVolumeRenderable(procdims[0], procdims[1], procdims[2],
        nx, ny, nz, _x, _y, _z, win_width, win_height, 
        _camera, trans_func, _id, _root, _gcomm);

    _renderable->initialize(_camera);

    // show empty buffer
    if (_id == _root) __sc_glsdl->swapBuffers();
  }

  void volume_renderall::genGrid()
  {
    int procdims[3], periodic[3], mycoord[3]; // we only really use the prodims
    MPI_Cart_get(_gcomm, 3, procdims, periodic, mycoord);

    // determine my coordinate -- different from how MPI does it
    // so we ignore the mycoord
    int mypz = _id /(procdims[0] * procdims[1]);
    int mypx = (_id - mypz * procdims[0] * procdims[1]) % procdims[0];
    int mypy = (_id - mypz * procdims[0] * procdims[1]) / procdims[0];

    uint64_t start[3];

    start[0] = mypx * width();
    start[1] = mypy * height();
    start[2] = mypz * depth();

    // set grid with evenly spaced 1-unit ticks on all axes

    _x = (double *)calloc(width(), sizeof(double));
    _y = (double *)calloc(height(), sizeof(double));
    _z = (double *)calloc(depth(), sizeof(double));

    int i;

    for (i = 0; i < width(); i++) {
      _x[i] = start[0] + i;
    }

    for (i = 0; i < height(); i++) {
      _y[i] = start[1] + i;
    }

    for (i = 0; i < depth(); i++) {
      _z[i] = start[2] + i;
    }

  }

  volume_renderall::~volume_renderall()
  {
    if (_x) {free((void*)_x); _x = NULL;}
    if (_y) {free((void*)_y); _y = NULL;}
    if (_z) {free((void*)_z); _z = NULL;}
    delete _renderable;
  }


  void volume_renderall::addVolume(void* dataptr, unsigned volumenum){
    _renderable->addVolume(dataptr, volumenum);
  }

  void volume_renderall::begin()
  {
    // TO DO:  clear previous data in _renderable's block first

    if (_id == _root) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

  }


  void volume_renderall::end()
  {

    exec();  // calls draw, which calls render

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
    MPI_Comm gcomm,
    int meshsizex, int meshsizey, int meshsizez,  // size of mesh in each dim
    size_t win_width, size_t win_height,
    glCamera* camera, trans_func_ab_t trans_func)
{
  if(!__sc_renderall){
    int procdims[3], periodic[3], mycoord[3];
    bool stop_mpi_after = false;
    int flag;
    MPI_Initialized(&flag);
    MPI_Comm agcomm = gcomm;

    // hardwire process dims to 1 1 1 for now
    if (!flag) {
      stop_mpi_after = true;
      int argc = 1;
      char argv;
      MPI_Init(&argc, (char***)&argv);
      procdims[0] = 1;
      procdims[1] = 1;
      procdims[2] = 1;
      periodic[0] = 0;
      periodic[1] = 0;
      periodic[2] = 0;
      MPI_Cart_create(MPI_COMM_WORLD, 3, procdims, periodic, 0, &agcomm);
    } 

    __sc_renderall = new volume_renderall(meshsizex, meshsizey, meshsizez,
        win_width, win_height, camera, trans_func, 0, agcomm, stop_mpi_after);
  } 
}


void __sc_add_volume(float* dataptr, unsigned volumenum)
{
  __sc_renderall->addVolume((void*)dataptr, volumenum);
}


