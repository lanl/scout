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
#include "scout/Runtime/renderall/mpi/RenderallVolume.h"
#include "scout/Runtime/opengl/glVolumeRenderable.h"
#include "scout/Runtime/types.h"

#ifdef SC_ENABLE_CUDA
#include "scout/Runtime/cuda/CudaDevice.h"
#endif

namespace scout 
{

  using namespace std;

  RenderallVolume::RenderallVolume(
      int nx, int ny, int nz,  // number mesh cells in each dimension for local mesh
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_ab_t trans_func,
      int root, MPI_Comm gcomm, bool stopMpiAfter)
    : RenderallBase(nx, ny, nz), camera_(camera),
     root_(root), gcomm_(gcomm), stopMpiAfter_(stopMpiAfter)
  {

    int myid;
    MPI_Comm_rank(gcomm_, &myid);
    id_ = myid;

    if(id_ == root){
      glsdl_ = glSDL::Instance(win_width, win_height);
    }

    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
      cerr << "Warning: creating default camera" << endl;

      camera_ = new glCamera();

      // for combustion test
      camera_->near = 70.0;
      camera_->far = 500.0;
      camera_->fov  = 40.0;
      const glfloat3 pos = glfloat3(350.0, -100.0, 650.0);
      const glfloat3 lookat = glfloat3(350.0, 200.0, 25.0);
      const glfloat3 up = glfloat3(-1.0, 0.0, 0.0);

      
      // for volren test
      /*
      camera_->near = 70.0;
      camera_->far = 100.0;
      camera_->fov  = 40.0;
      const glfloat3 pos = glfloat3(-300.0, -300.0, -300.0);
      const glfloat3 lookat = glfloat3(0.0, 0.0, 0.0);
      const glfloat3 up = glfloat3(0.0, 0.0, -1.0);
      */

      camera_->setPosition(pos);
      camera_->setLookAt(lookat);
      camera_->setUp(up);
      camera_->resize(win_width, win_height);
     
     // default 
     //_camera = new glCamera();
    }

    int procdims[3], periodic[3], mycoord[3];
    MPI_Cart_get(gcomm_, 3, procdims, periodic, mycoord);

    genGrid();

    renderable_ = new glVolumeRenderable(procdims[0], procdims[1], procdims[2],
        nx, ny, nz, x_, y_, z_, win_width, win_height,
        camera_, trans_func, id_, root_, gcomm_);

    renderable_->initialize(camera_);

    // show empty buffer
    if (id_ == root_) glsdl_->swapBuffers();
  }

  void RenderallVolume::genGrid()
  {
    int procdims[3], periodic[3], mycoord[3]; // we only really use the prodims
    MPI_Cart_get(gcomm_, 3, procdims, periodic, mycoord);

    // determine my coordinate -- different from how MPI does it
    // so we ignore the mycoord
    int mypz = id_ /(procdims[0] * procdims[1]);
    int mypx = (id_ - mypz * procdims[0] * procdims[1]) % procdims[0];
    int mypy = (id_ - mypz * procdims[0] * procdims[1]) / procdims[0];

    uint64_t start[3];

    start[0] = mypx * width();
    start[1] = mypy * height();
    start[2] = mypz * depth();

    // set grid with evenly spaced 1-unit ticks on all axes

    x_ = (double *)calloc(width(), sizeof(double));
    y_ = (double *)calloc(height(), sizeof(double));
    z_ = (double *)calloc(depth(), sizeof(double));

    int i;

    for (i = 0; i < width(); i++) {
      x_[i] = start[0] + i;
    }

    for (i = 0; i < height(); i++) {
      y_[i] = start[1] + i;
    }

    for (i = 0; i < depth(); i++) {
      z_[i] = start[2] + i;
    }

  }

  RenderallVolume::~RenderallVolume()
  {
    if (x_) {free((void*)x_); x_ = NULL;}
    if (y_) {free((void*)y_); y_ = NULL;}
    if (z_) {free((void*)z_); z_ = NULL;}
    delete renderable_;
  }


  void RenderallVolume::addVolume(void* dataptr, unsigned volumenum){
    renderable_->addVolume(dataptr, volumenum);
  }

  void RenderallVolume::begin()
  {
    // TO DO:  clear previous data in _renderable's block first

    if (id_ == root_) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

  }


  void RenderallVolume::end()
  {

    exec();  // calls draw, which calls render

    // show what we just drew
    if (id_ == root_) {
      glsdl_->swapBuffers();

      bool done = glsdl_->processEvent();

      // fix this
      if (done) {
        if (stopMpiAfter_) {
          MPI_Finalize();
        }
        exit(0);
      }
    }
  }


  void RenderallVolume::exec(){
    if (id_ == root_) glsdl_->update();
    renderable_->draw(camera_);
  }

}


void  __scrt_renderall_volume_init(
    MPI_Comm gcomm,
    int meshsizex, int meshsizey, int meshsizez,  // size of mesh in each dim
    size_t win_width, size_t win_height,
    glCamera* camera, trans_func_ab_t trans_func)
{
  if(!__scrt_renderall){
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

    __scrt_renderall = new RenderallVolume(meshsizex, meshsizey, meshsizez,
        win_width, win_height, camera, trans_func, 0, agcomm, stop_mpi_after);
  } 
}


void __scrt_renderall_add_volume(float* dataptr, unsigned volumenum)
{
  __scrt_renderall->addVolume((void*)dataptr, volumenum);
}


