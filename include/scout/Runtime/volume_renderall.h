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
#ifndef SCOUT_VOLUME_RENDERALL_H_
#define SCOUT_VOLUME_RENDERALL_H_

#include "scout/Runtime/base_types.h"
#include "scout/Runtime/vec_types.h"
#include "scout/Runtime/renderall_base.h"
#include "scout/Runtime/volren/hpgv/hpgv_render.h"
#include "scout/Runtime/opengl/glCamera.h"
#include <mpi.h>

namespace scout 
{
  class glCamera;
  class glVolumeRenderable;

  class volume_renderall : public renderall_base_rt {
    public:
     volume_renderall(
      int nx, int ny, int nz,
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_ab_t trans_func,
      int root, MPI_Comm gcomm, bool stop_mpi_after);

      ~volume_renderall();
      void genGrid();
      void addVolume(void* dataptr, unsigned volumenum); 
      void begin();
      void end();
    private:
      void exec();

    private:
      glVolumeRenderable* _renderable;
      glCamera* _camera;
      int _id;
      int _root;
      MPI_Comm _gcomm;
      double *_x, *_y, *_z;
      bool _stop_mpi_after;
  };

} // end namespace scout


using namespace scout;

extern void __sc_init_volume_renderall(
    MPI_Comm gcomm, int mesh_size_x, int mesh_size_y, int mesh_size_z,
    size_t win_width, size_t win_height,
    glCamera* camera, trans_func_ab_t trans_func);

extern "C" 
void __sc_add_volume(float* dataptr, unsigned volumenum);

#endif 
