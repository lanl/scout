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
#ifndef SCOUT_GLYPH_RENDERALL_H_
#define SCOUT_GLYPH_RENDERALL_H_

#include "runtime/base_types.h"
#include "runtime/vec_types.h"
#include "runtime/renderall_base.h"
#include "runtime/volren/hpgv/hpgv_render.h"
#include <mpi.h>

namespace scout 
{
  class glCamera;
  class glVolumeRenderable;

  class volume_renderall : public renderall_base_rt {
    public:
     volume_renderall(
      int npx, int npy, int npz, int nx, int ny, int nz,
      double* x, double* y, double* z, 
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_t* trans_func,
      int id, int root, MPI_Comm gcomm);

      ~volume_renderall();
      void begin();
      void end();
    private:
      void map_gpu_resources();
      void unmap_gpu_resources();
      void exec();

    private:
      glVolumeRenderable* _renderable;
      glCamera* _camera;
  };

} // end namespace scout

using namespace scout;

extern void __sc_init_volume_renderall(
    int npx, int npy, int npz,
    int nx, int ny, int nz, double* x, double* y, double* z,
    size_t win_width, size_t win_height,
    glCamera* camera, trans_func_t* trans_func,
    int id, int root, MPI_Comm gcomm);

#endif 
