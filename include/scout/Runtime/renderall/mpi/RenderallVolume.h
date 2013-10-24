/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ###########################################################################
 *
 * Notes
 *
 * #####
 */

#ifndef SCOUT_RENDERALL_VOLUME_H_
#define SCOUT_RENDERALL_VOLUME_H_

#include "scout/types.h"
#include "scout/Runtime/renderall/RenderallBase.h"
#include "scout/Runtime/volren/hpgv/hpgv_render.h"
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/opengl/glSDL.h"
#include <mpi.h>

namespace scout
{
  class glCamera;
  class glVolumeRenderable;

  class RenderallVolume : public RenderallBase {
    public:
     RenderallVolume(
      int nx, int ny, int nz,
      size_t win_width, size_t win_height,
      glCamera* camera, trans_func_ab_t trans_func,
      int root, MPI_Comm gcomm, bool stopMpiAfter);

      ~RenderallVolume();
      void genGrid();
      void addVolume(void* dataptr, unsigned volumenum);
      void begin();
      void end();
    private:
      void exec();

    private:
      glVolumeRenderable* renderable_;
      glCamera* camera_;
      int id_;
      int root_;
      MPI_Comm gcomm_;
      double *x_, *y_, *z_;
      bool stopMpiAfter_;
      glSDL *glsdl_;
  };

} // end namespace scout


using namespace scout;

extern void  __scrt_renderall_volume_init (
    MPI_Comm gcomm, int mesh_size_x, int mesh_size_y, int mesh_size_z,
    size_t win_width, size_t win_height,
    glCamera* camera, trans_func_ab_t trans_func);

extern "C"
void __scrt_renderall_add_volume(float* dataptr, unsigned volumenum);

#endif
