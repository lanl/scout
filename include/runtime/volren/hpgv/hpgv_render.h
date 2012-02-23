/**
 * hpgv_render.h
 *
 * Copyright (c) 2008 Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 * 
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */


#ifndef HPGV_RENDER_H
#define HPGV_RENDER_H

#include <mpi.h>    
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime/volren/hpgv/hpgv_gl.h"    
#include "runtime/volren/hpgv/hpgv_block.h"
#include "runtime/volren/hpgv/hpgv_utilmath.h"
#include "runtime/volren/hpgv/hpgv_util.h"
#include "runtime/volren/hpgv/hpgv_composite.h"
#include "runtime/volren/hpgv/hpgv_parameter.h"

namespace scout {

  extern "C" {
    /**
     * rgba_t
     *
     */
    typedef struct rgba_t {
      float red;
      float green;
      float blue;
      float alpha;
    } rgba_t;

    void hpgv_vis_para(para_input_t *para_input);

    typedef int trans_func_t(block_t* block, point_3d_t* pos, rgba_t* color);

    void hpgv_vis_render(block_t *block, int root, MPI_Comm comm, int opt, trans_func_t* xfer);

    const void * hpgv_vis_get_imageptr();

    int hpgv_vis_get_imagetype();

    int hpgv_vis_get_imageformat();

    void hpgv_vis_init(MPI_Comm comm, int root);

    void hpgv_vis_finalize();

    int hpgv_vis_valid();

  }

} // end namespace scout

#endif
