/**
 * hpgv_composite.h
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


#ifndef HPGV_COMPOSITE_H
#define HPGV_COMPOSITE_H

#include <mpi.h>
#include "scout/Runtime/volren/hpgv/hpgv_gl.h"
#include "scout/Runtime/volren/hpgv/hpgv_util.h"
#include "scout/Runtime/volren/hpgv/hpgv_utilmath.h"
#include "scout/Runtime/volren/hpgv/hpgv_utiltiming.h"
    

namespace scout 
{    
  extern "C" {

    /**
     * composite_t
     *
     */
    typedef enum composite_t {
      HPGV_TTSWAP,     /* 2-3 swap */
      HPGV_DRSEND      /* direct send */    
    } composite_t;



    /**
     * hpgv_composite
     *
     */
    int  
      hpgv_composite(
          /* dimnsions of pixel rectangle */
          int width,                   
          int height,              

          /* format of pixels data. Must be one 
             of GL_RGBA, GL_RGB, HPGV_RGBA, HPGV_RGB */
          int format,                  

          /* data type of pixel data. Must be one
             of GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_FLOAT,
             HPGV_UNSIGNED_BYTE, HPGV_UNSIGNED_SHORT, HPGV_FLOAT */
          int type,                    

          /* address of partial iamge */
          void *partialpixels,         

          /* address of final image (significant only at root) */
          void *finalpixels,           

          /* visibility order */
          float depth,                 

          /* rank of root process */
          int root,                    

          /* communicator */
          MPI_Comm comm,               

          /* composite method */ 
          composite_t composite_type
            );


    /**
     * hpgv_composite_init
     *
     */
    int
      hpgv_composite_init(MPI_Comm mpicomm);


    /**
     * hpgv_composite_finalize
     *
     */
    int
      hpgv_composite_finalize(MPI_Comm mpicomm);


    /**
     * hpgv_composite_valid
     *
     */
    int
      hpgv_composite_valid();


    /**
     * hpgv_composite_disable
     *
     */
    void
      hpgv_composite_disable(int entry);

    /**
     * hpgv_composite_enable
     *
     */
    void
      hpgv_composite_enable(int entry);

  }

} // end namespace scout

#endif
