/**
 * hpgv_block.h
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


#ifndef HPGV_BLOCK_H
#define HPGV_BLOCK_H

#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include "runtime/volren/hpgv/hpgv_error.h"
#include "runtime/volren/hpgv/hpgv_util.h"
#include "runtime/volren/hpgv/hpgv_utilmath.h"
#include "runtime/volren/hpgv/hpgv_gl.h"

namespace scout {

#define DATA_PARTICLE_HEALTH    0
#define GRID_SAMPLE             2048

  extern "C" {

    /**
     * block_t
     *
     */
    typedef struct blk_header_t {

      /*----------------- grid partition -----------------------------*/

      int   blk_num[3];                   /* block patition num along each axis */
      int   blk_id[3];                    /* block id along each axis */
      int   blk_neighbor_x[2];            /* x neighbor block mpiid */
      int   blk_neighbor_y[2];            /* y neighbor block mpiid */
      int   blk_neighbor_z[2];            /* z neighbor block mpiid */
      int   blk_neighbor[26];             /* neighbor block mpiid */

      /*----------------- physical coordinate --------------------------*/

      float domain_obj_near[3];               /* domain near end coordinate */
      float domain_obj_far[3];                /* domain far end coordinate */
      float domain_obj_size[3];               /* domain size */
      float domain_obj_center[3];             /* domain center */
      float domain_obj_maxsize;               /* domain maxsize */

      float blk_obj_near[3];                  /* block near end coordinate */
      float blk_obj_far[3];                   /* block far end coordinate */
      float blk_obj_size[3];                  /* size of block */
      float blk_obj_center[3];                /* block center */

      float blk_obj_packnear[3];              /* block near after pack */
      float blk_obj_packfar[3];               /* block far after pack */
      float blk_obj_packsize[3];              /* block size after pack */

      float blk_obj_raynear[3];               /* block near for ray casting */
      float blk_obj_rayfar[3];                /* block far for ray casting */

      float blk_obj_centridiff;               /* centrical difference space */
      /*----------------- grid coordinate -----------------------------*/
      int   domain_grid_near[3];               /* domain near end coordinate */
      int   domain_grid_far[3];                /* domain far end coordinate */
      int   domain_grid_size[3];              /* domain size */

      int   blk_grid_near[3];                 /* block near end coordinate */
      int   blk_grid_far[3];                  /* block far end coordinate */
      int   blk_grid_size[3];                 /* size of block */

      int   blk_grid_boundary[3][2];          /* boundary (ghost) size */
      int   blk_grid_packnear[3];             /* block near after pack */
      int   blk_grid_packfar[3];              /* block far after pack */
      int   blk_grid_packsize[3];             /* packed size */

      /*----------------- lookup table --------------------------------*/
      int   blk_coord_lookup;
      float blk_coord_obj2grid[3][GRID_SAMPLE];
      float blk_coord_grid2obj[3][GRID_SAMPLE];

    } blk_header_t;


    /**
     * volume_data_t
     *
     */
    typedef struct volume_data_t {
      int             mpiid;
      MPI_Comm        mpicomm;

      int             data_type;
      int             data_typesize;

      int             data_sphere;

      uint8_t         data_from_file;

      void            *data_original;
      void            *data_boundary_x[2];
      void            *data_boundary_y[2];
      void            *data_boundary_z[2];
      int             data_varname;

      int             exg_maxbuffsize;
      void            *exg_boundary_send[2];
      void            *exg_boundary_recv[2];
    } volume_data_t;


    /**
     * particle_t
     *
     */
    typedef struct particle_data_t {
      int         mpiid;
      MPI_Comm    mpicomm;

      int         data_type;
      int         data_typesize;

      double      *data_particle_x;         /* pointer to particle x location */
      double      *data_particle_y;         /* pointer to particle y location */
      double      *data_particle_z;         /* pointer to particle z location */
      double      *data_particle_v;         /* pointer to particle value */
      int         *data_particle_s;         /* pointer to particle state */
      double      *data_particle_loc;       /* pointer to the memory (in-situ) */
      int         data_particlename;        /* partilce variable name */
      uint32_t    data_particlesize;        /* number of particles */
    } particle_data_t;


    /**
     * block_t
     *
     */
    typedef struct block_t {
      int                 mpiid;
      MPI_Comm            mpicomm;

      blk_header_t        blk_header;

      int                 volume_num;
      volume_data_t       **volume_data;

      particle_data_t     *particle_data;
    } block_t;


    typedef float quantize_t(float value, int varname);
    void hpgv_block_print(int mpiid, int root, block_t *block);
    void header_new(int id, MPI_Comm mpicomm, int groupsize,
          double *x, double *y, double *z,
          int nx, int ny, int nz,
          int npx, int npy, int npz,
          blk_header_t *header);

    void block_init(block_t *block, int mpiid,  MPI_Comm mpicomm,
        blk_header_t header);

    void block_finalize(block_t *block);

    void block_set_quantize(quantize_t *quantize);

    void block_set_emulatesphere(block_t *block, int b);

    void block_add_volume(block_t *block, int datatype, void *data, int varname);

    void block_exchange_all_boundary(block_t *block);

    void block_exchange_boundary(block_t *block, int vol);

    int block_get_value(block_t *block, int vol, float x, float y, float z,
        float *v);

    int block_get_gradient(block_t *block, int vol, float x, float y, float z,
        point_3d_t *gradient);

    int block_get_particlevalue(block_t *block, int vol, float x, float y, float z,
        float *v);

    void block_read_data(block_t *block, int vol, char *filename);

    void block_write_data(block_t *block, int vol, char *filename);

    void block_init_neighbor(blk_header_t *header);


    int block_coord_obj2grid(blk_header_t *header,
        float ox, float oy, float oz,
        float *gx, float *gy, float *gz);

    int block_coord_grid2obj(blk_header_t *header,
        float gx, float gy, float gz,
        float *ox, float *oy, float *oz);

  }

} // end namespace scout

#endif
