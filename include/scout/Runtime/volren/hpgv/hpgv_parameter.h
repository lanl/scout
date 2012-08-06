/**
 * hpgv_parameter.h
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

#ifndef HPGV_PARAMETER_H
#define HPGV_PARAMETER_H

#include "scout/Runtime/volren/hpgv/hpgv_gl.h"
#include "scout/Runtime/volren/hpgv/hpgv_util.h"

namespace scout {    

  extern "C" {

#define TF_SIZE     1024
#define TF_SIZE_4   4096

    /**
     * para_view_t
     *
     */
    typedef struct para_view_t {
      double      view_matrix[16];
      double      proj_matrix[16];
      int         view_port[4];
      int         frame_width;
      int         frame_height;
      float       view_angle;
      float       view_scale;
    } para_view_t;


    /**
     * para_light_t
     *
     */
    typedef struct para_light_t {
      char        withlighting;
      float       lightpar[4];
    } para_light_t;


    /**
     * para_tf_t
     *
     */
    typedef struct para_tf_t {
      float       colormap[TF_SIZE_4];
    } para_tf_t;



    /**
     * para_image_t
     *
     */    
    typedef struct para_image_t {
      int             num_particle;       /* 0 or 1 */
      float           particleradius;     /* particle size */
      int             vol_particle;       /* particle volume */
      para_tf_t       *tf_particle;       /* tfs for particle rendering */
      para_light_t    *light_particle;    /* particle lighting parameter */

      int             num_vol;            /* multiple volume rendering */
      float           sampling_spacing;   /* volume rendering sampling spacing */
      int             *id_vol;            /* volume id */
      para_tf_t       *tf_vol;            /* tfs for volume rendering */
      para_light_t    *light_vol;         /* volume lighting parameters */
    } para_image_t;


    /**
     * para_input_t
     *
     */
    typedef struct para_input_t {
      /* color map infomation */
      int                 colormap_size;
      int                 colormap_format;
      int                 colormap_type;

      /* image format and type */
      int                 image_format;
      int                 image_type;

      /* view */
      para_view_t         para_view;

      /* all images */
      int                 num_image;
      para_image_t        *para_image;

    } para_input_t;


    typedef para_input_t hpgv_para_input_t;

    int hpgv_para_serialize(para_input_t *para_input, char **buf, int *size);

    int hpgv_para_read(hpgv_para_input_t **para_input, char *buf, int size);

    int hpgv_para_write(hpgv_para_input_t *para_input, char *filename);

    void hpgv_para_delete(hpgv_para_input_t *para_input);

    void hpgv_para_print(hpgv_para_input_t *para_input);



  }

} // end namespace scout

#endif
