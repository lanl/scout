/**
 * hpgv_render.c
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

#include "runtime/volren/hpgv/hpgv_render.h"

namespace scout {

#define INBOX(point_3d, ll_tick, ur_tick) \
(((point_3d).x3d >= (ll_tick)[0]) && ((point_3d).x3d < (ur_tick[0])) && \
 ((point_3d).y3d >= (ll_tick)[1]) && ((point_3d).y3d < (ur_tick[1])) && \
 ((point_3d).z3d >= (ll_tick)[2]) && ((point_3d).z3d < (ur_tick[2])))

#define TAG_PARTICLE_NUM            0x2143
#define TAG_PARTICLE_DAT            0x2144

int MY_STEP_GHOST_PARTICLE_TIME         = HPGV_TIMING_UNIT_134;
int MY_STEP_VOLUME_RENDER_TIME          = HPGV_TIMING_UNIT_135;
int MY_STEP_PARTICLE_RENDER_TIME        = HPGV_TIMING_UNIT_136;

int MY_STEP_MULTI_COMPOSE_TIME          = HPGV_TIMING_UNIT_150;
int MY_STEP_MULTI_VOLREND_TIME          = HPGV_TIMING_UNIT_151;
int MY_STEP_MULTI_PARREND_TIME          = HPGV_TIMING_UNIT_152;
int MY_STEP_MULTI_GHOST_TIME            = HPGV_TIMING_UNIT_153;

/**
 * rgb_t
 *
 */
typedef struct rgb_t {
    float r, g, b;
} rgb_t;


/**
 * ray_t
 *
 */
typedef struct ray_t {
    point_3d_t start;
    point_3d_t dir;
} ray_t;


/**
 * pixel_t
 *
 */
typedef struct pixel_t {
    int x, y;
} pixel_t;


/**
 * particle_t
 *
 */
typedef struct particle_t {
    double x3d, y3d, z3d;
    double v;
} particle_t;


/**
 * pixel_ctl_t
 *
 */
typedef struct cast_ctl_t {
    pixel_t     pixel;
    int32_t     firstpos;
    int32_t     lastpos;
    ray_t       ray;
    uint64_t    offset;
} cast_ctl_t;



/**
 * vis_control_t
 *
 */
typedef struct vis_control_t {
    /* MPI */
    int             id, groupsize, root; 
    MPI_Comm        comm;  
    
    /* rendering parameters*/
    int             updateview;
    para_input_t    *para_input;
    float           sampling_spacing;
    int             colormapsize;
    int             colormapformat;
    int             colormaptype;
    
    /* data */
    block_t         *block;
    
    /* internal rendering parameters */
    point_3d_t      eye_obj;
    double          screen_min_z;
    double          screen_max_z;
    float           block_depth;
    cast_ctl_t      *cast_ctl;
    uint64_t        castcount;
    
    
    /* compositing output */ 
    void            *colorimage;
    int             colorimagesize;
    int             colorimagetype;
    int             colorimageformat;
    int             rendercount;
    
    /* rendering choice */
    //int             renderparticle;
    int             rendervolume;
    
} vis_control_t;

static vis_control_t *theVisControl = NULL;


/**
 * vis_pixel_to_ray: 
 * 
 */
void 
vis_pixel_to_ray(vis_control_t *visctl, pixel_t pixel, 
               ray_t *ray, point_3d_t eye_obj)
{
    /* pixel object coordinate*/
    point_3d_t pixel_obj;
        
    /* calcuate the object coordinate*/
    hpgv_gl_unproject(pixel.x,
                      pixel.y,
                      visctl->screen_max_z,
                      &(pixel_obj.x3d),
                      &(pixel_obj.y3d),
                      &(pixel_obj.z3d));
    
    /* ray's start */
    VEC_CPY(eye_obj, ray->start);
    
    /* ray's direction*/    
    VEC_MINUS(pixel_obj, eye_obj, ray->dir);

    normalize(&(ray->dir));
}


/**
 * ray_intersect_box
 *
 */
int
vis_ray_intersect_box(ray_t *ray, float lend[3], float hend[3],
                      double *tnear, double *tfar)
{
    double ray_d[3] , ray_o[3], temp,t1, t2;
    uint8_t i, isfirst = 1;
    
    ray_d[0] = ray->dir.x3d;
    ray_d[1] = ray->dir.y3d;
    ray_d[2] = ray->dir.z3d;

    ray_o[0] = ray->start.x3d;
    ray_o[1] = ray->start.y3d;
    ray_o[2] = ray->start.z3d;

    for (i=0; i<3; i++) {
        if (ray_d[i]==0) {
            if (ray_o[i] < lend[i] || ray_o[i] > hend[i]) {
                return HPGV_FALSE;
            }
        } else {
            t1 = (lend[i] - ray_o[i]) / ray_d[i];
            t2 = (hend[i] - ray_o[i]) / ray_d[i];
            if (t1 > t2) {
                temp = t2;
                t2 = t1;
                t1 = temp;
            }    
            if (isfirst == 1) {
                *tnear = t1;
                *tfar = t2;
                isfirst = 0;
            } else  {
                if (t1 > *tnear) {
                    *tnear = t1;
                }
    
                if (t2 < *tfar) {
                    *tfar = t2;
                }
            }
            if (*tnear > *tfar) {
                return HPGV_FALSE;
            }
            if (*tfar < 0) {
                return HPGV_FALSE;
            }
        }
    }

    return HPGV_TRUE;
}


/**
 * vis_ray_clip_box
 *
 */
int
vis_ray_clip_box(int id, ray_t *ray, double sampling_spacing, 
                 float ll[3], float ur[3],
                 int32_t *pnfirst, int32_t *pnlast)
{
    double tnear = 0, tfar = 0;
    int32_t nstart, nend, sample_pos;
    point_3d_t real_sample;
    
    HPGV_ASSERT_P(id, pnfirst, "pnfirst is null.", HPGV_ERR_MEM);
    HPGV_ASSERT_P(id, pnlast,  "pnlast is null.",  HPGV_ERR_MEM);
    
    /* A quick check */
    if (vis_ray_intersect_box(ray, ll, ur, &tnear, &tfar) == HPGV_FALSE) {
        //fprintf(stderr, "did not intersect box\n");
        return HPGV_FALSE;
    }
    
    nstart = (int32_t)(floor(tnear / sampling_spacing));
    nend = (int32_t)(ceil(tfar / sampling_spacing));
   
    /* make sure that nstart is not greater than nend */
    HPGV_ASSERT_P(id, 
                  nstart <= nend, 
                  "ray_clip_box: fatal internal error. Abort!", 
                  HPGV_ERROR);
    
    /* if nstart is equal to nend, it's only possible when there is only
       one sampling point in the domain */    
    if (nstart == nend) {
        /* Find the entry position */
        real_sample.x3d 
            = ray->start.x3d + ray->dir.x3d * sampling_spacing * nstart;
        real_sample.y3d 
            = ray->start.y3d + ray->dir.y3d * sampling_spacing * nstart;
        real_sample.z3d 
            = ray->start.z3d + ray->dir.z3d * sampling_spacing * nstart;
        
        if (INBOX(real_sample, ll, ur)) {
            *pnfirst = nstart;
            *pnlast = nend + 1;
            return HPGV_TRUE;
        } else {
            //fprintf(stderr,"not INBOX\n");
            return HPGV_FALSE;
        }
    }
    
    /* Now check the normal case */
    
    /* Find the entry position */
    real_sample.x3d  = ray->start.x3d + ray->dir.x3d * sampling_spacing * nstart;
    real_sample.y3d  = ray->start.y3d + ray->dir.y3d * sampling_spacing * nstart;
    real_sample.z3d  = ray->start.z3d + ray->dir.z3d * sampling_spacing * nstart;

    if (INBOX(real_sample, ll, ur)) {
        /* probe backward */
        sample_pos = nstart;
        
        while (1) {
            real_sample.x3d  = ray->start.x3d + 
                ray->dir.x3d * sampling_spacing * (sample_pos - 1);
            real_sample.y3d  = ray->start.y3d + 
                ray->dir.y3d * sampling_spacing * (sample_pos - 1);
            real_sample.z3d  = ray->start.z3d + 
                ray->dir.z3d * sampling_spacing * (sample_pos - 1);
            
            /* INBOX test will fail sooner or later. So there will not
               be an infinite loop */
            if (INBOX(real_sample, ll, ur)) {
                sample_pos--;
            } else {
                break;
            }
        }

        /* sample_pos is inside the domain while sample_pos - 1 is outside */
        *pnfirst = sample_pos;
        
    } else {
        /* probe forward */
        sample_pos = nstart;
        
        /* might loop forever if being stubborn unwisely */
        do {
            sample_pos++;

            real_sample.x3d  = ray->start.x3d +
                ray->dir.x3d * sampling_spacing * sample_pos;
            real_sample.y3d  = ray->start.y3d +
                ray->dir.y3d * sampling_spacing * sample_pos;
            real_sample.z3d  = ray->start.z3d +
                ray->dir.z3d * sampling_spacing * sample_pos;

            if (INBOX(real_sample, ll, ur)) {
                break;
            }
        } while (sample_pos <= nend);

        if (sample_pos <= nend) {
            *pnfirst = sample_pos;
        } else {
            /* start and end are all out of domain */
            //fprintf(stderr, "start and end out of domain\n");
            return HPGV_FALSE;
        }
    }


    /* Now check the last position. */
    
    /* Check the nend position first */
    real_sample.x3d =  ray->start.x3d + ray->dir.x3d * sampling_spacing * nend;
    real_sample.y3d  = ray->start.y3d + ray->dir.y3d * sampling_spacing * nend;
    real_sample.z3d  = ray->start.z3d + ray->dir.z3d * sampling_spacing * nend;
        
    if (INBOX(real_sample, ll, ur)) {
        /* probe forward */
        sample_pos = nend;
        while (1) {
            sample_pos++;

            real_sample.x3d  = ray->start.x3d +
                ray->dir.x3d * sampling_spacing * sample_pos;
            real_sample.y3d  = ray->start.y3d +
                ray->dir.y3d * sampling_spacing * sample_pos;
            real_sample.z3d  = ray->start.z3d +
                ray->dir.z3d * sampling_spacing * sample_pos;

            /* The ray will sooner or later shoot out of the box. No
               infinite loop possible */
            if (!INBOX(real_sample, ll, ur)) {
                break;
            }
        }

        /* sample_pos is outside the domain. sample_pos - 1 is inside
           the domain */
        *pnlast = sample_pos;

    } else {
        /* probe backward */
        sample_pos = nend;

        /* might loop forever if being stubborn unwisely */
        do {
            real_sample.x3d  = ray->start.x3d +
                ray->dir.x3d * sampling_spacing * (sample_pos - 1);
            real_sample.y3d  = ray->start.y3d +
                ray->dir.y3d * sampling_spacing * (sample_pos - 1);
            real_sample.z3d  = ray->start.z3d +
                ray->dir.z3d * sampling_spacing * (sample_pos - 1);
        
            if (!INBOX(real_sample, ll, ur)) {
                sample_pos--;
            } else {
                break;
            }
        } while (sample_pos > *pnfirst);

        if (sample_pos > *pnfirst) {
            /* sample_pos is outside the domain, sample_pos - 1 is inside
               the domain */
            *pnlast = sample_pos;
        } else {
            return HPGV_FALSE;
        }
    }
    
    
    HPGV_ASSERT_P(id, 
                  *pnlast - *pnfirst >= 1,
                  "ray_clip_box: fatal internal error. Abort!", 
                  HPGV_ERROR);
    
    /* Must clip */
    return HPGV_TRUE;
}


/**
 * vis_update_eyedepth
 *
 */
void
vis_update_eyedepth(vis_control_t *visctl)
{
    point_3d_t eye_org;
    point_3d_t blk_obj_center;
    
    /* the eye position in the eye coordinate*/
    eye_org.x3d = 0;
    eye_org.y3d = 0;
    eye_org.z3d = 0;
    
    /* transform the eye position to the object space*/
    double view_matrix_inv[16];
    hpgv_gl_get_viewmatrixinv(view_matrix_inv);
    MAT_TIME_VEC(view_matrix_inv, eye_org, visctl->eye_obj);
        
    /* the block depth */
    VEC_SET(blk_obj_center,
            visctl->block->blk_header.blk_obj_center[0],
            visctl->block->blk_header.blk_obj_center[1],
            visctl->block->blk_header.blk_obj_center[2]);
    
    visctl->block_depth = DIST_SQR_VEC(blk_obj_center, visctl->eye_obj);
}




/**
 * vis_update_projarea
 *
 */
void
vis_update_projarea(vis_control_t *visctl, float sampling_spacing) 
{
    int i;
    int32_t x, y;
    int32_t minscrx = 0, minscry = 0, maxscrx = 0, maxscry = 0;
    double minscrz = 0, maxscrz = 0;
    uint64_t totalcasts, index, all_index;    
    point_3d_t vertex[8], screen[8];
    ray_t ray;
    pixel_t pixel;
    int32_t local_firstpos, local_lastpos;

    
    /* allocate an array to keep track of non-background local pixels */
    totalcasts = hpgv_gl_get_framesize();    
    visctl->cast_ctl = (cast_ctl_t *)realloc(visctl->cast_ctl, 
                        sizeof(cast_ctl_t) * totalcasts);
    HPGV_ASSERT_P(visctl->id, visctl->cast_ctl, "Out of memory.", HPGV_ERR_MEM);
    
    /* eight vertexes */
    for (i = 0 ; i < 8 ; i++) {
        if (i % 2 == 0)
            vertex[i].x3d = visctl->block->blk_header.domain_obj_near[0];
        else
            vertex[i].x3d = visctl->block->blk_header.domain_obj_far[0];
        
        if ((i>>1) % 2 == 0)
            vertex[i].y3d = visctl->block->blk_header.domain_obj_near[1];
        else
            vertex[i].y3d = visctl->block->blk_header.domain_obj_far[1];
        
        if ((i>>2) % 2 == 0)
            vertex[i].z3d = visctl->block->blk_header.domain_obj_near[2];
        else
            vertex[i].z3d = visctl->block->blk_header.domain_obj_far[2];
    }    
    
    /* eight projected screen vertexes and bounding box*/
    int framebuf_size_x = hpgv_gl_get_framewidth();
    int framebuf_size_y = hpgv_gl_get_frameheight();
    
    for ( i = 0; i < 8; i++) {        
        hpgv_gl_project(vertex[i].x3d, vertex[i].y3d, vertex[i].z3d,
                        &(screen[i].x3d), &(screen[i].y3d), &(screen[i].z3d));
    
        if (i == 0){
            minscrx = maxscrx = (int32_t)screen[i].x3d;
            minscry = maxscry = (int32_t)screen[i].y3d;
            minscrz = maxscrz = screen[i].z3d;
        }else{
            if (minscrx > (int32_t)screen[i].x3d)
                minscrx = (int32_t)screen[i].x3d;
            if (maxscrx < (int32_t)screen[i].x3d)
                maxscrx = (int32_t)screen[i].x3d;
            if (minscry > (int32_t)screen[i].y3d)
                minscry = (int32_t)screen[i].y3d;
            if (maxscry < (int32_t)screen[i].y3d)
                maxscry = (int32_t)screen[i].y3d;
            if (minscrz > screen[i].z3d)
                minscrz = screen[i].z3d;
            if (maxscrz < screen[i].z3d)
                maxscrz = screen[i].z3d;
        }
    }
    
    if (minscrx < 0 ) {
        minscrx = 0;
    }
    if (minscrx > framebuf_size_x - 1) {
        minscrx = framebuf_size_x - 1;
    }
    if (maxscrx < 0 ) {
        maxscrx = 0;
    }
    if (maxscrx > framebuf_size_x - 1) {
        maxscrx = framebuf_size_x - 1;
    }
        
    if (minscry < 0 ) {
        minscry = 0;
    }
    if (minscry > framebuf_size_y - 1) {
        minscry = framebuf_size_y - 1;
    }
    if (maxscry < 0 ) {
        maxscry = 0;
    }
    if (maxscry > framebuf_size_y - 1) {
        maxscry = framebuf_size_y - 1;
    }
    
    visctl->screen_max_z = maxscrz;
    visctl->screen_min_z = minscrz;

    /* filter out the background pixels */
    index = 0;
    all_index = 0;
    for (y = minscry; y <= maxscry; y++) {
        for (x = minscrx; x <= maxscrx; x++) {
            
            pixel.x = x;
            pixel.y = y;
            vis_pixel_to_ray(visctl, pixel, &ray, visctl->eye_obj);

          if ( vis_ray_clip_box(visctl->id,
                                    &ray, 
                                    sampling_spacing,
                                    visctl->block->blk_header.blk_obj_raynear,
                                    visctl->block->blk_header.blk_obj_rayfar, 
                                    &local_firstpos,
                                    &local_lastpos) == HPGV_TRUE) 
          {
                /* record the pixel */
                visctl->cast_ctl[index].pixel    = pixel;
                visctl->cast_ctl[index].firstpos = local_firstpos;
                visctl->cast_ctl[index].lastpos  = local_lastpos;
                visctl->cast_ctl[index].ray      = ray;
                visctl->cast_ctl[index].offset   = all_index;

                index++;
                
          }
        }
    }


    /* shrink the array */
    visctl->castcount = index;
    visctl->cast_ctl = (cast_ctl_t *)realloc(visctl->cast_ctl,
                       visctl->castcount * sizeof(cast_ctl_t));
    if (index != 0) {
        HPGV_ASSERT_P(visctl->id, visctl->cast_ctl, "Out of memory.",
                    HPGV_ERR_MEM);
    }
    
}



/**
 * vis_color_comoposite
 * 
 */
void
vis_color_comoposite(rgba_t *partialcolor, rgba_t *compositecolor) 
{
    float a = 1 - compositecolor->a;
    compositecolor->r   += a * partialcolor->r;
    compositecolor->g += a * partialcolor->g;
    compositecolor->b  += a * partialcolor->b;
    compositecolor->a += a * partialcolor->a;
}


/**
 * vis_volume_lighting
 *
 */
void
vis_volume_lighting(block_t *block, int vol, float lightpar[4],
                    point_3d_t *pos, point_3d_t eye, float *amb_diff, 
                    float *specular)
{
    point_3d_t gradient, lightdir, raydir, H;
    float ambient, diffuse, dotHV;
    
    if (block_get_gradient(block, vol, pos->x3d, pos->y3d, pos->z3d, &gradient)
        == HPGV_FALSE)
    {
        *amb_diff = lightpar[0];
        *specular = 0;
        return;
    }

    VEC_MINUS(eye, *pos, raydir);
    normalize(&raydir);
    
    //VEC_SET(lightdir, 0, 0, 1);
    //VEC_SET(lightdir, 1, 1, 1);
    VEC_SET(lightdir, raydir.x3d, raydir.y3d, raydir.z3d);
    
    normalize(&lightdir);
    
    float dot_a = VEC_DOT_VEC(lightdir, gradient);

    gradient.x3d *= -1;
    gradient.y3d *= -1;
    gradient.z3d *= -1;

    float dot_b = VEC_DOT_VEC(lightdir, gradient);

    float dot = (dot_a > dot_b ? dot_a : dot_b);

    ambient = lightpar[0];
    diffuse = lightpar[1] * dot;
    
    VEC_ADD(lightdir, raydir, H);
    normalize(&H);
   
     
    dot_a = VEC_DOT_VEC(H, gradient);

    gradient.x3d *= -1;
    gradient.y3d *= -1;
    gradient.z3d *= -1;

    dot_b = VEC_DOT_VEC(H, gradient);

    dotHV = (dot_a > dot_b ? dot_a : dot_b);

    *specular = 0.0;
    
    if (dotHV > 0.0) {
        *specular = lightpar[2] * pow(dotHV, lightpar[3]);
        *specular = CLAMP(*specular, 0, 1);
    }

    *amb_diff = ambient + diffuse;

    *amb_diff = CLAMP(*amb_diff, 0, 1);
}


int mycounter = 0;

/**
 * vis_render_pos
 *
 */
void
vis_render_pos(vis_control_t    *visctl,
               block_t          *block,
               ray_t            *ray,
               point_3d_t       *pos,
               rgba_t           *color,
               int              num_vol,
               float            sampling_spacing,
               int              *id_vol,
               para_tf_t        *tf_vol,
               para_light_t     *light_vol,
               trans_func_ab_t     transfer_func_user)
{

    rgba_t partialcolor;

#ifdef DRAW_PARTITION
    if (visctl->id % 3 == 0) {
        partialcolor.r   = (float)(visctl->id + 1)/visctl->groupsize;
        partialcolor.g = 0;
        partialcolor.b  = 0;
        partialcolor.a = 1;
    }
    
    if (visctl->id % 3 == 1) {
        partialcolor.r   = 0;
        partialcolor.g = (float)(visctl->id + 1)/visctl->groupsize;
        partialcolor.b  = 0;
        partialcolor.a = 1;
    }

    if (visctl->id % 3 == 2) {
        partialcolor.r   = 0;
        partialcolor.g = 0;
        partialcolor.b  = (float)(visctl->id + 1)/visctl->groupsize;
        partialcolor.a = 1;
    }
    
    vis_color_comoposite(&partialcolor, color);
    
#else
    float v;
    int id;
    int vol;
    
    for (vol = 0; vol < num_vol; vol++) {
      //int retval = transfer_func_user(block, pos, partcolor);
      int retval = transfer_func_user(block, pos, partialcolor);
      if (retval == HPGV_FALSE) {
        //printf("transfer func returned false!");
        //free(partcolor);
        return;
      }
        if (light_vol[vol].withlighting == 1 && partialcolor.a > 0.01){
            float amb_diff, specular;

            vis_volume_lighting(block,
                                id_vol[vol],
                                light_vol[vol].lightpar,
                                pos,
                                visctl->eye_obj,
                                &amb_diff,
                                &specular);

            partialcolor.r   = partialcolor.r   * amb_diff +
                                specular * partialcolor.a;
            partialcolor.g = partialcolor.g * amb_diff +
                                specular * partialcolor.a;
            partialcolor.b  = partialcolor.b  * amb_diff +
                                specular * partialcolor.a;
        }
        
        vis_color_comoposite(&partialcolor, color);
    }
    
#endif
}



/**
 * vis_ray_render_positions
 *
 */
void 
vis_ray_render_positions(vis_control_t  *visctl,
                         block_t        *block,
                         ray_t          *ray,
                         int32_t        firstpos,
                         int32_t        lastpos, 
                         rgba_t         *color, 
                         int            frag_x,
                         int            frag_y,
                         int            num_vol,
                         float          sampling_spacing,
                         int            *id_vol,
                         para_tf_t      *tf_vol,
                         para_light_t   *light_vol,
                         trans_func_ab_t  tf_user)
{
    int32_t pos;
    point_3d_t real_sample;
    
    for (pos = firstpos; pos < lastpos; pos++) {       
    
        real_sample.x3d
            = ray->start.x3d + ray->dir.x3d * sampling_spacing * pos;

        real_sample.y3d
            = ray->start.y3d + ray->dir.y3d * sampling_spacing * pos;

        real_sample.z3d
            = ray->start.z3d + ray->dir.z3d * sampling_spacing * pos;
 
        vis_render_pos(visctl,
                       block,
                       ray,
                       &real_sample,
                       color,
                       num_vol,
                       sampling_spacing,
                       id_vol,
                       tf_vol,
                       light_vol,
                       tf_user);
        
        if (color->a >= 1) {
            color->a = 1;
            break;
        }
    }
    
    return;
}


/**
 * hpgv_vis_clear_color
 *
 */
void
hpgv_vis_clear_color(vis_control_t *visctl)
{
    hpgv_gl_clear_color();
}


/**
 * vis_render_volume
 *
 */
void
vis_render_volume(vis_control_t *visctl, int num_vol, float sampling_spacing, int *id_vol, 
                  para_tf_t *tf_vol, para_light_t *light_vol, trans_func_ab_t tf_user)
{
    HPGV_TIMING_BARRIER(visctl->comm);
    HPGV_TIMING_BEGIN(MY_STEP_VOLUME_RENDER_TIME);
    HPGV_TIMING_COUNT(MY_STEP_VOLUME_RENDER_TIME);
    
    uint64_t index;
    ray_t *ray;
    int32_t firstpos, lastpos;
    pixel_t *pixel;    
    rgba_t color;

    /* clear the color buffer */
    hpgv_vis_clear_color(visctl);
   
    for (index = 0; index < visctl->castcount; index++) {

        ray = &(visctl->cast_ctl[index].ray);
        firstpos = visctl->cast_ctl[index].firstpos;
        lastpos = visctl->cast_ctl[index].lastpos;
        
        /* ray casting for normal compositing*/
        pixel = &(visctl->cast_ctl[index].pixel);
        color.r = color.g = color.b = color.a = 0.0;
        
        vis_ray_render_positions(visctl,
                                 visctl->block,
                                 ray,
                                 firstpos,
                                 lastpos,
                                 &color,
                                 pixel->x,
                                 pixel->y,
                                 num_vol,
                                 sampling_spacing,
                                 id_vol,
                                 tf_vol,
                                 light_vol,
                                 tf_user);
        
        hpgv_gl_fragcolor(pixel->x, pixel->y, 
                          color.r, color.g, color.b, color.a);
        
    }

    HPGV_TIMING_END(MY_STEP_VOLUME_RENDER_TIME);
}


/**
 * vis_particle_lighting
 *
 */
void
vis_particle_lighting(point_3d_t gradient, point_3d_t pos, float lightpar[4], 
                      point_3d_t eye, float *amb_diff, float *specular)
{
    point_3d_t lightdir, raydir, H;
    float ambient, diffuse, dotHV;
    
    VEC_MINUS(eye, pos, raydir);
    normalize(&raydir);
    
    //VEC_SET(lightdir, -1, 0, 0);
    VEC_SET(lightdir, 0, 0, 1);
    //VEC_SET(lightdir, raydir.x3d, raydir.y3d, raydir.z3d);
    
    ambient = lightpar[0];
    diffuse = lightpar[1] * VEC_DOT_VEC(lightdir, gradient);


    
    VEC_ADD(lightdir, raydir, H);
    normalize(&H);
    
    dotHV = VEC_DOT_VEC(H, gradient);
    
    *specular = 0.0;
    
    if (dotHV > 0.0) {
        *specular = lightpar[2] * pow(dotHV, lightpar[3]);
    }
    
    *amb_diff = ambient + diffuse;
}


/**
 * hpgv_vis_viewmatrix
 *
 */
void
hpgv_vis_viewmatrix(const double m[16])
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }
        
    theVisControl->updateview |= hpgv_gl_viewmatrix(m);
  
}
        
/**
 * hpgv_vis_projmatrix
 *
 */
void
hpgv_vis_projmatrix(const double m[16])
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }
        
    theVisControl->updateview |= hpgv_gl_projmatrix(m);
    
}

        
/**
 * hpgv_vis_viewport
 *
 */
void
hpgv_vis_viewport(int viewport[4])
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }
    
    theVisControl->updateview |= hpgv_gl_viewport(viewport);
}


/**
 * hpgv_vis_framesize
 *
 */
void
hpgv_vis_framesize(int width, int height, int type, int format, int framenum)
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;          
    }

    int updateview = HPGV_FALSE;
    
    updateview = hpgv_gl_framesize(width, height);
    
    int framesize = hpgv_gl_get_framesize();

    theVisControl->colorimagetype = type;
    theVisControl->colorimageformat = format;
    theVisControl->colorimagesize = framesize;
    
    if (theVisControl->id == theVisControl->root) {

        int bytenum = framesize * hpgv_typesize(type) * hpgv_formatsize(format);

        int realnum = 1;
        if (framenum > 1) {
            realnum = framenum;
        }
            
        theVisControl->colorimage 
            = (void *)realloc(theVisControl->colorimage, bytenum * realnum);
        
        HPGV_ASSERT_P(theVisControl->id, theVisControl->colorimage,
                      "Out of memory.", HPGV_ERR_MEM);
    }
    
    theVisControl->updateview |= updateview;
    
}

/**
 * hpgv_vis_tf_para
 *
 */
void
hpgv_vis_tf_para(int colormapsize, int format, int type)
{    
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }
    
    if (format != HPGV_RGBA || type != HPGV_FLOAT) {
        fprintf(stderr,
                "HPGV currently only supports the colormap with RGBA and float format.");
        return;
    }
    
    theVisControl->colormapformat = format;
    theVisControl->colormaptype = type;
    theVisControl->colormapsize = colormapsize;
}


/**
 * hpgv_vis_para
 *
 */
void
hpgv_vis_para(para_input_t *para_input)
{
    int framenum = para_input->num_image;
/*
    hpgv_vis_tf_para(para_input->colormap_size,
                     para_input->colormap_format,
                     para_input->colormap_type);
 */                   
    hpgv_vis_framesize(para_input->para_view.frame_width,
                       para_input->para_view.frame_height,
                       para_input->image_type,
                       para_input->image_format,
                       framenum);
                     
    hpgv_vis_viewmatrix(para_input->para_view.view_matrix);
    
    hpgv_vis_projmatrix(para_input->para_view.proj_matrix);
    
    hpgv_vis_viewport(para_input->para_view.view_port);


    /* We assume that all the volume rendering use the same sampling spacing.
       This will be fixed later */
    
    int updateview = HPGV_FALSE;

    if (framenum > 0) {
        if (theVisControl->sampling_spacing !=
            para_input->para_image[0].sampling_spacing)
        {
            theVisControl->sampling_spacing =
                para_input->para_image[0].sampling_spacing; 
            updateview = HPGV_TRUE;
        }
    }
    
    theVisControl->updateview |= updateview;

    theVisControl->para_input = para_input;
}


/**
 * hpgv_vis_get_imageptr
 *
 */
const void *
hpgv_vis_get_imageptr()
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return NULL;
    }

    return theVisControl->colorimage;
}


/**
 * hpgv_vis_get_imagetype
 *
 */
int
hpgv_vis_get_imagetype()
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return HPGV_ERROR;
    }

    return theVisControl->colorimagetype;
}


/**
 * hpgv_vis_get_imageformat
 *
 */
int
hpgv_vis_get_imageformat()
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return HPGV_ERROR;
    }

    return theVisControl->colorimageformat;
}



/**
 * hpgv_vis_set_rendervolume
 *
 */
int
hpgv_vis_set_rendervolume(int b)
{
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return HPGV_ERROR;
    }

    theVisControl->rendervolume = b;
    return HPGV_TRUE;
}



/**
 * hpgv_vis_valid
 *
 */
int
hpgv_vis_valid()
{
    if (theVisControl) {
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}

/**
 * hpgv_vis_composite_init
 *
 */
void
hpgv_vis_composite_init(MPI_Comm comm)
{
    hpgv_composite_init(comm);
}



/**
 * hpgv_vis_render_one_composite
 *
 */
void
hpgv_vis_render_one_composite(block_t *block, int root, MPI_Comm comm, trans_func_ab_t tf)
{
    //hpgv_msg_p(block->mpiid, root, "id: %d in hpgv_render_one_composite\n", block->mpiid);

    int img = 0;
    
    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }

    if (theVisControl->para_input->num_image == 0) {
        /* there is no image to render */
        if (block->mpiid == root) {
            fprintf(stderr, "There is no image to render.\n");
        }
        return;
    }
        
    if (theVisControl->block != block) {
        theVisControl->block = block;
        theVisControl->updateview |= HPGV_TRUE;
    }
    
    if (!(theVisControl->block)) {
        fprintf(stderr, "Skip the empty block.\n");
        return;
    }

    if (theVisControl->comm != comm) {
        MPI_Comm_rank(comm, &(theVisControl->id));
        MPI_Comm_size(comm, &(theVisControl->groupsize));
        theVisControl->comm = comm;
    }
    
    if (theVisControl->updateview) {
        vis_update_eyedepth(theVisControl);
    }    

    /* we assume that all volume rendering use the same sampling spacing, which
       will be fixed later */

    /* update the projection area */
    if (theVisControl->updateview) {
        vis_update_projarea(theVisControl, theVisControl->sampling_spacing);
        theVisControl->updateview = HPGV_FALSE;
    }

    /* prepare the buffer which holds all images */
    int i;
    int framebuf_size_x = hpgv_gl_get_framewidth();
    int framebuf_size_y = hpgv_gl_get_frameheight();
    int framebuf_size = hpgv_gl_get_framesize();
    
    int cbformat = hpgv_gl_get_cbformat();
    
    int destype = theVisControl->colorimagetype;
    int desformat = theVisControl->colorimageformat;
    
    HPGV_ASSERT_P(theVisControl->id,
                  desformat == HPGV_RGBA && cbformat == HPGV_RGBA,
                  "Unsupported pixel format.", HPGV_ERROR);
    
    int formatsize = hpgv_formatsize(desformat);
    
    void *colorbuf = NULL;
    colorbuf = (void *) calloc (framebuf_size,
                                hpgv_typesize(destype) *
                                hpgv_formatsize(desformat) *
                                theVisControl->para_input->num_image);
    HPGV_ASSERT_P(theVisControl->id, colorbuf, "Out of memory.",
                  HPGV_ERR_MEM);
    
    uint8_t  *colorbuf_uint8 = (uint8_t *)colorbuf;
    uint16_t *colorbuf_uint16 = (uint16_t *)colorbuf;
    float    *colorbuf_float = (float *)colorbuf;
    long     colorbuf_offset = framebuf_size * formatsize;

    HPGV_TIMING_COUNT(MY_STEP_MULTI_COMPOSE_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_VOLREND_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_PARREND_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_GHOST_TIME);
    
    for (img = 0; img < theVisControl->para_input->num_image; img++) {
        
        /* processing each image */
        para_image_t  *para_image
            = &(theVisControl->para_input->para_image[img]);

        if (para_image->num_vol > 0) {
            hpgv_vis_set_rendervolume(HPGV_TRUE);
        } else {
            hpgv_vis_set_rendervolume(HPGV_FALSE);
        }

        HPGV_TIMING_BARRIER(theVisControl->comm);
        HPGV_TIMING_BEGIN(MY_STEP_MULTI_VOLREND_TIME);
        
        /* volume render */
        if (para_image->num_vol > 0) {
            vis_render_volume(theVisControl,
                              para_image->num_vol,
                              theVisControl->sampling_spacing,
                              para_image->id_vol,
                              para_image->tf_vol,
                              para_image->light_vol, tf);
        }

        HPGV_TIMING_END(MY_STEP_MULTI_VOLREND_TIME);
        
        float *cbptr = hpgv_gl_get_cbptr();
        
        switch (destype) {
        case HPGV_UNSIGNED_BYTE :
            for (i = 0; i < framebuf_size * formatsize; i++) {
                if (cbptr[i] >= 1.0f) {
                    colorbuf_uint8[i] = 0xFF;
                } else if (cbptr[i] < 0.0f) {
                    colorbuf_uint8[i] = 0x00;
                } else {
                    colorbuf_uint8[i] = (uint8_t)(cbptr[i] * 0xFF);
                }
            }
            colorbuf_uint8 += colorbuf_offset;
            break;
        case HPGV_UNSIGNED_SHORT:
            for (i = 0; i < framebuf_size * formatsize; i++) {
                colorbuf_uint16[i] = (uint16_t)(cbptr[i] * 0xFFFF);
            }
            colorbuf_uint16 += colorbuf_offset;
            break;
        case HPGV_FLOAT:
            for (i = 0; i < framebuf_size * formatsize; i++) {
                colorbuf_float[i] = cbptr[i];
            }
            colorbuf_float += colorbuf_offset;
            break;
        default:
            HPGV_ABORT_P(theVisControl->id, "Unsupported pixel format.",
                         HPGV_ERROR);
        }
    }

    HPGV_TIMING_BARRIER(theVisControl->comm);
    HPGV_TIMING_BEGIN(MY_STEP_MULTI_COMPOSE_TIME);
    
    hpgv_composite(framebuf_size_x * framebuf_size_y,
                   theVisControl->para_input->num_image,
                   theVisControl->colorimageformat,
                   theVisControl->colorimagetype,
                   colorbuf,
                   theVisControl->colorimage,
                   theVisControl->block_depth,
                   theVisControl->root,
                   theVisControl->comm,
                   HPGV_TTSWAP);

    HPGV_TIMING_END(MY_STEP_MULTI_COMPOSE_TIME);
    
    free(colorbuf);

    theVisControl->updateview = HPGV_FALSE;
    theVisControl->rendercount++;
}



/**
 * hpgv_vis_render_multi_composite
 *
 */
void
hpgv_vis_render_multi_composite(block_t *block, int root, MPI_Comm comm, trans_func_ab_t tf)
{

    //hpgv_msg_p(block->mpiid, root, "id: %d in hpgv_render_multi_composite\n", block->mpiid);

    int img = 0;

    if (!theVisControl) {
        fprintf(stderr, "HPGV has not been initialized.\n");
        return;
    }

    if (theVisControl->para_input->num_image == 0) {
        /* there is no image to render */
        if (block->mpiid == root) {
            fprintf(stderr, "There is no image to render.\n");
        }
        return;
    }

    if (theVisControl->block != block) {
        theVisControl->block = block;
        theVisControl->updateview |= HPGV_TRUE;
    }

    if (!(theVisControl->block)) {
        fprintf(stderr, "Skip the empty block.\n");
        return;
    }

    if (theVisControl->comm != comm) {
        MPI_Comm_rank(comm, &(theVisControl->id));
        MPI_Comm_size(comm, &(theVisControl->groupsize));
        theVisControl->comm = comm;
    }

    if (theVisControl->updateview) {
        vis_update_eyedepth(theVisControl);
    }

    /* we assume that all volume rendering use the same sampling spacing, which
       will be fixed later */

    /* update the projection area */
    if (theVisControl->updateview) {
        vis_update_projarea(theVisControl, theVisControl->sampling_spacing);
        theVisControl->updateview = HPGV_FALSE;
    }

    HPGV_TIMING_COUNT(MY_STEP_MULTI_COMPOSE_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_VOLREND_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_PARREND_TIME);
    HPGV_TIMING_COUNT(MY_STEP_MULTI_GHOST_TIME);

    for (img = 0; img < theVisControl->para_input->num_image; img++) {

        /* processing each image */
        para_image_t  *para_image
            = &(theVisControl->para_input->para_image[img]);

        if (para_image->num_vol > 0) {
            hpgv_vis_set_rendervolume(HPGV_TRUE);
        } else {
            hpgv_vis_set_rendervolume(HPGV_FALSE);
        }
        
        HPGV_TIMING_BARRIER(theVisControl->comm);
        HPGV_TIMING_BEGIN(MY_STEP_MULTI_VOLREND_TIME);
        
        /* volume render */
        if (para_image->num_vol > 0) {
            vis_render_volume(theVisControl,
                              para_image->num_vol,
                              theVisControl->sampling_spacing,
                              para_image->id_vol,
                              para_image->tf_vol,
                              para_image->light_vol,
                              tf);
        }

        HPGV_TIMING_END(MY_STEP_MULTI_VOLREND_TIME);
       
#ifdef HPGV_DEBUG_LOCAL_IMG
{
        uint64_t index, offset;
        pixel_t *pixel;

        char filename[MAXLINE];
        snprintf(filename, MAXLINE, "./image_local_t%04d_p%04d_g%04d.ppm",
                 theVisControl->rendercount, theVisControl->id, img);

        hpgv_vis_saveppm(hpgv_gl_get_framewidth(),
                         hpgv_gl_get_frameheight(),
                         hpgv_gl_get_cbformat(),
                         hpgv_gl_get_cbtype(),
                         hpgv_gl_get_cbptr(),
                         filename);
}
#endif

        int i;
        int framebuf_size_x = hpgv_gl_get_framewidth();
        int framebuf_size_y = hpgv_gl_get_frameheight();
        int framebuf_size = hpgv_gl_get_framesize();

        float *cbptr = hpgv_gl_get_cbptr();
        int cbtype = hpgv_gl_get_cbtype();
        int cbformat = hpgv_gl_get_cbformat();

        int destype = theVisControl->colorimagetype;
        int desformat = theVisControl->colorimageformat;

        HPGV_ASSERT_P(theVisControl->id,
                      desformat == HPGV_RGBA && cbformat == HPGV_RGBA,
                      "Unsupported pixel format.", HPGV_ERROR);

        int formatsize = hpgv_formatsize(desformat);

        void *colorbuf = NULL;
        uint8_t  *colorbuf_uint8 = NULL;
        uint16_t *colorbuf_uint16 = NULL;

        if (desformat == cbformat && destype == cbtype) {
            // same format and type
            colorbuf = cbptr;
        } else {
            // different format and type
            colorbuf = (void *) calloc (framebuf_size,
                                        hpgv_typesize(destype) *
                                        hpgv_formatsize(desformat));

            HPGV_ASSERT_P(theVisControl->id, colorbuf, "Out of memory.",
                          HPGV_ERR_MEM);

            switch (destype) {
            case HPGV_UNSIGNED_BYTE :
                colorbuf_uint8 = (uint8_t *) colorbuf;
                for (i = 0; i < framebuf_size * formatsize; i++) {
                    if (cbptr[i] >= 1.0f) {
                        colorbuf_uint8[i] = 0xFF;
                    } else if (cbptr[i] < 0.0f) {
                        colorbuf_uint8[i] = 0x00;
                    } else {
                        colorbuf_uint8[i] = (uint8_t)(cbptr[i] * 0xFF);
                    }
                }
                break;
            case HPGV_UNSIGNED_SHORT:
                colorbuf_uint16 = (uint16_t *) colorbuf;
                for (i = 0; i < framebuf_size * formatsize; i++) {
                    colorbuf_uint16[i] = (uint16_t)(cbptr[i] * 0xFFFF);
                }
                break;
            default:
                HPGV_ABORT_P(theVisControl->id, "Unsupported pixel format.",
                             HPGV_ERROR);
            }
        }

        long offset = img *
            framebuf_size *
            hpgv_typesize(destype) *
            hpgv_formatsize(desformat);

        char *desimage = ((char *)theVisControl->colorimage) + offset;

        HPGV_TIMING_BARRIER(theVisControl->comm);
        HPGV_TIMING_BEGIN(MY_STEP_MULTI_COMPOSE_TIME);
        
        hpgv_composite(framebuf_size_x,
                       framebuf_size_y,
                       theVisControl->colorimageformat,
                       theVisControl->colorimagetype,
                       colorbuf,
                       desimage,
                       theVisControl->block_depth,
                       theVisControl->root,
                       theVisControl->comm,
                       HPGV_TTSWAP);

        HPGV_TIMING_END(MY_STEP_MULTI_COMPOSE_TIME);
        
        if (!(desformat == cbformat && destype == cbtype)) {
            free(colorbuf);
        }
    }

    theVisControl->updateview = HPGV_FALSE;
    theVisControl->rendercount++;
}
        

/**
 * hpgv_vis_render
 *
 */
void
hpgv_vis_render(block_t *block, int root, MPI_Comm comm, int opt, trans_func_ab_t tf_user)
{
    if (opt == 0) {
        hpgv_vis_render_multi_composite(block, root, comm, tf_user);
    } else {
        hpgv_vis_render_one_composite(block, root, comm, tf_user);        
    }
}

/**
 * hpgv_vis_init
 *
 */
void
hpgv_vis_init(MPI_Comm comm, int root)
{
    int id, groupsize;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &groupsize);
    
    /* init vis control struct */
    theVisControl = (vis_control_t *)calloc(1, sizeof(vis_control_t));
    HPGV_ASSERT_P(id, theVisControl, "Out of memory.", HPGV_ERR_MEM);
    
    theVisControl->id = id;
    theVisControl->groupsize = groupsize;
    theVisControl->root = root;
    theVisControl->comm = comm;

    /* init hpgv gl moduel */
    if (!hpgv_gl_valid()) {
        hpgv_gl_init();
    }
    
    /* init hpgv compositing module */    
    if (!hpgv_composite_valid()) {
        hpgv_vis_composite_init(theVisControl->comm);
    }

    if (!HPGV_TIMING_VALID()) {
        HPGV_TIMING_INIT(root, comm);
    }
    
    static int init_timing = HPGV_FALSE;
    
    if (init_timing == HPGV_FALSE) {
        init_timing = HPGV_TRUE;
        
        HPGV_TIMING_NAME(MY_STEP_GHOST_PARTICLE_TIME,       "T_ghost");
        HPGV_TIMING_NAME(MY_STEP_VOLUME_RENDER_TIME,        "T_volrend");
        HPGV_TIMING_NAME(MY_STEP_PARTICLE_RENDER_TIME,      "T_parrend");

        HPGV_TIMING_NAME(MY_STEP_MULTI_COMPOSE_TIME,        "T_mcomp");
        HPGV_TIMING_NAME(MY_STEP_MULTI_VOLREND_TIME,        "T_mvolrend");
        HPGV_TIMING_NAME(MY_STEP_MULTI_PARREND_TIME,        "T_mparrend");
        HPGV_TIMING_NAME(MY_STEP_MULTI_GHOST_TIME,          "T_mghost");
    }
} 

/**
 * hpgv_vis_finalize
 *
 */
void 
hpgv_vis_finalize()
{
    
    if (!theVisControl) {
        fprintf(stderr, "Can not find visualization module.\n");
        return;
    }
    
    if (theVisControl->cast_ctl) {
        free(theVisControl->cast_ctl);
    }

    if (theVisControl->colorimage) {
        free(theVisControl->colorimage);
    }
       
    hpgv_composite_finalize(theVisControl->comm);
    
    hpgv_gl_finalize();
    
    free(theVisControl);

    /*
    fprintf(stderr, "Memory clear on PE %d.\n", theVisControl->id);
    */
}
}
