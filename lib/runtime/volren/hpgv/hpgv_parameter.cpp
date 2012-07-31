/**
 * hpgv_parameter.c
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

#include "runtime/volren/hpgv/hpgv_parameter.h"

namespace scout {

/**
 * hpgv_para_delete
 *
 */
void hpgv_para_delete(para_input_t *para_input)
{
    int i;
    
    if (para_input == NULL) {
        return;
    }

    if (para_input->para_image != NULL) {

        para_image_t *para_image;
        for (i = 0; i < para_input->num_image; i++) {
            para_image = &(para_input->para_image[i]);

            if (para_image->tf_particle != NULL) {
                free(para_image->tf_particle);
            }

            if (para_image->light_particle != NULL) {
                free(para_image->light_particle);
            }

            if (para_image->id_vol != NULL) {
                free(para_image->id_vol);
            }

            if (para_image->tf_vol != NULL) {
                free(para_image->tf_vol);
            }

            if (para_image->light_vol != NULL) {
                free(para_image->light_vol);
            }
        }

        free(para_input->para_image);
    }

    free(para_input);
    para_input = NULL;
}


/**
 * hpgv_para_print
 *
 */
void hpgv_para_print(para_input_t *para_input)
{
    int i, j;
    
    if (para_input == NULL) {
        return;
    }
      /* color map infomation */
    fprintf(stderr, "color map size: %d\n",  para_input->colormap_size);
    fprintf(stderr, "color map format: %d\n",  para_input->colormap_format);
    fprintf(stderr, "color map type: %d\n",  para_input->colormap_type);
    fprintf(stderr, "image type: %d\n",  para_input->image_type);
    fprintf(stderr, "image format: %d\n",  para_input->image_format);

    /* view */

    fprintf(stderr, "final view matrix: \n");
    for (i = 0; i < 4; i++ ) {
      for (j = 0; j < 4; j++ ) {
       fprintf(stderr, "%lf ", para_input->para_view.view_matrix[i*4 + j]);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "final proj matrix: \n");
    for (i = 0; i < 4; i++ ) {
      for (j = 0; j < 4; j++ ) {
       fprintf(stderr, "%lf ", para_input->para_view.proj_matrix[i*4 + j]);
      }
      fprintf(stderr, "\n");
    }

    fprintf(stderr, "viewport: \n");
    for (i = 0; i < 4; i++) {
      fprintf(stderr, "view port:  %d\n", para_input->para_view.view_port[i]);
    }
    
    fprintf(stderr, "view frame width:  %d\n", para_input->para_view.frame_width);
    fprintf(stderr, "view frame height:  %d\n", para_input->para_view.frame_height);
    fprintf(stderr, "view angle:  %f\n", para_input->para_view.view_angle);
    fprintf(stderr, "view scale:  %f\n", para_input->para_view.view_scale);

    if (para_input->para_image != NULL) {
        
        para_image_t *para_image;

        fprintf(stderr, "There are total %d images.\n", para_input->num_image);
        
        for (i = 0; i < para_input->num_image; i++) {

            fprintf(stderr, "Image #%d\n", i);
            
            para_image = &(para_input->para_image[i]);
        
            if (para_image->num_particle) {
                fprintf(stderr, "\tEnable Particle Rendering\n");
                fprintf(stderr, "\tParticle Volume #%d\n",
                        para_image->vol_particle);
                fprintf(stderr, "\tLighting (%0.4f, %0.4f, %0.4f, %0.4f)\n",
                        para_image->light_particle->lightpar[0],
                        para_image->light_particle->lightpar[1],
                        para_image->light_particle->lightpar[2],
                        para_image->light_particle->lightpar[3]);
                fprintf(stderr, "\tTF (%0.4f, %0.4f, %0.4f, %0.4f)\n",
                        para_image->tf_particle->colormap[0],
                        para_image->tf_particle->colormap[1],
                        para_image->tf_particle->colormap[2],
                        para_image->tf_particle->colormap[3]);
            } else {
                fprintf(stderr, "\tDisable Particle Rendering\n");
            }
            
            fprintf(stderr, "\n");

            if (para_image->num_vol) {
                fprintf(stderr, "\tEnable Volume Rendering\n");
                
                fprintf(stderr, "\tSampling Spacing %0.4f\n",
                        para_image->sampling_spacing);
                
                fprintf(stderr, "\tTotal Volume %d\n",
                        para_image->num_vol);

                for (j = 0; j < para_image->num_vol; j++) {
                    fprintf(stderr, "\tVol #%d ", para_image->id_vol[j]);
                    fprintf(stderr, "\tLighting (%0.4f, %0.4f, %0.4f, %0.4f)\n",
                            para_image->light_vol[j].lightpar[0],
                            para_image->light_vol[j].lightpar[1],
                            para_image->light_vol[j].lightpar[2],
                            para_image->light_vol[j].lightpar[3]);
/*
                    fprintf(stderr, "\tTF (%0.4f, %0.4f, %0.4f, %0.4f)\n",
                            para_image->tf_vol[j].colormap[0],
                            para_image->tf_vol[j].colormap[1],
                            para_image->tf_vol[j].colormap[2],
                            para_image->tf_vol[j].colormap[3]);
*/
/*
                    int c;
                    for (c = 0; c < 1024; c++) {
                      fprintf(stderr, "%f, %f, %f, %f,\n",
                        para_image->tf_vol[j].colormap[4*c+0],
                        para_image->tf_vol[j].colormap[4*c+1],
                        para_image->tf_vol[j].colormap[4*c+2],
                        para_image->tf_vol[j].colormap[4*c+3]);
                    }
*/
                }
            } else {
                fprintf(stderr, "\tDisable Volume Rendering\n");
            }

            fprintf(stderr, "\n");
        }
    }
}

/**
 * hpgv_para_serialize
 *
 */
int hpgv_para_serialize(para_input_t *para_input, char **buf, int *size)
{
    
#define CHECK_BUF(BUFFER, org_size, new_size) {\
    if (org_size < new_size) { \
        org_size = new_size + 1024;\
        (BUFFER) = (char*)realloc((BUFFER), org_size); \
        HPGV_ASSERT((BUFFER), "Out of memory.", HPGV_ERR_MEM);\
    }\
}
    
    int i, j;    
    char *buffer = NULL;
    uint64_t bufbyte = 0;
    
    uint64_t totalbyte = 0;    
    uint64_t readbyte = 0;
    
    HPGV_ASSERT(para_input, "Out of memory.", HPGV_ERR_MEM);
    
    /* color map infomation */
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->colormap_size), readbyte);
    totalbyte += readbyte;
        
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->colormap_format), readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->colormap_type), readbyte);
    totalbyte += readbyte;
    
    /* image format and type */
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->image_format), readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->image_type), readbyte);
    totalbyte += readbyte;
    
    /* view */
    readbyte = sizeof(double) * 16;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), para_input->para_view.view_matrix, readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(double) * 16;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), para_input->para_view.proj_matrix, readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(int) * 4;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), para_input->para_view.view_port, readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->para_view.frame_width), readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->para_view.frame_height), readbyte);
    totalbyte += readbyte;
    
    readbyte = sizeof(float) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->para_view.view_angle), readbyte);
    totalbyte += readbyte;

    readbyte = sizeof(float) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->para_view.view_scale), readbyte);
    totalbyte += readbyte;
    
    /* num of image */
    readbyte = sizeof(int) * 1;
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &(para_input->num_image), readbyte);
    totalbyte += readbyte;
    
    /* each image */
    para_image_t *para_image;
    for (i = 0; i < para_input->num_image; i++) {
        para_image = &(para_input->para_image[i]);
        
        /* particle */        
        readbyte = sizeof(int) * 1;
        CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
        memcpy(&(buffer[totalbyte]), &(para_image->num_particle), readbyte);
        totalbyte += readbyte;
        
        readbyte = sizeof(float) * 1;
        CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
        memcpy(&(buffer[totalbyte]), &(para_image->particleradius), readbyte);
        totalbyte += readbyte;
                
        readbyte = sizeof(int) * 1;
        CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
        memcpy(&(buffer[totalbyte]), &(para_image->vol_particle), readbyte);
        totalbyte += readbyte;
        
        
        HPGV_ASSERT(para_image->num_particle == 0 ||
                    para_image->num_particle == 1,
                    "Error number", HPGV_ERROR);
        
        if (para_image->num_particle == 1) {
            readbyte = sizeof(para_tf_t);
            CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
            memcpy(&(buffer[totalbyte]), para_image->tf_particle, readbyte);
            totalbyte += readbyte;
            
            readbyte = sizeof(char);
            CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
            memcpy(&(buffer[totalbyte]), 
                   &(para_image->light_particle->withlighting), 
                   readbyte);
            totalbyte += readbyte;
            
            readbyte = sizeof(float) * 4;
            CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
            memcpy(&(buffer[totalbyte]), 
                   para_image->light_particle->lightpar, 
                   readbyte);
            totalbyte += readbyte;
        }
        
        /* volume */        
        readbyte = sizeof(int) * 1;
        CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
        memcpy(&(buffer[totalbyte]), &(para_image->num_vol), readbyte);
        totalbyte += readbyte;
        
        readbyte = sizeof(float);
        CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
        memcpy(&(buffer[totalbyte]), &(para_image->sampling_spacing), readbyte);
        totalbyte += readbyte;
        
        
        if (para_image->num_vol > 0) {
            readbyte = sizeof(int) * para_image->num_vol;
            CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
            memcpy(&(buffer[totalbyte]), para_image->id_vol, readbyte);
            totalbyte += readbyte;
            
            readbyte = sizeof(para_tf_t) * para_image->num_vol;
            CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
            memcpy(&(buffer[totalbyte]),para_image->tf_vol, readbyte);
            totalbyte += readbyte;            
            
            for (j = 0; j < para_image->num_vol; j++) {
                readbyte = sizeof(char);
                CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
                memcpy(&(buffer[totalbyte]), 
                       &(para_image->light_vol[j].withlighting), 
                       readbyte);
                totalbyte += readbyte;
                
                readbyte = sizeof(float) * 4;
                CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
                memcpy(&(buffer[totalbyte]), 
                       para_image->light_vol[j].lightpar, 
                       readbyte);
                totalbyte += readbyte;
            } 
        }
        
    }
    
    uint64_t checktotal = totalbyte + sizeof(uint64_t);
    
    readbyte = sizeof(uint64_t);
    CHECK_BUF(buffer, bufbyte, (totalbyte + readbyte));
    memcpy(&(buffer[totalbyte]), &checktotal, readbyte);
    totalbyte += readbyte;
    
    *buf = buffer;
    *size = totalbyte;
    
    return HPGV_TRUE;
}

/**
 * hpgv_para_write
 *
 */
int hpgv_para_write(para_input_t *para_input, char *filename)
{   
    char *buf = NULL;
    int size = 0;
    
    hpgv_para_serialize(para_input, &buf, &size);
    
    FILE *fp = fopen(filename, "wb");
    HPGV_ASSERT(fp != NULL, "Can not open input file.", HPGV_ERR_IO);
     
    fwrite(buf, size, 1, fp);
    
    fclose(fp);
    
    if (buf) {
        free(buf);
    }
    
    return HPGV_TRUE;
}


/**
 * hpgv_para_read
 *
 */
int hpgv_para_read(para_input_t **out_para_input, char *buf, int size)
{
    int i, j;

    if (size == 0 || buf == NULL) {
        return HPGV_FALSE;
    }
    
    if (*out_para_input != NULL) {
        hpgv_para_delete(*out_para_input);
    }
    
    para_input_t *para_input = (hpgv_para_input_t *)calloc(1, sizeof(hpgv_para_input_t));
    HPGV_ASSERT(para_input, "Out of memory.", HPGV_ERR_MEM);

    *out_para_input = para_input;
    
    /* processing the buffer */
    int count = 0;
    int readsize = 0;

    /* color map infomation */
    readsize = sizeof(int) * 1;
    memcpy(&(para_input->colormap_size), &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(int) * 1;
    memcpy(&(para_input->colormap_format), &(buf[count]), readsize);
    count += readsize;
    
    readsize = sizeof(int) * 1;
    memcpy(&(para_input->colormap_type), &(buf[count]), readsize);
    count += readsize;
    
    /* image format and type */
    readsize = sizeof(int) * 1;
    memcpy(&(para_input->image_format), &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(int) * 1;
    memcpy(&(para_input->image_type), &(buf[count]), readsize);
    count += readsize;
    
    /* view */
    readsize = sizeof(double) * 16;
    memcpy(para_input->para_view.view_matrix, &(buf[count]), readsize);
    count += readsize;
    
    readsize = sizeof(double) * 16;
    memcpy(para_input->para_view.proj_matrix, &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(int) * 4;
    memcpy(para_input->para_view.view_port, &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(int);
    memcpy(&(para_input->para_view.frame_width), &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(int);
    memcpy(&(para_input->para_view.frame_height), &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(float);
    memcpy(&(para_input->para_view.view_angle), &(buf[count]), readsize);
    count += readsize;

    readsize = sizeof(float);
    memcpy(&(para_input->para_view.view_scale), &(buf[count]), readsize);
    count += readsize;

    /* num of image */
    readsize = sizeof(int);
    memcpy(&(para_input->num_image), &(buf[count]), readsize);
    count += readsize;
    
    if (para_input->num_image > 0) {
        para_input->para_image
            = (para_image_t *)calloc(para_input->num_image, sizeof(para_image_t));
        HPGV_ASSERT(para_input->para_image, "Out of memory.", HPGV_ERR_MEM);
    }


    /* each image */
    para_image_t *para_image;
    for (i = 0; i < para_input->num_image; i++) {
        
        para_image = &(para_input->para_image[i]);

        /* particle */
        readsize = sizeof(int);
        memcpy(&(para_image->num_particle), &(buf[count]), readsize);
        count += readsize;

        HPGV_ASSERT(para_image->num_particle == 0 ||
                    para_image->num_particle == 1,
                    "Error number", HPGV_ERROR);

        readsize = sizeof(float);
        memcpy(&(para_image->particleradius), &(buf[count]), readsize);
        count += readsize;

        readsize = sizeof(int);
        memcpy(&(para_image->vol_particle), &(buf[count]), readsize);
        count += readsize;

        if (para_image->num_particle == 1) {
            
            para_image->tf_particle
                = (para_tf_t *)calloc(1, sizeof(para_tf_t));
            HPGV_ASSERT(para_image->tf_particle,
                        "Out of memory.", HPGV_ERR_MEM);

            readsize = sizeof(para_tf_t);
            memcpy(para_image->tf_particle, &(buf[count]), readsize);
            count += readsize;

            para_image->light_particle
                = (para_light_t *)calloc(1, sizeof(para_light_t));
            HPGV_ASSERT(para_image->light_particle,
                        "Out of memory.", HPGV_ERR_MEM);

            readsize = sizeof(char);
            memcpy(&(para_image->light_particle->withlighting),
                   &(buf[count]), readsize);
            count += readsize;

            readsize = sizeof(float) * 4;
            memcpy(para_image->light_particle->lightpar,
                   &(buf[count]), readsize);
            count += readsize;
        }
        
        /* volume */
        readsize = sizeof(int);
        memcpy(&(para_image->num_vol), &(buf[count]), readsize);
        count += readsize;

        readsize = sizeof(float);
        memcpy(&(para_image->sampling_spacing), &(buf[count]), readsize);
        count += readsize;
        
        if (para_image->num_vol > 0) {
            /* volume id */
            para_image->id_vol
                = (int *)calloc(para_image->num_vol, sizeof(int));
            HPGV_ASSERT(para_image->id_vol, "Out of memory.", HPGV_ERR_MEM);

            readsize = sizeof(int) * para_image->num_vol;
            memcpy(para_image->id_vol, &(buf[count]), readsize);
            count += readsize;

            /* volume tf */
            para_image->tf_vol
                = (para_tf_t *)calloc( para_image->num_vol, sizeof(para_tf_t));
            HPGV_ASSERT(para_image->tf_vol, "Out of memory.", HPGV_ERR_MEM);

            
            readsize = sizeof(para_tf_t) * para_image->num_vol;
            memcpy(para_image->tf_vol, &(buf[count]), readsize);
            count += readsize;
            
            /* volume lighting */
            para_image->light_vol
                = (para_light_t *)calloc(para_image->num_vol,
                                         sizeof(para_light_t));
            HPGV_ASSERT(para_image->light_vol, "Out of memory.", HPGV_ERR_MEM);

            for (j = 0; j < para_image->num_vol; j++) {
                readsize = sizeof(char);
                memcpy(&(para_image->light_vol[j].withlighting),
                       &(buf[count]), readsize);
                count += readsize;
                
                readsize = sizeof(float) * 4;
                memcpy(para_image->light_vol[j].lightpar,
                       &(buf[count]), readsize);
                count += readsize;
            }
        }
        
    }

    uint64_t totalbyte = 0;
    
    readsize = sizeof(uint64_t);
    memcpy(&(totalbyte), &(buf[count]), readsize);
    count += readsize;

    if (count != size || count != totalbyte) {
      fprintf(stderr, "Inconsistent read %d %d %ld\n", count, size, (long)totalbyte);
        exit(HPGV_ERR_IO);
    }

    free(buf);
    return HPGV_TRUE;
}

}
