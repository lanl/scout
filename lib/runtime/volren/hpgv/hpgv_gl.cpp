/**
 * hpgv_gl.c
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

#include "runtime/volren/hpgv/hpgv_util.h"
#include "runtime/volren/hpgv/hpgv_utilmath.h"
#include "runtime/volren/hpgv/hpgv_gl.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

namespace scout{

/**
 * hpgv_rgba_t
 *
 */
typedef struct hpgv_rgba_t {
    float r, g, b, a;
} hpgv_rgba_t;


typedef struct hpgv_gl_context_t {    
    double      view_matrix[16];
    double      proj_matrix[16];
    int         view_port[4];
    double      view_matrix_inv[16];    
    double      viewproj_matrix[16];
    double      viewproj_matrix_inv[16];    
        
    uint32_t     framebuf_size_x;
    uint32_t     framebuf_size_y;
    uint32_t     framebuf_size;
    hpgv_rgba_t  *colorbuf;
    float        *depthbuf;    
    hpgv_rgba_t  *colorbuf_bind;
    float        *depthbuf_bind;
    
    int          use_colorbuf_bind;
    int          use_depthbuf_bind;
    int          use_depth_test;
    
} hpgv_gl_context_t;


hpgv_gl_context_t *theGLContext = NULL;


/**
 * hpgv_gl_init
 *
 */
void hpgv_gl_init()
{
    if (theGLContext != NULL) {
        free(theGLContext);
    }
    
    theGLContext = (hpgv_gl_context_t *)calloc(1, sizeof(hpgv_gl_context_t));
    HPGV_ASSERT(theGLContext, "Out of memory.", HPGV_ERR_MEM);
}


/**
 * hpgv_gl_finalize
 *
 */
void hpgv_gl_finalize()
{
    if (theGLContext == NULL) {
        return;
    }
    
    if (theGLContext->colorbuf) {
        free(theGLContext->colorbuf);
        theGLContext->colorbuf = NULL;
    }
    
    if (theGLContext->depthbuf) {
        free(theGLContext->depthbuf);
        theGLContext->depthbuf = NULL;
    }
}


/**
 * hpgv_gl_valid
 *
 */
int hpgv_gl_valid()
{
    if (!theGLContext) {      
        return HPGV_FALSE;
    }
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_viewmatrix
 *
 */
int hpgv_gl_viewmatrix(const double m[16])
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
      
    if(memcmp(theGLContext->view_matrix, m, sizeof(double) * 16) != 0 ) {
        
        memcpy(theGLContext->view_matrix, m, sizeof(double) * 16);
        
        /* inverse the view matrix*/
        if (inverse_mat(theGLContext->view_matrix, 
                        theGLContext->view_matrix_inv) != HPGV_SUCCESS) 
        {
            HPGV_ERR_MSG("Inverse matrix does not exist");
        }  
        
        if ( !is_mat_zero(theGLContext->proj_matrix)) {
            /* calcuate model-view-project matrix */
            mat_time_mat(theGLContext->proj_matrix, theGLContext->view_matrix,
                        theGLContext->viewproj_matrix);
            
            
            /* inverse the model-view-projection matrix*/
            if (inverse_mat(theGLContext->viewproj_matrix, 
                            theGLContext->viewproj_matrix_inv) != HPGV_SUCCESS)
            {
                HPGV_ERR_MSG("Inverse matrix does not exist");
            } 
        }
        
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}


/**
 * hpgv_gl_projmatrix
 *
 */
int hpgv_gl_projmatrix(const double m[16])
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
      
    if(memcmp(theGLContext->proj_matrix, m, sizeof(double) * 16) != 0 ) {
        
        memcpy(theGLContext->proj_matrix, m, sizeof(double) * 16);
        
        if ( !is_mat_zero(theGLContext->view_matrix)) {
            
            /* calcuate model-view-project matrix */
            mat_time_mat(theGLContext->proj_matrix, theGLContext->view_matrix,
                        theGLContext->viewproj_matrix);
            
            
            /* inverse the model-view-projection matrix*/
            if (inverse_mat(theGLContext->viewproj_matrix, 
                            theGLContext->viewproj_matrix_inv) != HPGV_SUCCESS)
            {
                HPGV_ERR_MSG("Inverse matrix does not exist");
            }
        }
        
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}


/**
 * hpgv_gl_viewport
 *
 */
int hpgv_gl_viewport(int viewport[4])
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if(memcmp(theGLContext->view_port, viewport, sizeof(int) * 4) != 0 )
    {
        memcpy(theGLContext->view_port, viewport, sizeof(int) * 4);
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}


/**
 * hpgv_gl_framesize
 *
 */
int hpgv_gl_framesize(int width, int height)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
      
    if (theGLContext->framebuf_size_x != width ||
        theGLContext->framebuf_size_y != height)
    {
        theGLContext->framebuf_size_x = width;
        theGLContext->framebuf_size_y = height;        
        theGLContext->framebuf_size = width * height;
                                   
        theGLContext->colorbuf = (hpgv_rgba_t *)realloc(theGLContext->colorbuf, 
            theGLContext->framebuf_size * sizeof(hpgv_rgba_t));
        
        HPGV_ASSERT(theGLContext->colorbuf, "Out of memory.", HPGV_ERR_MEM);
        
        theGLContext->depthbuf = (float *)realloc(theGLContext->depthbuf,
            theGLContext->framebuf_size * sizeof(float));
        
        HPGV_ASSERT(theGLContext->depthbuf, "Out of memory.", HPGV_ERR_MEM);
        
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}


/**
 * hpgv_gl_project
 *
 */
int hpgv_gl_project(const double objx, 
                    const double objy, 
                    const double objz,
                    double *winx,
                    double *winy,
                    double *winz)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    double in[4], out[4];
    in[0] = objx;
    in[1] = objy;
    in[2] = objz;
    in[3] = 1.0;
    
    mat_time_vec(theGLContext->viewproj_matrix, in, out);
    
    if (out[3] == 0) {
        return HPGV_FALSE;
    }
    
    out[0] /= out[3];
    out[1] /= out[3];
    out[2] /= out[3];
    
    /*transform 2d point to screen coordinate*/
    *winx = theGLContext->view_port[0] + 
        theGLContext->view_port[2] * (out[0] + 1.0f) / 2.0f;
    
    *winy = theGLContext->view_port[1] + 
        theGLContext->view_port[3] * (out[1] + 1.0f) / 2.0f;
    
    *winz = (out[2] + 1.0f) / 2.0f;
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_unproject
 *
 *
 */
int hpgv_gl_unproject(const double winx,
                      const double winy,
                      const double winz,
                      double *objx,
                      double *objy,
                      double *objz)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    /*transform 2d screen point to 3d*/
    double in[4], out[4];
    
    
    in[0] = (winx - theGLContext->view_port[0]) * 2.0f / 
        theGLContext->view_port[2] - 1.0f;
    
    in[1] = (winy - theGLContext->view_port[1]) * 2.0f / 
        theGLContext->view_port[3] - 1.0f;
    
    in[2] = winz * 2.0f - 1.0f;
    
    in[3] = 1.0;
    
    mat_time_vec(theGLContext->viewproj_matrix_inv, in, out);
    
    if (out[3] == 0) {
        return HPGV_FALSE;
    }
    
    *objx = out[0] / out[3];
    *objy = out[1] / out[3];
    *objz = out[2] / out[3];
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_clear_color
 *
 */
int hpgv_gl_clear_color()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if (theGLContext->use_colorbuf_bind) {
        memset(theGLContext->colorbuf_bind, 0, 
               sizeof(hpgv_rgba_t) * theGLContext->framebuf_size);
    }
    
    memset(theGLContext->colorbuf, 0, 
           sizeof(hpgv_rgba_t) * theGLContext->framebuf_size);
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_clear_depth
 *
 */
int hpgv_gl_clear_depth()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    float *depthbuf = NULL;
    
    if (theGLContext->use_depthbuf_bind) {
        depthbuf = theGLContext->depthbuf_bind;
    } else {
        depthbuf = theGLContext->depthbuf;
    }
    
    int i;
    for (i = 0; i < theGLContext->framebuf_size; i++) {
        depthbuf[i] = 2;
    }
        
    return HPGV_TRUE;
}


/**
 * hpgv_gl_get_viewmatrix
 *
 */
int hpgv_gl_get_viewmatrix(double m[16])
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    memcpy(m, theGLContext->view_matrix, sizeof(double) * 16);
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_get_viewmatrixinv
 *
 */
int hpgv_gl_get_viewmatrixinv(double m[16])
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    memcpy(m, theGLContext->view_matrix_inv, sizeof(double) * 16);
    
    return HPGV_TRUE;
}


/**
 * hpgv_gl_get_framesize
 *
 */    
int hpgv_gl_get_framesize()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    return theGLContext->framebuf_size;
}



/**
 * hpgv_gl_get_framewidth
 *
 */    
int hpgv_gl_get_framewidth()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    return theGLContext->framebuf_size_x;
}


/**
 * hpgv_gl_get_frameheight
 *
 */    
int hpgv_gl_get_frameheight()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    return theGLContext->framebuf_size_y;
}


/**
 * hpgv_gl_get_depthptr
 *
 */
void * hpgv_gl_get_depthptr() 
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if (theGLContext->use_depthbuf_bind) {
        return (void *)theGLContext->depthbuf_bind;
    }
    
    return (void *)theGLContext->depthbuf;
}



/**
 * hpgv_gl_get_cbptr
 *
 */
float * hpgv_gl_get_cbptr()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if (theGLContext->use_colorbuf_bind) {
        return (float *)theGLContext->colorbuf_bind;
    }
    
    return (float *)theGLContext->colorbuf;
}

/**
 * hpgv_gl_get_cbtype
 *
 */
int hpgv_gl_get_cbtype()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    return HPGV_FLOAT;
}


/**
 * hpgv_gl_get_cbformat
 *
 */
int hpgv_gl_get_cbformat()
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    return HPGV_RGBA;
}


/**
 * hpgv_gl_fragcolor
 *
 */    
int hpgv_gl_fragcolor(int x, int y, float r, float g, float b, float a)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if (x < 0 && x >= theGLContext->framebuf_size_x) {
        return HPGV_FALSE;
    }
    
    if (y < 0 && y >= theGLContext->framebuf_size_y) {
        return HPGV_FALSE;
    }    
    
    hpgv_rgba_t *colorbuf = NULL;
    
    if (theGLContext->use_colorbuf_bind) {
        colorbuf = theGLContext->colorbuf_bind;
    } else {
        colorbuf = theGLContext->colorbuf;
    }
    
    int offset = x + y * theGLContext->framebuf_size_x;
    
    colorbuf[offset].r = r;
    colorbuf[offset].g = g;
    colorbuf[offset].b = b;
    colorbuf[offset].a = a;
    
    return HPGV_TRUE;        
}


/**
 * hpgv_gl_fragdepth
 *
 */    
int hpgv_gl_fragdepth(int x, int y, float d)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    if (x < 0 && x >= theGLContext->framebuf_size_x) {
        return HPGV_FALSE;
    }
    
    if (y < 0 && y >= theGLContext->framebuf_size_y) {
        return HPGV_FALSE;
    }    
    
    float *depthbuf = NULL;
    
    if (theGLContext->use_depthbuf_bind) {
        depthbuf = theGLContext->depthbuf_bind;
    } else {
        depthbuf = theGLContext->depthbuf;
    }
    
    int offset = x + y * theGLContext->framebuf_size_x;
    
    if (theGLContext->use_depth_test) {
        if (depthbuf[offset] > d) {
            depthbuf[offset] = d;
            return HPGV_TRUE;
        } else {
            return HPGV_FALSE;
        }        
    } else {
        depthbuf[offset] = d;
        return HPGV_TRUE;
    }
    
    return HPGV_FALSE;
}



/**
 * hpgv_gl_bind_colorbuf
 *
 */        
int hpgv_gl_bind_colorbuf(void *colorbuf)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    theGLContext->colorbuf_bind = (hpgv_rgba_t *)colorbuf;
    
    return HPGV_TRUE;
}
    
    
/**
 * hpgv_gl_bind_depthbuf
 *
 */        
int hpgv_gl_bind_depthbuf(void *depthbuf)
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    theGLContext->depthbuf_bind = (float *)depthbuf;
    
    return HPGV_TRUE;

}
    
    
/**
 * hpgv_gl_enable
 *
 */        
int hpgv_gl_enable(int n) 
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    switch(n) {
    case HPGV_BIND_COLORBUF: theGLContext->use_colorbuf_bind = HPGV_TRUE;
        return HPGV_TRUE;
        break;
    case  HPGV_BIND_DEPTHBUF: theGLContext->use_depthbuf_bind = HPGV_TRUE;
        return HPGV_TRUE;
        break;
    case HPGV_DEPTH_TEST: theGLContext->use_depth_test = HPGV_TRUE;
        return HPGV_TRUE;
        break;
    }    
    
    return HPGV_FALSE;
}


/**
 * hpgv_gl_disable
 *
 */        
int hpgv_gl_disable(int n) 
{
    if (!hpgv_gl_valid()) {        
        return HPGV_FALSE;
    }
    
    switch(n) {
    case HPGV_BIND_COLORBUF: theGLContext->use_colorbuf_bind = HPGV_FALSE;
        return HPGV_TRUE;
        break;
    case  HPGV_BIND_DEPTHBUF: theGLContext->use_depthbuf_bind = HPGV_FALSE;
        return HPGV_TRUE;
        break;
    case HPGV_DEPTH_TEST: theGLContext->use_depth_test = HPGV_FALSE;
        return HPGV_TRUE;
        break;
    }    
    
    return HPGV_FALSE;
}


/**
 * hpgv_typesize
 *
 */
int hpgv_typesize(int type) {
    switch (type) {
        case HPGV_BYTE : return 1;
        case HPGV_UNSIGNED_BYTE : return 1;
        case HPGV_SHORT : return 2;
        case HPGV_UNSIGNED_SHORT : return 2;
        case HPGV_INT : return 4;
        case HPGV_UNSIGNED_INT : return 4;
        case HPGV_FLOAT : return 4;
        case HPGV_DOUBLE : return 8;
    }
    
    return 0;
}


/**
 * hpgv_formatsize
 *
 */
int hpgv_formatsize(int format)
{
    switch (format) {        
        case HPGV_RGB : return 3;
        case HPGV_RGBA : return 4;
        case HPGV_PIXEL : return sizeof(hpgv_pixel_t);
    }    
    return 0;    
}





/**
 * hpgv_vis_saveppm  
 *
 */
int hpgv_vis_saveppm(int width, int height, int format, int type, 
                     const void *pixels, char *filename)
{
    uint64_t i;
    uint8_t r = 0, g = 0, b = 0;
    float fr, fg, fb;
    int t, x, y;
    float    *fb_f = NULL;
    uint16_t *fb_ushort = NULL;
    uint8_t  *fb_uchar = NULL;
    
    if (format != HPGV_RGB && format != HPGV_RGBA) {
        fprintf(stderr, "Unsupported pixel format.\n");
        return HPGV_ERROR;
    }
    
    if (type != HPGV_FLOAT &&
        type != HPGV_UNSIGNED_SHORT &&
        type != HPGV_UNSIGNED_BYTE)
    {
        fprintf(stderr, "Unsupported pixel type.\n");
        return HPGV_ERROR;
    }
        
    FILE *fp;
    
    fp = fopen(filename, "w" );
    if (!fp) {
        fprintf(stderr, "Can not open file %s.\n", filename);
        return HPGV_ERROR;
    }
    
    fprintf(fp,"P6\n");    
    fprintf(fp,"%i %i\n", width, height);
    fprintf(fp,"255\n");
    fclose(fp);
    
    /* reopen in binary append mode */
    fp = fopen( filename, "ab" );  
    if (!fp) {
        fprintf(stderr, "Can not open file %s.\n", filename);
        return HPGV_ERROR;
    }
        
    t = hpgv_formatsize(format);
    
    if (type == HPGV_FLOAT) {
        fb_f = (float *)pixels;
    } else if (type == HPGV_UNSIGNED_SHORT) {
        fb_ushort = (uint16_t *)pixels;
    } else if (type == HPGV_UNSIGNED_BYTE) {
        fb_uchar = (uint8_t *) pixels;
    }

    for (y = height - 1; y >= 0; y--) {
        for (x = 0; x < width; x++) {
            i = x+ y * width;
                    
            if (type == HPGV_FLOAT) {
                
                fr = CLAMP(fb_f[i * t], 0, 1);
                fg = CLAMP(fb_f[i * t + 1], 0, 1);
                fb = CLAMP(fb_f[i * t + 2], 0, 1);

                r = (uint8_t)(fr * 0xFF);
                g = (uint8_t)(fg * 0xFF);
                b = (uint8_t)(fb * 0xFF);
                
            } else if (type == HPGV_UNSIGNED_SHORT) {
                r = (uint8_t)(fb_ushort[i * t] * 0xFF / 65535.0f );
                g = (uint8_t)(fb_ushort[i * t + 1] * 0xFF / 65535.0f);
                b = (uint8_t)(fb_ushort[i * t + 2] * 0xFF / 65535.0f);
            }else if (type == HPGV_UNSIGNED_BYTE) {
                r = fb_uchar[i * t];
                g = fb_uchar[i * t + 1];
                b = fb_uchar[i * t + 2];
            }
            
            fputc(r, fp);
            fputc(g, fp);
            fputc(b, fp);
        }
    }
    
    fclose(fp);
    
    return HPGV_SUCCESS;
}



/**
 * hpgv_vis_saveraw
 *
 */
int hpgv_vis_saveraw(int width, int height, int format, int type, int imgnum,
                     const void *pixels, char *filename)
{
    if (pixels == NULL) {
        fprintf(stderr, "Empty pixel data.\n");
        return HPGV_ERROR;
    }
    
    FILE *fp;
    
    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Can not open file %s.\n", filename);
        return HPGV_ERROR;
    }

    int realnum = imgnum;

    if (realnum < 1) {
        realnum = 1;
    }
    
    long size = width * height * hpgv_formatsize(format) * hpgv_typesize(type) *
                realnum;

    fwrite(&width,  sizeof(int), 1, fp);
    fwrite(&height, sizeof(int), 1, fp);
    fwrite(&format, sizeof(int), 1, fp);
    fwrite(&type,   sizeof(int), 1, fp);
    fwrite(&realnum, sizeof(int), 1, fp);
    fwrite(pixels, size, 1, fp);

    fclose(fp);

    return HPGV_TRUE;
}


/**
 * hpgv_vis_loadraw
 *
 */
void * hpgv_vis_loadraw(int *width, int *height, int *format, int *type,
                        int *imgnum, char *filename)
{
    FILE *fp;
    
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Can not open file %s.\n", filename);
        return NULL;
    }
    
    fread(width,  sizeof(int), 1, fp);
    fread(height, sizeof(int), 1, fp);
    fread(format, sizeof(int), 1, fp);
    fread(type,   sizeof(int), 1, fp);
    fread(imgnum, sizeof(int), 1, fp);

    long size = (*width) * (*height) *
                hpgv_formatsize(*format) *
                hpgv_typesize(*type) *
                (*imgnum);

    void *pixels = (void *)malloc(size);

    if (pixels == NULL) {
        fprintf(stderr, "Empty pixel data.\n");
        fclose(fp);
        return NULL;
    }
    
    
    fread(pixels, size, 1, fp);
    
    fclose(fp);
    
    return pixels;
}
}
