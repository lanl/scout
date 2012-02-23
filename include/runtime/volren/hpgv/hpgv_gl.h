/**
 * hpgv_gl.h
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

#ifndef HPGV_GL_H
#define HPGV_GL_H

#include <stdint.h>

namespace scout {

  extern "C" {

#if defined(__gl_h_) || defined(__GL_H__)  
#define HPGV_RGB GL_RGB
#define HPGV_RGBA GL_RGBA
#define HPGV_BYTE GL_BYTE
#define HPGV_UNSIGNED_BYTE GL_UNSIGNED_BYTE
#define HPGV_SHORT GL_SHORT
#define HPGV_UNSIGNED_SHORT GL_UNSIGNED_SHORT
#define HPGV_INT GL_INT
#define HPGV_UNSIGNED_INT GL_UNSIGNED_INT
#define HPGV_FLOAT GL_FLOAT
#define HPGV_DOUBLE GL_DOUBLE
#define HPGV_LIGHTING GL_LIGHTING
#define HPGV_DEPTH_TEST GL_DEPTH_TEST
#else        
#define HPGV_RGB 0x1907
#define HPGV_RGBA 0x1908
#define HPGV_BYTE 0x1400
#define HPGV_UNSIGNED_BYTE 0x1401
#define HPGV_SHORT 0x1402
#define HPGV_UNSIGNED_SHORT 0x1403
#define HPGV_INT 0x1404
#define HPGV_UNSIGNED_INT 0x1405
#define HPGV_FLOAT 0x1406
#define HPGV_DOUBLE 0x140A  
#define HPGV_LIGHTING 0x0B50 
#define HPGV_DEPTH_TEST 0x0B71    
#endif

#define HPGV_BIND_COLORBUF 0x10000
#define HPGV_BIND_DEPTHBUF 0x10001    


#define HPGV_PIXEL      0x2000

#define SET_USEFUL(v)       ((v) |= 0x04)    
#define SET_USELESS(v)      ((v) &= 0xFB)
#define SET_FRONTFACE(v)    ((v) |= 0x02)
#define SET_BACKFACE(v)     ((v) |= 0x01)
#define GET_USEFUL(v)       ((v) & 0x04)
#define GET_FACE(v)         ((v) & 0x03)
#define FACE_FRONT          (0x02)
#define FACE_BACK           (0x01)
#define FACE_BOTH           (0x03)
#define FACE_NONE           (0x00)

    /**
     * hpgv_pixel_t
     *
     */
    typedef struct hpgv_pixel_t {
      float red, green, blue, alpha;
      uint32_t offset;
      uint8_t flags;
    } hpgv_pixel_t;    


    /**
     * hpgv_gl_valid
     *
     */
    int hpgv_gl_valid();    

    /**
     * hpgv_gl_init
     *
     */
    void hpgv_gl_init();


    /**
     * hpgv_gl_finalize
     *
     */
    void hpgv_gl_finalize();  

    /**
     * hpgv_gl_viewmatrix
     *
     */
    int hpgv_gl_viewmatrix(const double m[16]);

    /**
     * hpgv_gl_projmatrix
     *
     */
    int hpgv_gl_projmatrix(const double m[16]);


    /**
     * hpgv_gl_viewport
     *
     */
    int hpgv_gl_viewport(int viewport[4]); 

    /**
     * hpgv_gl_framesize
     *
     */
    int hpgv_gl_framesize(int width, int height);    

    /**
     * hpgv_gl_project
     *
     */
    int hpgv_gl_project(const double objx, const double objy, const double objz,
        double *winx, double *winy, double *winz);

    /**
     * hpgv_gl_unproject
     *
     *
     */
    int hpgv_gl_unproject(const double winx, const double winy, const double winz,
        double *objx, double *objy, double *objz);


    /**
     * hpgv_gl_get_viewmatrix
     *
     */
    int hpgv_gl_get_viewmatrix(double m[16]);


    /**
     * hpgv_gl_get_viewmatrixinv
     *
     */
    int hpgv_gl_get_viewmatrixinv(double m[16]);


    /**
     * hpgv_gl_get_framesize
     *
     */    
    int hpgv_gl_get_framesize();    


    /**
     * hpgv_gl_get_framewidth
     *
     */    
    int hpgv_gl_get_framewidth();


    /**
     * hpgv_gl_get_frameheight
     *
     */    
    int hpgv_gl_get_frameheight();


    /**
     * hpgv_gl_get_depthptr
     *
     */
    void * hpgv_gl_get_depthptr(); 


    /**
     * hpgv_gl_get_cbptr
     *
     */
    float * hpgv_gl_get_cbptr();


    /**
     * hpgv_gl_get_cbtype
     *
     */
    int hpgv_gl_get_cbtype();


    /**
     * hpgv_gl_get_cbformat
     *
     */
    int hpgv_gl_get_cbformat();    


    /**
     * hpgv_gl_clear_color
     *
     */
    int hpgv_gl_clear_color(); 


    /**
     * hpgv_gl_clear_depth
     *
     */
    int hpgv_gl_clear_depth();



    /**
     * hpgv_gl_fragcolor
     *
     */    
    int hpgv_gl_fragcolor(int x, int y, float r, float g, float b, float a);


    /**
     * hpgv_gl_fragdepth
     *
     */    
    int hpgv_gl_fragdepth(int x, int y, float d);


    /**
     * hpgv_gl_bind_colorbuf
     *
     */        
    int hpgv_gl_bind_colorbuf(void *colorbuf);    


    /**
     * hpgv_gl_bind_depthbuf
     *
     */        
    int hpgv_gl_bind_depthbuf(void *depthbuf); 


    /**
     * hpgv_gl_enable
     *
     */        
    int hpgv_gl_enable(int n);  


    /**
     * hpgv_gl_disable
     *
     */        
    int hpgv_gl_disable(int n);    


    /**
     * hpgv_typesize
     *
     */    
    int hpgv_typesize(int type);

    /**
     * hpgv_formatsize
     *
     */    
    int hpgv_formatsize(int format);

    /**
     * hpgv_vis_saveppm  
     *
     */    
    int hpgv_vis_saveppm(int width, int height, int format, int type, 
        const void *pixels, char *filename);


    /**
     * hpgv_vis_saveraw
     *
     */
    int hpgv_vis_saveraw(int width, int heigh, int format, int type, int imgnum,
        const void *pixels, char *filename);


    /**
     * hpgv_vis_loadraw
     *
     */
    void *hpgv_vis_loadraw(int *width, int *height, int *format, int *type,
        int *imgnum, char *filename);


  }

} // end namespace scout

#endif
