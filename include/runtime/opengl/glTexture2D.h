/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *-----
 * 
 */

#ifndef OGLE_GL_TEXTURE_2D_H_
#define OGLE_GL_TEXTURE_2D_H_

#include "ogle.h"
#include "glTexture.h"

namespace ogle
{
  // ..... glTexture2D
  //
  template <typename T>
  class glTexture2D: public glTexture
  {
   public:

    // ---- glTexture2D
    //
    glTexture2D(GLsizei w,
                GLsizei h,
                GLenum iformat  = glTextureTraits<T>::iformat,
                GLenum format   = glTextureTraits<T>::format,
                GLenum type     = glTextureTraits<T>::type)
        : glTexture(GL_TEXTURE_2D, iformat, format, type)
    {
      _width   = w;
      _height  = h;
    }

    // ---- ~glTexture2D
    //
    ~glTexture2D()
    { /* no-op -- base class destroys */ }


    // ---- width
    //
    GLsizei width() const
    { return _width; }

    // ---- height
    GLsizei height() const
    { return _height; }
    
    
    // ---- initialize
    //
    void initialize(void *p_data)
    {
      enable();
      setParameters();
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 32.0f);
      glTexImage2D(target(),              // target texture type (GL_TEXTURE_2D)
                   0,                     // level of detail 
                   internalFormat(),      
                   width(),
                   height(),
                   0,                     // border width
                   pixelFormat(),
                   type(),
                   (void*)p_data);
      OpenGLErrorCheck();      
      disable();
    }


    // ---- canDownload
    //
    bool canDownload() 
    {
      glTexImage2D(GL_PROXY_TEXTURE_2D,
                   0, _iformat, width(), height(), 0, _format,_type, 0);
      GLsizei w;
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
      OpenGLErrorCheck();      
      return(w != 0);
    }


    // ---- update 
    // 
    void update(const T *p_data)
    {
      glTexSubImage2D(target(), 0, 0, 0, width(), height(),
                      _iformat, _type, (void*)p_data);
      OpenGLErrorCheck();
    }

    // ---- update
    //
    void update(const T *p_data, GLsizei x_offset, GLsizei y_offset,
                GLsizei subwidth, GLsizei subheight)
    {
      glTexSubImage2D(target(), 0, 0, 0, x_offset, y_offset,
                      subwidth, subheight, _format, _type, (void*)p_data);
      OpenGLErrorCheck();
    }

    // ---- read
    // Readback the texture data to address specified by 'p_data'.
    void read(void *p_data) const
    {
      glGetTexImage(target(), 0, _format, _type, p_data);
      OpenGLErrorCheck();
    }

   protected:
    GLsizei    _width;
    GLsizei    _height;    
  };

  typedef glTexture2D<unsigned char> glUCharTexture2D;
  typedef shared_pointer<glUCharTexture2D> pglUCharTexture2D;
  
  typedef glTexture2D<float>         glFloatTexture2D;
  typedef shared_pointer<glFloatTexture2D> pglFloatTexture2D;  
}

#endif
