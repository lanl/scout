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

#ifndef SCOUT_GL_TEXTURE_1D_H_
#define SCOUT_GL_TEXTURE_1D_H_

#include "glTexture.h"

namespace scout
{
  
  // ..... glTexture1D
  //
  template <typename T>
  class glTexture1D: public glTexture
  {
   public:

    // ---- glTexture1D
    //
    glTexture1D(GLsizei width, 
                GLenum iformat = glTextureTraits<T>::iformat,
                GLenum format  = glTextureTraits<T>::format,
                GLenum type    = glTextureTraits<T>::type)
        : glTexture(GL_TEXTURE_1D, iformat, format, type)
    {
      _width   = width;
      _iformat = iformat;
    }

    // ---- ~glTexture1D
    //
    ~glTexture1D()
    { /* no-op -- base class destroys */ }


    // ---- width
    //
    GLsizei width() const
    { return _width; }
    
    // ---- initialize
    //
    void initialize(void *p_data)
    {
      enable();
      setParameters();
      glTexImage1D(target(),              // target texture type (GL_TEXTURE_1D)
                   0,                     // level of detail 
                   internalFormat(),      
                   width(),
                   0,                     // border width
                   pixelFormat(),
                   type(),
                   (void*)p_data);
      OpenGLErrorCheck();
    }


    // ---- canDownload
    //
    bool canDownload() 
    {
      glTexImage1D(GL_PROXY_TEXTURE_1D, 0, _iformat, width(), 0, _format,_type, 0);
      GLsizei w;
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_1D, 0, GL_TEXTURE_WIDTH, &w);
      OpenGLErrorCheck();      
      return(w != 0);
    }


    // ---- update 
    // 
    void update(const T *p_data)
    {
      glTexSubImage1D(target(), 0, 0, width(), _iformat, _type, (void*)p_data);
      OpenGLErrorCheck();
    }

    // ---- update
    //
    void update(const T *p_data, GLsizei offset, GLsizei subwidth)
    {
      glTexSubImage1D(target(), 0, offset, subwidth,
                      _format, _type, (void*)p_data);
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
  };

}

#endif
