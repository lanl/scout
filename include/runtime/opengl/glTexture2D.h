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

#ifndef SCOUT_OPENGL_TEXTURE_2D_H_
#define SCOUT_OPENGL_TEXTURE_2D_H_

#include "runtime/opengl/glTexture.h"

namespace scout
{
  
  // ..... glTexture2D
  //
  class glTexture2D: public glTexture
  {
   public:
    glTexture2D(GLsizei w, GLsizei h);
    ~glTexture2D();
      
    GLsizei width() const
    { return _width; }

    GLsizei height() const
    { return _height; }
    
    
    void initialize(const float *p_data);
    bool canDownload() const;

    void update(const float* p_data);
    void update(const float* p_data, GLsizei x_offset, GLsizei y_offset, GLsizei subwidth, GLsizei subheight);
    void read(float *p_data) const;

   protected:
    GLsizei    _width;
    GLsizei    _height;    
  };
}

#endif
