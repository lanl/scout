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

#ifndef SCOUT_OPENGL_TEXTURE_1D_H_
#define SCOUT_OPENGL_TEXTURE_1D_H_

#include "scout/Runtime/opengl/glTexture.h"

namespace scout
{
  
  // ..... glTexture1D
  //
  class glTexture1D: public glTexture
  {
   public:
    glTexture1D(GLsizei width);
    ~glTexture1D();

    GLsizei width() const
    { return _width; }

    void initialize(const float* p_data);

    bool canDownload() const;

    void update(const float *p_data);
    void update(const float *p_data, GLsizei offset, GLsizei subwidth);

    void read(float *p_data) const;
    
   protected:
    GLsizei    _width;
  };
  
}

#endif
