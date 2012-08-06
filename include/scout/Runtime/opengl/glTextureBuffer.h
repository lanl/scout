/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#ifndef SCOUT_GL_TEXTURE_BUFFER_H_
#define SCOUT_GL_TEXTURE_BUFFER_H_

#include "scout/Runtime/opengl/glBufferObject.h"

namespace scout
{
  // ----- glTextureBuffer
  //
  class glTextureBuffer: public glBufferObject {
    
   public:
    glTextureBuffer()
        : glBufferObject(GL_PIXEL_UNPACK_BUFFER)
    { /* details handled by glBufferObject... */ }
    
    ~glTextureBuffer()
    { /* no-op -- handled by glBufferObject */ }

  };
}

#endif
