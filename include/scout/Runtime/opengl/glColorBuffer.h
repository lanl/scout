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

#ifndef SCOUT_OPENGL_COLOR_BUFFER_H_
#define SCOUT_OPENGL_COLOR_BUFFER_H_

#include "runtime/opengl/glBufferObject.h"

namespace scout
{
  
  // ---- glColorBuffer
  //
  class glColorBuffer: public glBufferObject {
    
   public:
    glColorBuffer()
        : glBufferObject(GL_ARRAY_BUFFER)
    { /* everything else is handled by glBufferObject... */ }
    
    ~glColorBuffer()
    { /* no-op */ }

  };
}

#endif
