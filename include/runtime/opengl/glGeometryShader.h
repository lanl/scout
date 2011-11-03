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

#ifndef SCOUT_GEOMETRY_SHADER_H_
#define SCOUT_GEOMETRY_SHADER_H_

#include "runtime/opengl/glShader.h"

namespace scout
{

  // ..... glGeometryShader
  //
  class glGeometryShader: public glShader {

   public:
    glGeometryShader()
        : glShader(GL_GEOMETRY_SHADER_EXT)
    { };

    ~glGeometryShader()
    { /* no-op -- handled by base class. */ }
    
  };
}

#endif

