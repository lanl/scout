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

#ifndef SCOUT_GL_VERTEX_SHADER_H_
#define SCOUT_GL_VERTEX_SHADER_H_

#include "scout/Runtime/opengl/glShader.h"

namespace scout
{

  // ..... glVertexShader
  //
  class glVertexShader: public glShader {

   public:
    glVertexShader()
        : glShader(GL_VERTEX_SHADER)
    { }

    glVertexShader(const std::string& shader_file)
        : glShader(shader_file, GL_VERTEX_SHADER)
    { }

    ~glVertexShader()
    { /* no-op -- handled by base class. */ }
    
  };
}

#endif

