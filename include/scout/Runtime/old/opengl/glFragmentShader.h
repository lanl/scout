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

#ifndef SCOUT_GL_FRAGMENT_SHADER_H_
#define SCOUT_GL_FRAGMENT_SHADER_H_

#include "scout/Runtime/opengl/glShader.h"

namespace scout
{

  // ..... glFragmentShader
  //
  class glFragmentShader: public glShader {

   public:
    
    glFragmentShader()
        : glShader(GL_FRAGMENT_SHADER)
    { };

    glFragmentShader(const std::string& shader_file)
        : glShader(shader_file, GL_FRAGMENT_SHADER)
    { };
    
    ~glFragmentShader()
    { /* no-op -- handled by base class. */ }
    
  };
}

#endif

