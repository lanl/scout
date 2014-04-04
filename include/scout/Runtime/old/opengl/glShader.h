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

#ifndef SCOUT_GL_SHADER_H_
#define SCOUT_GL_SHADER_H_

#include <string>

#include "scout/Runtime/opengl/opengl.h"

namespace scout
{

  // ..... glShader
  // 
  class glShader {

   public:
    glShader(GLenum shader_type);
    glShader(const std::string& filename, GLenum shader_type);
    
    virtual ~glShader();

    GLuint id() const
    { return shader_id; }
    
    void setSource(const GLchar* src);
    
    void setSource(const std::string& src)
    { setSource(src.c_str());  }
    
    GLchar* source();
    
    GLint sourceLength() const;
    
    bool compile();

    bool isCompiled() const
    { return is_compiled; }
    
    const std::string& compileLog() const
    { return compile_log; }    

   private:
    GLuint       shader_id;
    bool         is_compiled;
    GLenum       shader_type;
    std::string  compile_log;
  };
}

#endif
