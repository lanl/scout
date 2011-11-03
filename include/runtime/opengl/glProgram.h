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

#ifndef SCOUT_GL_PROGRAM_H_
#define SCOUT_GL_PROGRAM_H_

#include <iostream>
#include <string>
#include <list>

#include "runtime/opengl/glVertexShader.h"
#include "runtime/opengl/glGeometryShader.h"
#include "runtime/opengl/glFragmentShader.h"
#include "runtime/opengl/glUniformValue.h"

namespace scout
{
  
  // ..... glProgram
  //
  class glProgram {

   public:
    glProgram();
    ~glProgram();

    GLuint id() const
    { return _prog_id; }

    bool link();
    
    bool isLinked()
    { return _is_linked; }
    
    void enable();
    void disable();

    void attachShader(glVertexShader* shader);
    void attachShader(glGeometryShader* shader);
    void attachShader(glFragmentShader* shader);

    bool hasAttachedShaders() const;

    template <typename ElementType>
    void bindUniformValue(const std::string& name, const ElementType* value)
    {
      assert(glIsProgram(_prog_id) && _is_linked == true);
      
      GLint uloc = glGetUniformLocation(_prog_id, (const GLchar*)name.c_str());
      OpenGLErrorCheck();
      if (uloc == -1) {
        std::cerr << "\t***warning: unable to locate uniform value '"
                  << name << "'.\n";
      } else {
        glUniformValue* uval = new glTypedUniformValue<ElementType>(uloc,
                                                                    value);
        OpenGLErrorCheck();        
        _uniforms.push_back(uval);
      }
    }

    const std::string& linkLog() const
    { return _log; }
    
    
    int numberOfUniformValues() const;

   private:
    typedef std::list<glUniformValue*>  glUniformList;
    
    bool compileAllShaders();
    bool compileShader(glShader* shader);

    GLuint            _prog_id;
    bool              _is_linked;
    glVertexShader*   _vert_shader;
    glGeometryShader* _geom_shader;
    glFragmentShader* _frag_shader;
    glUniformList     _uniforms;
    std::string       _log;
  };
}

#endif
