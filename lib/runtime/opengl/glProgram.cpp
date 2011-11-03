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

#include <cassert>
#include <iostream>

#include "runtime/opengl/glProgram.h"

using namespace std;
using namespace scout;


// ----- glProgram
//
glProgram::glProgram()
{
  _prog_id = glCreateProgram();
  assert(glIsProgram(_prog_id));
  OpenGLErrorCheck();
  _is_linked = false;
}


// ----- ~glProgram
//
glProgram::~glProgram()
{
  if (glIsProgram(_prog_id)) {
    glDeleteProgram(_prog_id);
    OpenGLErrorCheck();
  }

  glUniformList::iterator it = _uniforms.begin(), end = _uniforms.end();
  while(it != end) {
    delete *it;
    ++it;
  }
  
}


// ----- hasAttachedShaders
//
bool glProgram::hasAttachedShaders() const
{ return (_vert_shader != 0 || _geom_shader != 0 || _frag_shader != 0); }


// ----- link
//
bool glProgram::link()
{
  assert(glIsProgram(_prog_id));

  if (hasAttachedShaders() && _is_linked == false) {

    // Check to make sure all shaders are compiled...
    if (compileAllShaders() == false) {
      return false;
    }
    
    glLinkProgram(_prog_id);

    GLint link_status;
    glGetProgramiv(_prog_id, GL_LINK_STATUS, &link_status);
    if (link_status == GL_TRUE) {
      _log.clear();
      _is_linked = true;
    } else {
      // Errors occurred during link.
      GLint   log_length;
      GLchar* char_log;

      glGetProgramiv(_prog_id, GL_INFO_LOG_LENGTH, &log_length);
      if (log_length > 0) {
        char_log = new GLchar[log_length];
        glGetProgramInfoLog(_prog_id, log_length, &log_length, char_log);
      
        _log.assign(char_log);
        delete []char_log;
      }
      
      _is_linked = false;
    }
  }

  return _is_linked;
}


// ----- enable
//
void glProgram::enable()
{
  assert(glIsProgram(_prog_id));
  glUseProgram(_prog_id);

  glUniformList::iterator it = _uniforms.begin(), end = _uniforms.end();
  while(it != end) {
    (*it)->bind();
    ++it;
  }
}


// ----- disable
//
void glProgram::disable()
{ glUseProgram(0); }



// ---- attachShader
//
void glProgram::attachShader(glVertexShader* shader)
{
  if (_vert_shader != 0) 
    glDetachShader(_prog_id, _vert_shader->id());

  _is_linked = false;
  _vert_shader = shader;
  glAttachShader(_prog_id, _vert_shader->id());
}

    
// ---- attachShader
//
void glProgram::attachShader(glGeometryShader* shader)
{
  if (_geom_shader != 0) 
    glDetachShader(_prog_id, _geom_shader->id());

  _is_linked = false;
  _geom_shader = shader;
  glAttachShader(_prog_id, _geom_shader->id());
}


// ---- attachShader
//
void glProgram::attachShader(glFragmentShader* shader)
{
  if (_frag_shader != 0) 
    glDetachShader(_prog_id, _frag_shader->id());

  _is_linked = false;
  _frag_shader = shader;
  glAttachShader(_prog_id, _frag_shader->id());
}


// ----- numberOfUniformValues
//
int glProgram::numberOfUniformValues() const
{
  assert(glIsProgram(_prog_id));
  int count;
  glGetProgramiv(_prog_id, GL_ACTIVE_UNIFORMS, &count);
  return count;
}


// ----- compileAllShaders
//
bool glProgram::compileAllShaders()
{
  if (_vert_shader != 0 && compileShader(_vert_shader) == false)
    return false;

  if (_geom_shader != 0 && compileShader(_geom_shader) == false)
    return false;

  if (_frag_shader != 0 && compileShader(_frag_shader) == false)
    return false;

  return true;
}


// ----- compileShader
//
bool glProgram::compileShader(glShader* shader)
{
  assert(shader != 0);
  
  if (shader->compile() == false) {
    _log = "Error compiling shader.\n\n" +
      std::string(shader->source()) + "\n\n***** Errors:\n\n";
    _log += shader->compileLog();
    return false;
  } else {
    return true;
  }
}
  
