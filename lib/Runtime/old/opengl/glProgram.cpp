/*
 *  
 *###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 */ 

#include <cassert>
#include <iostream>

#include "scout/Runtime/opengl/glProgram.h"

using namespace std;
using namespace scout;


// ----- glProgram
//
glProgram::glProgram()
    :_frag_shader(0), _vert_shader(0), _geom_shader(0)
{
  _prog_id = glCreateProgram();
  assert(glIsProgram(_prog_id));
  oglErrorCheck();
  _is_linked = false;
}


// ----- ~glProgram
//
glProgram::~glProgram()
{
  if (glIsProgram(_prog_id)) {
    glDeleteProgram(_prog_id);
    oglErrorCheck();
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
  
