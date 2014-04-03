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
#include <fstream>
#include <string.h>

using namespace std;

#include "scout/Runtime/opengl/glShader.h"

using namespace scout;


// ----- glShader
//
glShader::glShader(GLenum type)
{
  shader_type = type;
  shader_id = glCreateShader(shader_type);
  assert(glIsShader(shader_id));
  
  oglErrorCheck();
  
  is_compiled = false;
}


// ----- glShader
//
glShader::glShader(const std::string& filename, GLenum type)
{
  shader_type = type;
  shader_id = glCreateShader(shader_type);
  assert(glIsShader(shader_id));
  
  oglErrorCheck();
  
  is_compiled = false;

  if (glIsShader(shader_id)) {
    
    ifstream is;
    is.open(filename.c_str(), ios::binary);
    if (is.is_open()) {
      is.seekg(0, ios::end);
      int length = is.tellg();
      is.seekg(0, ios::beg);

      GLchar* src = new GLchar[length+1];
      is.read(src, length-1);
      src[length-1] = '\0';
      is.close();

      setSource(src);

      delete []src;
    } else {
      cerr << "\t***warning: Unable to open shader source '"
           << filename << "'.\n";
    }
  }
}



// ----- ~glShader
//
glShader::~glShader()
{
  if (glIsShader(shader_id)) {
    glDeleteShader(shader_id);
    oglErrorCheck();
  }
}


// ----- setSource
//
void glShader::setSource(const GLchar* src)
{
  assert(glIsShader(shader_id));
  glShaderSource(shader_id, 1, &src, NULL);
  is_compiled = false;
}


// ----- source
//
GLchar* glShader::source()
{
  if (glIsShader(shader_id) == GL_FALSE)
    return NULL;
  
  GLint length = sourceLength();
  
  if (length > 0) {
    GLchar* src = new GLchar[length+1];
    src[length] = '\0';
    glGetShaderSource(shader_id, length, &length, src);
    return src;
  } else {
    return NULL;
  }
}


// ----- sourceLength
//
GLint glShader::sourceLength() const
{
  assert(glIsShader(shader_id));
  
  GLint length;
  glGetShaderiv(shader_id, GL_SHADER_SOURCE_LENGTH, &length);
  return length;
}


// ----- compile
//
bool glShader::compile()
{
  assert(glIsShader(shader_id));
  
  if (sourceLength() == 0 || is_compiled == true) {
    // If we have no source to compile, or the shader has    
    // already been compiled, we're done...
    is_compiled = true;
  } else {

    glCompileShader(shader_id);
    GLint was_compiled;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &was_compiled);

    if (was_compiled == GL_FALSE) {
      GLint   log_length;
      GLchar* log;

      glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);
      log = new GLchar[log_length];
      glGetShaderInfoLog(shader_id, log_length, &log_length, log);
    
      compile_log.assign(log);
      delete []log;
    
      is_compiled = false;
    
    } else {
      compile_log.clear();
      is_compiled = true;
    }
  }
  
  return is_compiled;
}

