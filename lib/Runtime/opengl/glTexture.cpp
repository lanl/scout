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
#include "scout/Runtime/opengl/glTexture.h"

using namespace scout;

// ---- glTexture
//
glTexture::glTexture(GLenum target, GLenum iformat, GLenum format, GLenum type)
{
  _target   = target;
  _iformat  = iformat;
  _format   = format;
  _type     = type;
  _tex_unit = GL_TEXTURE0;
  
  glGenTextures(1, &_id);
  OpenGLErrorCheck();  
}


// ---- ~glTexture
//
glTexture::~glTexture()
{
  if (glIsTexture(_id))
    glDeleteTextures(1, &_id);
}


// ---- isResident
//
bool glTexture::isResident() const
{
  if (glIsTexture(_id)) {
    GLboolean dummy;
    return(glAreTexturesResident(1, &_id, &dummy) == GL_TRUE);
  } else {
    return false;
  }
}


// ---- setParameters
//
void glTexture::setParameters()
{
  glTexParameterList::iterator it  = _parameters.begin();
  glTexParameterList::iterator end = _parameters.end();
  while(it != end) {
    glTexParameteri(_target, (*it).name, (*it).param);
    ++it;
  }
}

