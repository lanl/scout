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

#ifndef SCOUT_GL_TEXTURE_PARAMETER_H_
#define SCOUT_GL_TEXTURE_PARAMETER_H_

#include <list>
#include "opengl.h"

namespace scout
{
  // ..... glTextureParameter
  // 
  struct glTextureParameter
  {
    GLenum name;
    GLint  param;
  };

  typedef std::list<glTextureParameter>  glTexParameterList;
}

#endif
