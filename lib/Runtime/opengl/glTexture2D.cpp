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

#include "scout/Runtime/opengl/glTexture2D.h"

using namespace scout;


// ..... glTexture2D
//
glTexture2D::glTexture2D(GLsizei w, GLsizei h)
    : glTexture(GL_TEXTURE_2D, SC_GL_IFORMAT, GL_RGBA, GL_FLOAT)
{
  _width   = w;
  _height  = h;
}


// ---- ~glTexture2D
//
glTexture2D::~glTexture2D()
{ /* no-op -- base class destroys */ }


// ---- initialize
//
void glTexture2D::initialize(const float* p_data)
{
  enable();
  setParameters();
  //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 32.0f);
  
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(target(), 0, internalFormat(), width(), height(), 0, pixelFormat(), type(), (void*)p_data);
  oglErrorCheck();      
}


// ---- canDownload
//
bool glTexture2D::canDownload() const 
{
  glTexImage2D(GL_PROXY_TEXTURE_2D, 0, internalFormat(), width(), height(), 0, pixelFormat(), type(), 0);
  GLsizei w;
  glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
  oglErrorCheck();      
  return(w != 0);
}


// ---- update 
// 
void glTexture2D::update(const float* p_data)
{
  glTexSubImage2D(target(), 0, 0, 0, width(), height(), internalFormat(), type(), (void*)p_data);
  oglErrorCheck();
}


// ---- update
//
void glTexture2D::update(const float* p_data, GLsizei x_offset, GLsizei y_offset, GLsizei subwidth, GLsizei subheight)
{
  glTexSubImage2D(target(), 0, x_offset, y_offset, subwidth, subheight, pixelFormat(), type(), (void*)p_data);
  oglErrorCheck();
}


// ---- read
// Readback the texture data to address specified by 'p_data'.
void glTexture2D::read(float* p_data) const
{
  glGetTexImage(target(), 0, _format, _type, p_data);
  oglErrorCheck();
}
