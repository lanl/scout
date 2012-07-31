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
#include "runtime/opengl/glTexture1D.h"

using namespace scout;


// ----- glTexture1D
//
glTexture1D::glTexture1D(GLsizei width) 
    : glTexture(GL_TEXTURE_1D, GL_RGBA32F_ARB, GL_RGBA, GL_FLOAT)
{
  _width   = width;
}


// ---- ~glTexture1D
//
glTexture1D::~glTexture1D()
{ /* no-op -- base class destroys */ }


// ---- initialize
//
void glTexture1D::initialize(const float* p_data)
{
  enable();
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  
  setParameters();
  glTexImage1D(target(), 0, internalFormat(), width(), 0, pixelFormat(), type(), (void*)p_data);
  OpenGLErrorCheck();
}


// ---- canDownload
//
bool glTexture1D::canDownload() const
{
  glTexImage1D(GL_PROXY_TEXTURE_1D, 0, internalFormat(), width(), 0, pixelFormat(), type(), 0);
  GLsizei w;
  glGetTexLevelParameteriv(GL_PROXY_TEXTURE_1D, 0, GL_TEXTURE_WIDTH, &w);
  OpenGLErrorCheck();      
  return(w != 0);
}

// ---- update 
// 
void glTexture1D::update(const float *p_data)
{
  glTexSubImage1D(target(), 0, 0, width(),internalFormat(), type(), (const void*)p_data);
  OpenGLErrorCheck();
}


// ---- update
//
void glTexture1D::update(const float* p_data, GLsizei offset, GLsizei subwidth)
{
  glTexSubImage1D(target(), 0, offset, subwidth, pixelFormat(), type(), (void*)p_data);
  OpenGLErrorCheck();
}


// ---- read
// Readback the texture data to address specified by 'p_data'.
void glTexture1D::read(float* p_data) const
{
  glGetTexImage(target(), 0, pixelFormat(), type(), p_data);
  OpenGLErrorCheck();
}
