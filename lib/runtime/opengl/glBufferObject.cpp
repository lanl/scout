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
#include <iostream>
#include <cassert>
#include "glBufferObject.h"

using namespace std;
using namespace ogle;


// ---- glBufferObject
//
glBufferObject::glBufferObject(GLenum type)
{
  glGenBuffers(1, &_id);
  OpenGLErrorCheck();
  _bound         = false;
  _mapped        = false;  
  _type          = type;
  _size_in_bytes = 0;
}


// ---- ~glBufferObject
//
glBufferObject::~glBufferObject()
{
  if (glIsBuffer(_id)) {
    bind(); // Do we need to bind before a delete?
    glDeleteBuffers(1, &_id);
    _id = 0;
    release();
    OpenGLErrorCheck();
  }
}


// ---- alloc
//
void glBufferObject::alloc(size_t nbytes, GLenum mode, void *data_ptr)
{
  assert(nbytes != 0 && glIsBuffer(_id));
  glBufferData(_type, nbytes, data_ptr, mode);
  OpenGLErrorCheck();
  _size_in_bytes = nbytes;
}


// ---- write
//
void glBufferObject::write(void *data_ptr, size_t nbytes)
{
  bind();
  glBufferSubData(_type, 0, nbytes, data_ptr);
  OpenGLErrorCheck();
  release();
}

