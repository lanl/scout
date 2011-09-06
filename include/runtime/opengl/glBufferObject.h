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

#ifndef SCOUT_GL_BUFFER_OBJECT_H_
#define SCOUT_GL_BUFFER_OBJECT_H_

#include "opengl.h"

namespace scout
{
  
  // ---- glBufferObject 
  //
  class glBufferObject {
    
   public:
    
    glBufferObject(GLenum type);
    virtual ~glBufferObject();

    GLuint id() const
    { return _id; }

    void alloc(size_t nbytes, GLenum mode, void* data_ptr = 0);
    void write(void *data, size_t nbytes);

    void* mapForWrite()
    {
      assert(_mapped == false);
      bind();
      void *vp = glMapBuffer(_type, GL_WRITE_ONLY);
      OpenGLErrorCheck();
      _mapped = true;
      return vp;
    }
    
    void* mapForRead()
    {
      assert(_mapped == false);
      bind();
      void *vp = glMapBuffer(_type, GL_READ_ONLY);
      OpenGLErrorCheck();
      _mapped = true;
      return vp;
    }

    void* mapForReadWrite()
    {
      assert(_mapped == false);
      bind();
      void *vp = glMapBuffer(_type, GL_READ_WRITE);
      OpenGLErrorCheck();
      _mapped = true;
      return vp;
    }

    void unmap()
    {
      bind();
      assert(_mapped == true);
      glUnmapBuffer(_type);
      OpenGLErrorCheck();
      _mapped = false;
      release();
    }

    void bind()
    {
      glBindBuffer(_type, _id);
      _bound = true;
    }
    
    void release()
    {
      glBindBuffer(_type, 0);
      _bound = false;
    }
    
   private:
    bool        _bound, _mapped;
    GLuint      _id;
    GLenum      _type;    
    size_t      _size_in_bytes;
  };
}

#endif
