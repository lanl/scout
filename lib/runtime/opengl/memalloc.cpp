/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#include "runtime/opengl/opengl.h"
#include "runtime/opengl/glTextureBuffer.h"

namespace scout 
{

  // ----- __sc_pbo_cast
  // Just a helper to clean up the code a tad. 
  static glTextureBuffer* __sc_pbo_cast(void* ptr)
  { return (ptr != 0 ? (glTextureBuffer*)ptr : 0); }

  
  // ----- __sc_pbo_malloc
  //
  void* __sc_pbo_malloc(size_t size)
  {
    glTextureBuffer* pbo = new glTextureBuffer;
    pbo->bind();
    pbo->alloc(size, GL_STREAM_DRAW);
    pbo->release();
    return (void*)pbo;
  }


  // ----- __sc_pbo_free
  //
  void __sc_pbo_free(void* ptr)
  {
    glTextureBuffer* pbo = __sc_pbo_cast(ptr);
    if (pbo != 0) {
      delete pbo;
    }
  }

  
  // ----- __sc_pbo_map
  //
  void* __sc_pbo_map(void* ptr)
  {
    glTextureBuffer* pbo = __sc_pbo_cast(ptr);
    if (pbo != 0) {
      pbo->bind();
      return (void*)pbo->mapForWrite();
    } else {
      return 0;
    }
  }

  
  // ----- __sc_pbo_unmap
  //
  void __sc_pbo_unmap(void* ptr)
  {
    glTextureBuffer* pbo = __sc_pbo_cast(ptr);
    if (pbo != 0) {
      pbo->unmap();
      pbo->release();
    }
  }
  
}


