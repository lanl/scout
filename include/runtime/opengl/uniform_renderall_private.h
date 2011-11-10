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

#ifndef SCOUT_UNIFORM_RENDERALL_PRIVATE_H_
#define SCOUT_UNIFORM_RENDERALL_PRIVATE_H_

#include "runtime/opengl/glTexture1D.h"
#include "runtime/opengl/glTexture2D.h"
#include "runtime/opengl/glTextureBuffer.h"
#include "runtime/opengl/glTexCoordBuffer.h"
#include "runtime/opengl/glVertexBuffer.h"

#ifdef SC_ENABLE_CUDA
#include <cuda.h>
#include <cuda_gl_interop.h>
#endif

namespace scout
{
  struct uniform_renderall_t {
    glVertexBuffer*   vbo;           // vertex buffer for mesh geometry (really a quad)
    glTexture*        tex;           // texture for storing colors computed by renderall
    glTextureBuffer*  pbo;           // buffer object for faster data transfers (for texture)    
    glTexCoordBuffer* tcbo;          // texture coordinate buffer object.
    unsigned short    ntexcoords;    // dimensions of texture coordinates (1,2,3).
    unsigned int      nverts;        // number of vertices stored in the vbo.
    
    //
    // TODO: need to add cuda interop support here...
    //
  };
}

#endif
