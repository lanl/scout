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

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/types.h"
#include "runtime/opengl/glTexture1D.h"
#include "runtime/opengl/glTexture2D.h"
#include "runtime/opengl/glTextureBuffer.h"
#include "runtime/opengl/glTexCoordBuffer.h"
#include "runtime/opengl/glVertexBuffer.h"
namespace scout 
{
  #include "sphere_cast_vs.h"
  #include "sphere_cast_fs.h"

  struct glyph_renderall_t {
    glVertexBuffer*   vbo;       // vertex buffer for glyph locations (points)
    unsigned int      npoints;   // number of vertices stored in the vbo. 
  };
}



