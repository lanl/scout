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

#ifndef SCOUT_GLYPH_RENDERALL_H_
#define SCOUT_GLYPH_RENDERALL_H_

#include "runtime/base_types.h"
#include "runtime/vec_types.h"
#include "runtime/renderall_base.h"

namespace scout 
{
  class glCamera;
  class glGlyphRenderable;

  class glyph_renderall : public renderall_base_rt {
    public:
      glyph_renderall(size_t width, size_t height, size_t depth, size_t npts,
          glCamera* camera);
      ~glyph_renderall();
      void begin();
      void end();
    private:
      void map_gpu_resources();
      void unmap_gpu_resources();
      void exec();

    private:
      glGlyphRenderable* _renderable;
      glCamera* _camera;
  };

} // end namespace scout

using namespace scout;

extern void __sc_init_glyph_renderall(size_t width, size_t height, 
    size_t depth, size_t npoints, glCamera* camera = NULL);

#endif 
