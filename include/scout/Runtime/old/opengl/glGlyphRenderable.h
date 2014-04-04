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

#ifndef GL_GLYPH_RENDERABLE_H_
#define GL_GLYPH_RENDERABLE_H_

#include <cassert>
#include <string>
#include <list>

#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/opengl/glRenderable.h"
#include "scout/Runtime/opengl/glAttributeBuffer.h"
#include "scout/Runtime/opengl/glyph_vertex.h"


// ----- glGlyphRenderable 
// 

namespace scout
{

  //#include "sphere_cast_vs.h"
  //#include "sphere_cast_fs.h"

  class glGlyphRenderable: public glRenderable {

    public:

      glGlyphRenderable(size_t npoints);

      ~glGlyphRenderable();

      void initialize(glCamera* camera);

      void draw(glCamera* camera);

      GLuint get_buffer_object_id() { return _abo->id(); } 

      glyph_vertex* map_vertex_data() 
      { return (glyph_vertex*)_abo->mapForWrite(); }

      void unmap_vertex_data() { _abo->unmap(); }

    private:
      void loadShaders(const glCamera* camera);
      void allocateBuffer();

    private:

      size_t                  _npoints;
      glAttributeBuffer      *_abo;      // attributes

  };

}

#endif
