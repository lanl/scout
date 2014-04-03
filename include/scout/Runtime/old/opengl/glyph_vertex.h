 
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

#ifndef SCOUT_GLYPH_VERTEX_H_
#define SCOUT_GLYPH_VERTEX_H_

namespace scout 
{
  struct glyph_vertex {
    float x, y, z;
    float radius;
    float r, g, b, a;
  };

} // end namespace scout

#endif 
