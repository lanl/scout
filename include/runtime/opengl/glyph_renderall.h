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

namespace scout 
{
  struct glyph_renderall_t;

  glyph_renderall_t* __sc_init_glyph_renderall(dim_t n_glyphs);

  float4* __sc_map_glyph_colors(glyph_renderall_t* info);

  void __sc_unmap_glyph_colors(glyph_renderall_t* info);

  float* __sc_map_glyph_positions(glyph_renderall_t* info);

  void __sc_unmap_glyph_positions(glyph_renderall_t* info);

  void __sc_exec_glyph_renderall(glyph_renderall_t* info);

  void __sc_destroy_glyph_renderall(glyph_renderall_t* info);
}

#endif 
