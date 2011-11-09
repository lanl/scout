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

#ifndef SCOUT_UNIFORM_RENDERALL_H_
#define SCOUT_UNIFORM_RENDERALL_H_

#include "runtime/base_types.h"
#include "runtime/vec_types.h"

namespace scout 
{
  struct uniform_renderall_t;

  uniform_renderall_t* __sc_init_uniform_renderall(dim_t xdim);

  uniform_renderall_t* __sc_init_uniform_renderall(dim_t xdim, dim_t ydim);

  float4* __sc_map_uniform_colors(uniform_renderall_t* info);

  void __sc_unmap_uniform_colors(uniform_renderall_t* info);

  void __sc_exec_uniform_renderall(uniform_renderall_t* info);

  void __sc_destroy_uniform_renderall(uniform_renderall_t* info);
}

#endif
