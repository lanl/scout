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

namespace scout 
{
  extern struct uniform_renderall_t;

  uniform_renderall_t* __sc_init_uniform_renderall(dim_t xdim);
  uniform_renderall_t* __sc_init_uniform_renderall(dim_t xdim, dim_t ydim);
  uniform_renderall_t* __sc_init_uniform_renderall(dim_t xdim, dim_t ydim, dim_t zdim);

  float4* __sc_map_colors(uniform_renderall_t* ra_info);
  void    __sc_unmap_colors(uniform_renderall_t* ra_info);

  
}


