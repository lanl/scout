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

#ifndef SCOUT_MESHES_H_
#define SCOUT_MESHES_H_

#include "scout/Runtime/base_types.h"

namespace scout
{
  typedef void* geom_t;
  
  struct mesh_info_t {
    rank_t    rank;     // mesh rank.
    dim_t     dims[3];  // mesh dimensions.
    geom_t    geom;     // cached geometry of mesh.
  };

  
}

#endif

