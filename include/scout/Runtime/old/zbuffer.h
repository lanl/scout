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

#ifndef SCOUT_ZBUFFER_H_
#define SCOUT_ZBUFFER_H_

#include <float.h>
#include "scout/types.h"

namespace scout  
{
  
  struct zbuffer_rt {
    zbuffer_rt(dim_t w, dim_t h, float depth=FLT_MIN);
    ~zbuffer_rt();

    void clear();

    dim_t  width, height;
    float depth;
    float *values;
  };

}

#endif
