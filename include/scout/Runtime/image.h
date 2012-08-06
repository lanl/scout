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

#ifndef SCOUT_IMAGE_H_
#define SCOUT_IMAGE_H_

#include <list>

#include "scout/Runtime/types.h"

namespace scout 
{

  // ----- image_rt
  //
  class image_rt {

   public:
  
    image_rt(dim_t w, dim_t h, float4 bgColor, 
	     const char* filename)
        : width(w), height(h)
    { }

    virtual ~image_rt();

    dim_t          width, height;
  };
}

#endif
