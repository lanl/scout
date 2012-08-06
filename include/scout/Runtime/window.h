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

#ifndef SCOUT_WINDOW_H_
#define SCOUT_WINDOW_H_

#include <list>

#include "scout/Runtime/types.h"
#include "scout/Runtime/viewport.h"
#include "scout/Runtime/vec_types.h"

namespace scout 
{

  // ----- window_rt
  //
  class window_rt {

   public:
 
    // r, g, b and a are for background color 
    window_rt(dim_t w, dim_t h, float bg_r, float bg_g, float bg_b, float bg_a,
	      bool saveFrames, const char* filename)
        : width(w), height(h)
    { }

    virtual ~window_rt();

    void add_viewport(viewport_rt_p vp);

    void validate_viewport(viewport_rt_p vp);

    typedef std::list<viewport_rt_p> viewport_list;  

    dim_t          width, height;
    viewport_list  viewports;
  };
}

#endif
