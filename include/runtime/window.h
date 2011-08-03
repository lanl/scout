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

#include "runtime/types.h"
#include "runtime/viewport.h"

namespace scout 
{

  // ----- window_rt
  //
  class window_rt {

   public:
  
    window_rt(dim_t w = 1024, dim_t h = 1024)
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
