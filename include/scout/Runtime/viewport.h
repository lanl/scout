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

#ifndef SCOUT_VIEWPORT_H_
#define SCOUT_VIEWPORT_H_

namespace scout 
{
  
  struct viewport_rt {
  
    viewport_rt(float x = 0.0f, float y = 0.0f, float w = 1.0f, float h = 1.0f)
        : xpos(x), ypos(y), width(w), height(h) 
    { }
  
    ~viewport_rt()
    { /* currently a no-op */}

    float xpos, ypos;
    float width, height;
  };

  typedef viewport_rt* viewport_rt_p;
  
}

#endif


