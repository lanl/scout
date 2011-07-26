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

#include "runtime/framebuffer.h"


namespace scout 
{
  // base class for windows. 
  class window_t {
    
   public:
    window_t(dim_t w, dim_t h)
    {
      _width   = w;
      _height  = h;
      _fbuffer = 0;  // create in an 'unbound' state.
    }

    virtual ~window_t()
    {
      // currently a no-op at this level
    }

    virtual void refresh() = 0;
    
    void bindFramebuffer(framebuffer_t *fb)
    {
      if (hasFramebuffer())
        releaseFramebuffer();
      _fbuffer = fb;
    }
    
    void releaseFramebuffer()
    {
      _fbuffer = 0; // simplistic approach at this point...
    }

    bool hasFramebuffer() const
    { return _fbuffer != 0; }

    dim_t             _width, _height;
    framebuffer_t    *_fbuffer;
  };
}

#endif



