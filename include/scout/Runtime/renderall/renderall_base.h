/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * -----
 * 
 */

#ifndef SCOUT_RENDERALL_BASE_H_
#define SCOUT_RENDERALL_BASE_H_

#include <cstdlib>

namespace scout{

  class renderall_base_rt{
  public:
    renderall_base_rt(size_t width, size_t height, size_t depth);

    virtual ~renderall_base_rt();

    virtual void begin() = 0;

    virtual void end() = 0;

    virtual void addVolume(void* dataptr, unsigned volumenum) = 0;

    size_t width(){
      return width_;
    }

    size_t height(){
      return height_;
    }

    size_t depth(){
      return depth_;
    }

  private:
    size_t width_;
    size_t height_;
    size_t depth_;
  };
  
} // end namespace scout

extern scout::renderall_base_rt* __sc_renderall;

extern "C" void __sc_begin_renderall();
extern "C" void __sc_end_renderall();
extern "C" void __sc_delete_renderall();

#endif // SCOUT_RENDERALL_BASE_H_

