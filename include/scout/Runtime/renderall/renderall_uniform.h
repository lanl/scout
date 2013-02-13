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

#ifndef SCOUT_RENDERALL_UNIFORM_H_
#define SCOUT_RENDERALL_UNIFORM_H_

#include <cstdlib>

#include "scout/Runtime/renderall/renderall_base.h"
#include "scout/Runtime/vec_types.h"

namespace scout{

  class renderall_uniform_rt : public renderall_base_rt {
  public:
    renderall_uniform_rt(size_t width, size_t height, size_t depth);

    ~renderall_uniform_rt();

    void begin();

    void end();

    void addVolume(void* dataptr, unsigned volumenum){}

  private:
    class renderall_uniform_rt_* x_;
  };

} // end namespace scout

extern void __sc_begin_uniform_renderall(size_t width,
					 size_t height,
					 size_t depth);

#endif // SCOUT_RENDERALL_UNIFORM_H_

