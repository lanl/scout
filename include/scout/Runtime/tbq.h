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

#ifndef SCOUT_TBQ_H_
#define SCOUT_TBQ_H_

#include <list>

#include "scout/Runtime/types.h"
#include "scout/Runtime/range.h"

namespace scout 
{
  
  class tbq_rt{
  public:
    tbq_rt();

    ~tbq_rt();

    void run(void* blockLiteral, int numDimensions, int numFields);

  private:
    class tbq_rt_* x_;
  };
}

#endif
