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

#include "runtime/types.h"
#include "runtime/range.h"

namespace scout 
{
  
  struct tbq_params_rt{
    void* m;
  };

  class tbq_rt{
  public:
    tbq_rt();

    ~tbq_rt();

    //void run(void (^block)(index_t*,index_t*,index_t*,tbq_params_rt),
    //         range_t xRange, range_t yRange, range_t zRange);
    
  private:
    class tbq_rt_* x_;
  };
}

#endif
