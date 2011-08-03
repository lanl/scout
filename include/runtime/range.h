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

#ifndef SCOUT_RANGE_H_
#define SCOUT_RANGE_H_

#include "scout/types.h"

namespace scout 
{

  // ----- range_t
  //
  // Range types are a simple struct that store the starting, ending and
  // stride values for a range of integral values.
  //
  struct range_t {
    index_t  lower_bound, upper_bound;
    index_t  stride;
  };
  
}

#endif
