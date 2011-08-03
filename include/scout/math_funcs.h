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

#ifndef SCOUT_MATH_FUNCS_H_
#define SCOUT_MATH_FUNCS_H_

namespace scout 
{
  
  template <typename T>
  T clamp(T val, T lower, T upper)
  {
    T retval;
    if (val < lower)
      retval = lower;
    else if (val > upper)
      retval = upper;
    else
      retval = val;
  
    return retval;
  }
  
}

#endif
