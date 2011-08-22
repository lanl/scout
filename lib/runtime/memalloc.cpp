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
#include <stdio.h>
#include <cassert>
#include <stdlib.h>

#include "runtime/memalloc.h"

namespace scout 
{

  #define SC_PREFERRED_ALIGNMENT 64
  
  // ----- __sc_alloc
  // 
  void* __sc_alloc(size_t size)
  {
    assert(size != 0);

    void* a_ptr;
    int failure = posix_memalign(&a_ptr, SC_PREFERRED_ALIGNMENT, size);
    if (! failure)
      return a_ptr;
    else
      return 0;
  }

    
  // ----- __sc_realloc
  //
  void* __sc_realloc(void* ptr, size_t size)
  {
    assert(ptr != 0);
    assert(size != 0);

    return realloc(ptr, size);
  }
  
    
  // ----- __sc_free
  //
  void __sc_free(void* a_ptr)
  {
    free(a_ptr);
  }

}
  
/*
int main()
{
  using namespace scout;
  float *ptr = (float*)__sc_alloc(sizeof(float) * 16);
  if (__sc_is_ptr_aligned(ptr))
    printf("yep, looks happy.\n");
  else
    printf("nope, that is bad news.\n");

  __sc_free(ptr);
  
  return 0;
}
*/
