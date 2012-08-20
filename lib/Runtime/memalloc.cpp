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
#include <stdint.h>

#include "scout/Runtime/memalloc.h"

namespace scout 
{

  #define SC_PREFERRED_ALIGNMENT 64

  // make sure this size is a multiple of SC_PREFERRED_ALIGNMENT
  struct sc_array_header{
#if SC_PREFERRED_ALIGNMENT == 64
    uint64_t size;
#else
    uint32_t size;
#endif
  };  

  // ----- __sc_aligned_malloc
  // 
  void* __sc_aligned_malloc(size_t size)
  {
    assert(size != 0);

    void* a_ptr;
    int failure = posix_memalign(&a_ptr, SC_PREFERRED_ALIGNMENT, size);
    if (! failure)
      return a_ptr;
    else
      return 0;
  }

  // ----- __sc_aligned_array_malloc
  // 
  void* __sc_aligned_array_malloc(size_t size){
    void* a_ptr;
    int failure = posix_memalign(&a_ptr, SC_PREFERRED_ALIGNMENT,
				 sizeof(sc_array_header) + size);

    if(failure){
      return 0;
    }

    ((sc_array_header*)a_ptr)->size = size;
 
    return (char*)a_ptr + sizeof(sc_array_header);
  }

  // ----- __sc_aligned_array_size
  // 
  size_t __sc_aligned_array_size(void* a_ptr){
    sc_array_header* h  = 
      (sc_array_header*)((char*)a_ptr - sizeof(sc_array_header));
    return h->size;
  }   
 
  // ----- __sc_realloc
  //
  void* __sc_aligned_realloc(void* ptr, size_t size)
  {
    assert(ptr != 0);
    assert(size != 0);

    return realloc(ptr, size);
  }
  
    
  // ----- __sc_free
  //
  void __sc_aligned_free(void* a_ptr)
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
