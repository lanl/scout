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

#ifndef SCOUT_ALLOC_H_
#define SCOUT_ALLOC_H_

namespace scout 
{
  extern void* __sc_aligned_malloc(size_t size);
  extern void* __sc_aligned_realloc(void* ptr, size_t size);
  extern void  __sc_aligned_free(void* a_ptr);
  extern void* __sc_aligned_array_malloc(size_t size);
  extern size_t __sc_aligned_array_size(void* a_ptr);
  
  inline bool __sc_is_ptr_aligned(void* a_ptr)
  { return (((unsigned long)a_ptr) & 15) == 0; }
}


#endif
