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
  
  // ----- __sc_is_ptr_aligned
  //
  inline bool __sc_is_ptr_aligned(void* a_ptr)
  { return (((unsigned long)a_ptr) & 15) == 0; }

  extern void* __sc_alloc(size_t size);
  extern void* __sc_realloc(void* ptr, size_t size);
  extern void  __sc_free(void* a_ptr);
}

#endif
