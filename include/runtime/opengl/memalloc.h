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

#ifndef SCOUT_OPENGL_MEMALLOC_H_
#define SCOUT_OPENGL_MEMALLOC_H_

namespace scout 
{
  extern void* __sc_pbo_malloc(size_t size);
  extern void  __sc_pbo_free(void* ptr);
  extern void* __sc_pbo_map(void* ptr);
  extern void  __sc_pbo_unmap(void* ptr);
}

#endif
