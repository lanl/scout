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

#ifndef SCOUT_BASE_TYPES_H_
#define SCOUT_BASE_TYPES_H_

#include <stdint.h>

// Some shorthand for various types...
typedef unsigned char    uchar;
typedef unsigned char*   ucharp;  
typedef unsigned short   ushort;
typedef unsigned short*  ushortp;
typedef unsigned long    ulong;
typedef unsigned long*   ulongp;
  
  
// Use these types when supporting operations on mesh types.
// 
//    * rank_t   - the dimensionality of a mesh.
//    * dim_t    - the size of a particular mesh dimension.
//    * stride_t - stride in cells/edges/etc. through the mesh.
//    * index_t  - index value into a specific mesh location.
// 
typedef uint16_t   rank_t;
typedef uint32_t   dim_t;
typedef uint64_t   stride_t;
typedef uint64_t   index_t;

#endif
