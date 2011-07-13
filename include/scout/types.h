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

#ifndef SCOUT_TYPES_H_
#define SCOUT_TYPES_H_

#include <cassert>
#include <stdint.h>

namespace scout 
{
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

  // We pack mesh type detials into the bits of a unsigned integral value. 
  typedef uint64_t  typemask_t;

  enum mesh_mask_bits_t {
    InvalidMeshMask      = 0,
    UniformMeshMask      = 1UL << 1,
    RectilinearMeshMask  = 1UL << 2,
    StructuredMeshMask   = 1UL << 3,
    UnstructuredMeshMask = 1UL << 4,

    AllMeshMask          = UniformMeshMask      |
                           RectilinearMeshMask  |
                           StructuredMeshMask   |
                           UnstructuredMeshMask
  };

  inline bool isValidMesh(typemask_t mask)
  { return (mask & typemake_t(AllMeshMask)) != 0; }
  
  inline bool isUniformMesh(typemask_t mask)
  { return (mask & typemask_t(UniformMeshMask)) != 0; }

  inline bool isRectilinearMesh(typemask_t mask)
  { return (mask & typemask_t(RectilinearMeshMask)) != 0; }

  inline bool isStructuredMesh(typemask_t mask)
  { return (mask & typemask_t(StructuredMeshMask)) != 0; }
  
  inline bool isUnstructuredMesh(typemask_t mask)
  { return (mask & typemask_t(UnstructuredMeshMask)) != 0; }
  
}


#endif 



  
