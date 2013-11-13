/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 */

#ifndef __SCOUT_TYPES_SCH__
#define __SCOUT_TYPES_SCH__

#include <stdint.h>

// The language has built-in vector types that range in size from 2-
// to 16-components in width.  These are actually not implemented within
// the additions we've made within Clang; instead they are supported by
// Clang's vector and extended vector extensions.  See:
//
//   http://clang.llvm.org/docs/LanguageExtensions.html
//

// +--- Support for 2,4,8,16 width vector types -----------------------------+
typedef bool bool16                  __attribute__((ext_vector_type(16)));
typedef bool bool8                   __attribute__((ext_vector_type(8)));
typedef bool bool4                   __attribute__((ext_vector_type(4)));
typedef bool bool3                   __attribute__((ext_vector_type(3)));
typedef bool bool2                   __attribute__((ext_vector_type(2)));
typedef char char16                  __attribute__((ext_vector_type(16)));
typedef char char8                   __attribute__((ext_vector_type(8)));
typedef char char4                   __attribute__((ext_vector_type(4)));
typedef char char3                   __attribute__((ext_vector_type(3)));
typedef char char2                   __attribute__((ext_vector_type(2)));
typedef unsigned char uchar16        __attribute__((ext_vector_type(16)));
typedef unsigned char uchar8         __attribute__((ext_vector_type(8)));
typedef unsigned char uchar4         __attribute__((ext_vector_type(4)));
typedef unsigned char uchar3         __attribute__((ext_vector_type(3)));
typedef unsigned char uchar2         __attribute__((ext_vector_type(2)));
typedef short short16                __attribute__((ext_vector_type(16)));
typedef short short8                 __attribute__((ext_vector_type(8)));
typedef short short4                 __attribute__((ext_vector_type(4)));
typedef short short3                 __attribute__((ext_vector_type(3)));
typedef short short2                 __attribute__((ext_vector_type(2)));
typedef unsigned short ushort16      __attribute__((ext_vector_type(16)));
typedef unsigned short ushort8       __attribute__((ext_vector_type(8)));
typedef unsigned short ushort4       __attribute__((ext_vector_type(4)));
typedef unsigned short ushort3       __attribute__((ext_vector_type(3)));
typedef unsigned short ushort2       __attribute__((ext_vector_type(2)));
typedef int int16                    __attribute__((ext_vector_type(16)));
typedef int int8                     __attribute__((ext_vector_type(8)));
typedef int int4                     __attribute__((ext_vector_type(4)));
typedef int int3                     __attribute__((ext_vector_type(3)));
typedef int int2                     __attribute__((ext_vector_type(2)));
typedef unsigned int uint16          __attribute__((ext_vector_type(16)));
typedef unsigned int uint8           __attribute__((ext_vector_type(8)));
typedef unsigned int uint4           __attribute__((ext_vector_type(4)));
typedef unsigned int uint3           __attribute__((ext_vector_type(3)));
typedef unsigned int uint2           __attribute__((ext_vector_type(2)));
typedef long long16                  __attribute__((ext_vector_type(16)));
typedef long long8                   __attribute__((ext_vector_type(8)));
typedef long long4                   __attribute__((ext_vector_type(4)));
typedef long long3                   __attribute__((ext_vector_type(3)));
typedef long long2                   __attribute__((ext_vector_type(2)));
typedef long long llong16            __attribute__((ext_vector_type(16)));
typedef long long llong8             __attribute__((ext_vector_type(8)));
typedef long long llong4             __attribute__((ext_vector_type(4)));
typedef long long llong3             __attribute__((ext_vector_type(3)));
typedef long long llong2             __attribute__((ext_vector_type(2)));
typedef unsigned long ulong16        __attribute__((ext_vector_type(16)));
typedef unsigned long ulong8         __attribute__((ext_vector_type(8)));
typedef unsigned long ulong4         __attribute__((ext_vector_type(4)));
typedef unsigned long ulong3         __attribute__((ext_vector_type(3)));
typedef unsigned long ulong2         __attribute__((ext_vector_type(2)));
typedef unsigned long long ullong16  __attribute__((ext_vector_type(16)));
typedef unsigned long long ullong8   __attribute__((ext_vector_type(8)));
typedef unsigned long long ullong4   __attribute__((ext_vector_type(4)));
typedef unsigned long long ullong3   __attribute__((ext_vector_type(3)));
typedef unsigned long long ullong2   __attribute__((ext_vector_type(2)));
typedef float float16                __attribute__((ext_vector_type(16)));
typedef float float8                 __attribute__((ext_vector_type(8)));
typedef float float4                 __attribute__((ext_vector_type(4)));
typedef float float3                 __attribute__((ext_vector_type(3)));
typedef float float2                 __attribute__((ext_vector_type(2)));
typedef double double16              __attribute__((ext_vector_type(16)));
typedef double double8               __attribute__((ext_vector_type(8)));
typedef double double4               __attribute__((ext_vector_type(4)));
typedef double double3               __attribute__((ext_vector_type(3)));
typedef double double2               __attribute__((ext_vector_type(2)));
// +-------------------------------------------------------------------------+


// +--- Types to capture sizes of various mesh attributes. ------------------+
//
// We use the follow terminology when it comes to mesh attributes:
//
//   - 'rank' is the dimensionality of the mesh (e.g. a two dimensional
//     mesh has rank == 2).
//
//   - 'dimension' is the size of particular mesh along a given rank.
//     For example a 128x256 uniform mesh has a dimension of 128 along
//     rank 0 (the x-axis) and 256 along rank 1 (the y-axis).
//
//   - 'stride' is the distance taken when operating on elements of a mesh.
//     This is typically in references to a particular set of mesh elements
//     such as cells, vertices, or edges.
//
//   -  An 'index' is typically used to index into a specific mesh location.
//      This is rarely used in user-level code but is here for completeness.
//
// These values are returned by various language-level intrinsics.
typedef uint16_t rank_t;
typedef uint32_t dim_t;
typedef uint64_t stride_t;
typedef uint32_t index_t;
typedef uint64_t address_t;
typedef index_t  position_t          __attribute__((ext_vector_type(3)));
// +-------------------------------------------------------------------------+

#endif
