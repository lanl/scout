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

#ifndef __SCOUT_MATH_SCH__
#define __SCOUT_MATH_SCH__

#include "scout/types.sch"

#if defined(__scout_cxx__)
 #include <cmath>
 #include <algorithm>
#else
 #include <math.h>
#endif

// Apologies in advance...  This is a messy set of macros for building
// type-centric versions of the clamp functions.  You would think there
// would be a decent way to do this in C++ via templates and
// specialization -- however, there are some issues in this approach
// with overloading (see http://www.gotw.ca/publications/mill17.htm).

// +--- Scalar value clamping ------------------------------------------------+
#define SC_SCALAR_CLAMP_FUNC(ValueType)                                   \
  template <typename Type>                                                \
  ValueType clamp(const ValueType &Value, Type MinValue, Type MaxValue) { \
    return std::min(std::max(Value, ValueType(MinValue)),                 \
                    ValueType(MaxValue));                                 \
  }

SC_SCALAR_CLAMP_FUNC(short);
SC_SCALAR_CLAMP_FUNC(int);
SC_SCALAR_CLAMP_FUNC(long);
//SC_SCALAR_CLAMP_FUNC(unsigned short);
//SC_SCALAR_CLAMP_FUNC(unsigned int);
//SC_SCALAR_CLAMP_FUNC(unsigned long);

// +--- floating point clamp
#define SC_SCALAR_FLT_CLAMP_FUNC(ValueType)                               \
  template <typename Type>                                                \
  ValueType clamp(const ValueType &Value, Type MinValue, Type MaxValue) { \
    return fmin(fmax(Value, ValueType(MinValue)),                         \
                ValueType(MaxValue));                                     \
  }

SC_SCALAR_FLT_CLAMP_FUNC(float);
SC_SCALAR_FLT_CLAMP_FUNC(double);
// +--------------------------------------------------------------------------+


// +--- Vector value clamping ------------------------------------------------+
#define SC_VECTOR2_CLAMP_FUNC(VecType, ValueType)                       \
  template <typename Type>                                              \
  VecType clamp(const VecType &Vec, Type MinValue, Type MaxValue) {     \
    VecType Result;                                                     \
    Result.x = clamp(Vec.x, ValueType(MinValue), ValueType(MaxValue));  \
    Result.y = clamp(Vec.y, ValueType(MinValue), ValueType(MaxValue));  \
    return Result;                                                      \
  }

SC_VECTOR2_CLAMP_FUNC(short2,  short);
SC_VECTOR2_CLAMP_FUNC(int2,    int);
SC_VECTOR2_CLAMP_FUNC(long2,   long);
SC_VECTOR2_CLAMP_FUNC(float2,  float);
SC_VECTOR2_CLAMP_FUNC(double2, double);

#define SC_VECTOR3_CLAMP_FUNC(VecType, ValueType)                       \
  template <typename Type>                                              \
  VecType clamp(const VecType &Vec, Type MinValue, Type MaxValue) {     \
    VecType Result;                                                     \
    Result.x = clamp(Vec.x, ValueType(MinValue), ValueType(MaxValue));  \
    Result.y = clamp(Vec.y, ValueType(MinValue), ValueType(MaxValue));  \
    Result.z = clamp(Vec.z, ValueType(MinValue), ValueType(MaxValue));  \
    return Result;                                                      \
  }

SC_VECTOR3_CLAMP_FUNC(short3,  short);
SC_VECTOR3_CLAMP_FUNC(int3,    int);
SC_VECTOR3_CLAMP_FUNC(long3,   long);
SC_VECTOR3_CLAMP_FUNC(float3,  float);
SC_VECTOR3_CLAMP_FUNC(double3, double);

#define SC_VECTOR4_CLAMP_FUNC(VecType, ValueType)                       \
  template <typename Type>                                              \
  VecType clamp(const VecType &Vec, Type MinValue, Type MaxValue) {     \
    VecType Result;                                                     \
    Result.x = clamp(Vec.x, ValueType(MinValue), ValueType(MaxValue));  \
    Result.y = clamp(Vec.y, ValueType(MinValue), ValueType(MaxValue));  \
    Result.z = clamp(Vec.z, ValueType(MinValue), ValueType(MaxValue));  \
    Result.w = clamp(Vec.w, ValueType(MinValue), ValueType(MaxValue));  \
    return Result;                                                      \
  }

SC_VECTOR4_CLAMP_FUNC(short4,  short);
SC_VECTOR4_CLAMP_FUNC(int4,    int);
SC_VECTOR4_CLAMP_FUNC(long4,   long);
SC_VECTOR4_CLAMP_FUNC(float4,  float);
SC_VECTOR4_CLAMP_FUNC(double4, double);
/*
 * This has C++ 11 features -- disabled for now...
 *
 *

#define SC_VECTORN_CLAMP_FUNC(VecType, ValueType, NComponents)             \
  template <typename Type, unsigned N=NComponents>                         \
  VecType clamp(const VecType &Vec, Type MinValue, Type MaxValue) {        \
    VecType Result;                                                        \
    for(unsigned i = 0; i < N; ++i) {                                      \
      Result[i] = clamp(Vec[i], ValueType(MinValue), ValueType(MaxValue)); \
    }                                                                      \
    return Result;                                                         \
  }

 SC_VECTORN_CLAMP_FUNC(short8,    short,  8);
 SC_VECTORN_CLAMP_FUNC(short16,   short, 16);
 SC_VECTORN_CLAMP_FUNC(int8,        int,  8);
 SC_VECTORN_CLAMP_FUNC(int16,       int, 16);
 SC_VECTORN_CLAMP_FUNC(long8,      long,  8);
 SC_VECTORN_CLAMP_FUNC(long16,     long, 16);
 SC_VECTORN_CLAMP_FUNC(float8,    float,  8);
 SC_VECTORN_CLAMP_FUNC(float16,   float, 16);
 SC_VECTORN_CLAMP_FUNC(double8,  double,  8);
 SC_VECTORN_CLAMP_FUNC(double16, double, 16);
 */
// +--------------------------------------------------------------------------+


#endif

