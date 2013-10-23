/*
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
 * +--------------------------------------------------------------------------+
 * SC_TODO :
 *
 *   - Should we consider a case where our color clamping between 0.0 and
 *     1.0 is optional?  This at least avoids likely conditionals within the
 *     body of our rendering constructs.
 */

#ifndef __SCOUT_COLOR_SCH__
#define __SCOUT_COLOR_SCH__

#include "scout/math.sch"
#include "scout/types.sch"

// The language has a built-in (implicit) color vector within rendering
// constructs.  This color represents a 4-channel color containing red,
// green, blue channels and a fourth channel for alpha (representing
// transparency).  All channel values are floating point values and
// are expected to fall within the range [ 0.0 ... 1.0 ] (inclusive).
// Note that alpha values equal to 0.0 are completely transparent and
// values equal to 1.0 are considered opaque.
//

// +--- Support for 32-bit (or 64-bit) color channels ------------------------+
// We can configure the color settings to use either a 32-bit or 64-bit
// value for storing color channels.  In general, it is very unlikely
// that a 64-bit value is needed per channel (as most displays can't
// handle that precision).  Thus our default behavior is to use 32-bit
// values for each channel.
#ifdef SC_USE_64BIT_COLORS // 'color' is 64-bit (double) in this mode.
typedef double4 color_t;
typedef double  color_channel_t;
#else
typedef float4 color_t;    // 'color' is 32-bit (single) in this mode.
typedef float  color_channel_t;
#endif

// These values are for flexibility.  Our policy is to fix the range to
// [ 0.0 ... 1.0] (inclusive) for all color channels.  However, there
// are many details beyond simply changing these values that must be
// considered.  For this reason, it is strongly suggested that these
// values not be changed unless you are familiar with the details...
extern const color_channel_t SC_MIN_CHANNEL_VALUE;
extern const color_channel_t SC_MAX_CHANNEL_VALUE;
// +--------------------------------------------------------------------------+



// +--------------------------------------------------------------------------+
// Color-related functions for use in rendering constructs.  These
// should help simplify the computation of colors similar to what is
// done in OpenGL shading language constructs.  There are also several
// other helpful routines within the math header ("math.sch").

/// Convert a hue-saturation-value color representation into an RGB
/// color -- alpha value remains constant.  The hue value is clamped
/// between 0.0 and 360.0 degrees (inclusive) -- all other values are
/// clamped to the range [ 0.0 ... 1.0 ] (inclusive).
extern color_t hsva(color_channel_t hue,
                    color_channel_t saturation,
                    color_channel_t value,
                    color_channel_t alpha);

/// Convert a hue-saturation-value color representation into an RGB
/// color -- return alpha value is fixed at 1.0 (opaque).  The hue
/// value is clamped between 0.0 and 360.0 degrees (inclusive) -- all
/// other values are clamped to the range [ 0.0 ... 1.0] (inclusive).
extern color_t hsv(color_channel_t hue,
                   color_channel_t saturation,
                   color_channel_t value);

/// Linear blend of two color channels values with an associated
/// alpha weight.  The blend is straightforward and is implemented
/// as: c0 + (c1 - c0) * alpha.  As with other color functions,
/// all values are assumed to lie in the range [ 0.0 ... 1.0]
/// (inclusive).
extern color_channel_t mix(color_channel_t ch0,
                           color_channel_t ch1,
                           color_channel_t alpha);

/// Linear blend of the two colors with a single alpha weighting.
/// The blend is straightforward and is implemented as a vector
/// operation: c0 + (c1 - c0) * alpha.  As with other color
/// operations, all values are assumed to lie in the range
/// [ 0.0 ... 1.0] (inclusive).
extern color_t mix(const color_t         &c0,
                   const color_t         &c1,
                   const color_channel_t  alpha);

/// Linear blend of the two colors with an associated set of
/// alpha values stored in the 'alpha' vector parameter.  The
/// blend is straightforward and is implemented as a vector
/// operation: c0 + (c1 - c0) * alpha.  As with other color
/// operations, all values are assumed to lie in the range
/// [ 0.0 ... 1.0] (inclusive).
extern color_t mix(const color_t &c0,
                   const color_t &c1,
                   const color_t &alpha);
// +--------------------------------------------------------------------------+

#endif
