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
 *
 */
#include <stdio.h>

const color_channel_t SC_MIN_CHANNEL_VALUE =   0.0;
const color_channel_t SC_MAX_CHANNEL_VALUE =   1.0;

const color_channel_t SC_MIN_HUE_VALUE     =   0.0;
const color_channel_t SC_MAX_HUE_VALUE     = 360.0;

color_channel_t clamp(color_channel_t d, color_channel_t min, color_channel_t max) {
  color_channel_t t = d < min ? min : d;
  return t > max ? max : t;
}

  // +--- Convert HSV color to RGB color ---------------------------------------+
  //
  color_t hsv(color_channel_t hue,
	      color_channel_t saturation,
	      color_channel_t value) {

    color_t rgbaColor;
    rgbaColor.a = 0.0;

    hue         = clamp(hue, SC_MIN_HUE_VALUE, SC_MAX_HUE_VALUE);
    saturation  = clamp(saturation, SC_MIN_CHANNEL_VALUE, SC_MAX_CHANNEL_VALUE);
    value       = clamp(value, SC_MIN_CHANNEL_VALUE, SC_MAX_CHANNEL_VALUE);

    if (saturation == SC_MIN_CHANNEL_VALUE) {
      rgbaColor.rgb = value;
      return rgbaColor;
    }

    int   i;
    float f, p, q, t;

    hue = hue / 60.0;
    i   = (int)(hue);
    f   = hue - (float)(i);
    p   = value * (1.0 - saturation);
    q   = value * (1.0 - saturation * f);
    t   = value * (1.0 - saturation * (1.0 - f));

    switch(i) {

    case 0:
      rgbaColor.r = value;
      rgbaColor.g = t;
      rgbaColor.b = p;
      break;

    case 1:
      rgbaColor.r = q;
      rgbaColor.g = value;
      rgbaColor.b = p;
      break;

    case 2:
      rgbaColor.r = p;
      rgbaColor.g = value;
      rgbaColor.b = t;
      break;

    case 3:
      rgbaColor.r = p;
      rgbaColor.g = q;
      rgbaColor.b = value;
      break;

    case 4:
      rgbaColor.r = t;
      rgbaColor.g = p;
      rgbaColor.b = value;
      break;

    default:
      rgbaColor.r = value;
      rgbaColor.g = p;
      rgbaColor.b = q;
      break;
    }

    return rgbaColor;
  }

  // +--- Convert HSV-alpha color to RGB-alpha color ---------------------------+
  //
  color_t hsva(color_channel_t hue,
	       color_channel_t saturation,
	       color_channel_t value,
	       color_channel_t alpha) {
    //printf("hsva %f %f %f %f\n", hue, saturation, value, alpha); 
    color_t rgbaColor = hsv(hue, saturation, value);
    rgbaColor.a = clamp(alpha, SC_MIN_CHANNEL_VALUE, SC_MAX_CHANNEL_VALUE);
    return rgbaColor;
  }


// +--- Linear blend of 2 channel values and an alpha weighting. -------------+
//
color_channel_t mix(color_channel_t ch0,
                    color_channel_t ch1,
                    color_channel_t alpha) {
  color_channel_t a = clamp(alpha, SC_MIN_CHANNEL_VALUE, SC_MAX_CHANNEL_VALUE);
  return ch0 + (ch1 - ch0) * alpha;
}

