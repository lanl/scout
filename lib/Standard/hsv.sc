/*
 *	
 * ###########################################################################
 * Copyrigh (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
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
 * ###########################################################################
 *
 * Notes
 *
 *  The routines in this file allow for the manipulation of colors in
 *  the HSV color space.  In particular there are routines for mapping
 *  the HSV color space into the RGB(A) space required by the
 *  underlying graphics/rendering systems in Scout.
 * 
 */ 


// ----- hsva
//
// Take an HSV set of parameters, including an alpha value, and return a
// corresponding RGBA value.
//
float4 hsva(float hue, float sat, float value, float alpha) {
  
  float4 color;

  if (sat == 0) {
    // the color is on the black-white center line.
    color.r =
      color.g =
      color.b  = value;
    color.a = alpha;
  } else {
      
    color.a = alpha;      

    if (hue == 360.0) {
      hue = 0.0;
    }

    hue = hue / 60.0f;
      
    int   i = int(floor(hue));
    float f = hue - i;
    float p = value * (1.0f - sat);
    float q = value * (1.0f - sat * f);
    float t = value * (1.0f - sat * (1.0f - f));

    switch(i) {

      case 0:
        color.r = value;
        color.g = t;
        color.b = p;
        break;

      case 1:
        color.r = q;
        color.g = value;
        color.b = p;
        break;

      case 2:
        color.r = p;
        color.g = value;
        color.b = t;
        break;          

      case 3:
        color.r = p;
        color.g = q;
        color.b = value;
        break;          

      case 4:
        color.r = t;
        color.g = p;
        color.b = value;
        break;          

      default:
        color.r = value;
        color.g = p;
        color.b = q;
        break;                    
    }
  }
  
  return color;  
}


// ----- hsv
//
float4 hsv(float hue, float sat, float value) {
  return hsva(hue, sat, value, 1.0f);
}
