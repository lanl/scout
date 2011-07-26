/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * This file contains the various hue-saturation value functions that
 * make up the scout runtime library calls. 
 */

#include <math.h>
#include <xmmintrin.h> 
#include "scout/types.h"

// ----- hsva
//
// Take an HSV set of parameters, including an alpha value, and return a
// corresponding RGBA value.
//
scout::float4 hsva(float hue, float sat, float value, float alpha)
{
  using namespace scout;
  float4 color;

  if (sat == 0) {
    // the color is on the black-white center line.
    color.components[0] =
      color.components[1] =
      color.components[2] = value;
    color.components[3] = alpha;
  } else {
      
    color.components[3] = alpha;      

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
        color.components[0] = value;
        color.components[1] = t;
        color.components[2] = p;
        break;

      case 1:
        color.components[0] = q;
        color.components[1] = value;
        color.components[2] = p;
        break;

      case 2:
        color.components[0] = p;
        color.components[1] = value;
        color.components[2] = t;
        break;          

      case 3:
        color.components[0] = p;
        color.components[1] = q;
        color.components[2] = value;
        break;          

      case 4:
        color.components[0] = t;
        color.components[1] = p;
        color.components[2] = value;
        break;          

      default:
        color.components[0] = value;
        color.components[1] = p;
        color.components[2] = q;
        break;                    
    }
  }
  
  return color;  
}









