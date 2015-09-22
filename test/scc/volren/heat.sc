/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2015. Los Alamos National Security, LLC. This software was
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
 * ########################################################################### 
 *
 */

#include <stdio.h>

#define N_BODIES   1
#define SOLUBLE    0
#define SHOW_SOLID (1-SOLUBLE)
#define MESH_DIM   90

float4 hsva2(float hue,
             float saturation,
             float value,
             float alpha){

  if(hue > 360.0f){
    hue = 360.0f;
  }
  else if(hue < 0.0f){
    hue = 0.0f;
  } 

  if(saturation > 1.0f){
    saturation = 1.0f;
  }
  else if(saturation < 0.0f){
    saturation = 0.0f;
  } 

  if(value > 1.0f){
    value = 1.0f;
  }
  else if(value < 0.0f){
    value = 0.0f;
  }

  if(alpha > 1.0f){
    alpha = 1.0f;
  }
  else if(alpha < 0.0f){
    alpha = 0.0f;
  }  

  float4 rgbaColor;
  rgbaColor.a = alpha;

  if (saturation == 0.0f) {
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

int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 200;
  const float MAX_TEMP      = 100.0f;

  window render_win[1024,1024];

  uniform mesh HeatMeshType{
  cells:
    float h;
    float h_next;
    float mask;
  };

  HeatMeshType heat_mesh[MESH_DIM, MESH_DIM, MESH_DIM];

  int c_x[N_BODIES] = {67};
  int c_y[N_BODIES] = {45};
  int c_z[N_BODIES] = {90};

  int r2cyl = MESH_DIM / 16;
  float u = 0.001;

  forall cells c in heat_mesh {
    h = 0.0f;
    h_next = 0.0f;
    mask = 1.0;

    if (position().y == 0 || position().y == (height()-1)) {
      h = MAX_TEMP;
      h_next = MAX_TEMP;
      mask = 0.0;
    }

    for (int i = 0; i < N_BODIES; i++) {
      float r2 = (position().x - c_x[i])*(position().x - c_x[i]) +
        (position().y - c_y[i])*(position().y - c_y[i]) +
        (position().z - c_z[i])*(position().z - c_z[i]);

      if (r2 < r2cyl) {
        if (SOLUBLE) {
          mask = r2/r2cyl;
        } else {
          mask = 0.0;
        }
        h = MAX_TEMP;
        h_next = MAX_TEMP;
      }
    }
  }

  const float dx    = 10.0f / MESH_DIM;
  const float dy    = 10.0f / MESH_DIM;
  const float dz    = 10.0f / MESH_DIM;
  const float alpha = 0.00001f;
  const float dt    = 0.5f * (dx * dx + dy * dy + dz * dz)/4.0f/alpha;

  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {
    
    forall cells c in heat_mesh {
      float ddx = 0.5*(cshift(c.h, 1, 0) - cshift(c.h, -1, 0))/dx;
      float ddy = 0.5*(cshift(c.h, 0, 1) - cshift(c.h, 0, -1))/dy;

      float d2dx2 = cshift(c.h, 1, 0) - 2.0f * c.h + cshift(c.h, -1, 0);
      d2dx2 /= dx * dx;

      float d2dy2 = cshift(c.h, 0, 1) - 2.0f * c.h + cshift(c.h, 0, -1);
      d2dy2 /= dy * dy;

      float d2dz2 = cshift(c.h, 0, 0, 1) - 2.0f * c.h + cshift(c.h, 0, 0, -1);
      d2dz2 /= dz * dz;

      h_next = mask*dt*(alpha * (d2dx2 + d2dy2 + d2dz2) - mask*u*ddx*ddy) + c.h;
    }

    swapFields(heat_mesh.h, heat_mesh.h_next);

    renderall cells c in heat_mesh to render_win {
      color = hsva2(240.0f - 240.0f * (h / MAX_TEMP), 1.0f, 1.0f, 1.0f);
    }
  }

  return 0;
}
