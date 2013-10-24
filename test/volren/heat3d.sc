/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
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
 * ########################################################################### 
 * 
 * Notes
 *
 * Simplistic 2D heat transfer...
 *
 * ##### 
 */ 

#define XDIM 32 
#define YDIM 32 
#define ZDIM 32 
#define PRINTFIELD 0


//SC_TODO: temporary workaround for broken scout vectors in camera
typedef float float3d __attribute__((ext_vector_type(3)));

void myhsva(float h, float s, float v, float a, float r[4])
{

  r[3] = a;

  int i;
  float f, p, q, t;
  if (s == 0.0f) {
    r[0] = v;
    r[1] = v;
    r[2] = v;
    return;
  }

  h /= 60.0f;
  i = int(h);
  f = h - float(i);
  p = v * (1.0 - s);
  q = v * (1.0 - s * f);
  t = v * (1.0 - s * (1.0 - f));

  switch(i) {

    case 0:
      r[0] = v;
      r[1] = t;
      r[2] = p;
      break;

    case 1:
      r[0] = q;
      r[1] = v;
      r[2] = p;
      break;

    case 2:
      r[0] = p;
      r[1] = v;
      r[2] = t;
      break;

    case 3:
      r[0] = p;
      r[1] = q;
      r[2] = v;
      break;

    case 4:
      r[0] = t;
      r[1] = p;
      r[2] = v;
      break;

    default:
      r[0] = v;
      r[1] = p;
      r[2] = q;
      break;
  }

}

int main(int argc, char *argv[])
{
  const int NTIME_STEPS       = 25;
  const float MAX_TEMP        = 100.0f;

  uniform mesh HeatMeshType{
  cells:
    float t1, t2;
  };

  HeatMeshType heat_mesh[XDIM, YDIM, ZDIM];

  forall cells c in heat_mesh {
    t1 = 0.0;
    t2 = 0.0;
  }

  forall cells c in heat_mesh {
    t1 = MAX_TEMP;
    t2 = MAX_TEMP;

    if (c.position.x == 0 || c.position.x == (heat_mesh.width-1)) {
      t1 = 0.0;
      t2 = 0.0;
    }

    if (c.position.y == 0 || c.position.y == (heat_mesh.height-1)) {
      t1 = 0.0;
      t2 = 0.0;
    }

    if (c.position.z == 0 || c.position.z == (heat_mesh.depth-1)) {
      t1 = 0.0;
      t2 = 0.0;
    }    
  }

  const float dx    = 1.0f / heat_mesh.width;
  const float dy    = 1.0f / heat_mesh.height;
  const float dz    = 1.0f / heat_mesh.depth;  
  const float alpha = 0.00039f; // good for 32x32x32
  //const float alpha = 0.00039f; // good for 64x64x64
  //const float alpha = 0.0000039f;
  const float dt    = 0.4 * (alpha / 4.0f) * ((1.0f / (dx * dx)) + (1.0f / (dy * dy)) + (1.0f / (dz * dz)));
  printf("dt: %f\n", dt);

#if PRINTFIELD
  float outfield[XDIM*YDIM*ZDIM];
  for (int i = 0; i < XDIM*YDIM*ZDIM; i++) {
      outfield[i] = -1.0;
  }
  forall cells c in heat_mesh {
        int index = (c.position.z*YDIM + c.position.y)*XDIM + c.position.x;
        outfield[index] = t1;
  }
#endif

  // Time steps loop.
  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {

    //printf("time: %u\n", t);
    forall cells c in heat_mesh {


      if (c.position.x > 0 && c.position.x < heat_mesh.width-1 &&
          c.position.y > 0 && c.position.y < heat_mesh.height-1 &&
          c.position.z > 0 && c.position.z < heat_mesh.depth-1) {

        float val;

        float d2dx2 = cshift(c.t1, 1, 0, 0) - 2.0f * c.t1 + cshift(c.t1, -1, 0, 0);
        d2dx2 /= dx * dx;

        float d2dy2 = cshift(c.t1, 0, 1, 0) - 2.0f * c.t1 + cshift(c.t1, 0, -1, 0);
        d2dy2 /= dy * dy;

        float d2dz2 = cshift(c.t1, 0, 0, 1) - 2.0f * c.t1 + cshift(c.t1, 0, 0, -1);
        d2dz2 /= dz * dz;        

        val = (alpha * dt * (d2dx2 + d2dy2 + d2dz2)) + c.t1;
        t2 = val;
#if PRINTFIELD
        int index = (c.position.z*YDIM + c.position.y)*XDIM + c.position.x;
        outfield[index] = val;
#endif
      }
    }

#if PRINTFIELD
    for (int i = 0; i < XDIM*YDIM*ZDIM; i++) {
      printf("outfield[%d] = %f\n", i, outfield[i]);
    }
#endif

    forall cells c in heat_mesh {
      t1 = t2;
    }

    //SC_TODO: temporary workaround for broken scout vectors in camera
    float3d mypos = (float3d){32.0f, 32.0f, 100.0f};
    float3d mylookat = (float3d){32.0f, 32.0f, 0.0f};
    float3d myup = (float3d){0.0f, 1.0f, 0.0f};

    camera cam {
      near = 64.0;
      far = 200.0;
      fov = 95.0;
      pos = mypos;
      lookat = mylookat;
      up = myup;
    };

    renderall cells c in heat_mesh with cam 
    {
      float norm_t1;
      if (t1 > MAX_TEMP) {
       norm_t1 = 1.0f;
      } else {
       norm_t1 = t1 / MAX_TEMP;
      } 
      float hue = 240.0f - 240.0f * norm_t1;
      float thiscolor[4];
      myhsva(hue, 1.0f, 1.0f, norm_t1, thiscolor);
      color.r = thiscolor[0];
      color.g = thiscolor[1];
      color.b = thiscolor[2];
      color.a = thiscolor[3];
    }
  }

  return 0;
}
