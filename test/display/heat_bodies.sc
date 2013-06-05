/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * Simplistic 2D heat transfer...
 * Modified to include (quasi-)soluble bodies and constant advection term
 * Jamal Mohd-Yusof 10/20/11
 *
 */

#include <stdio.h>

#define N_BODIES   5
#define SOLUBLE    1
#define SHOW_SOLID (1-SOLUBLE)
#define MESH_DIM   512

int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 100;
  const float MAX_TEMP      = 100.0f;

  uniform mesh HeatMeshType{
   cells:
    float h;
    float h_next;
    float mask;
  };

  HeatMeshType heat_mesh[MESH_DIM, MESH_DIM];

  int c_x[N_BODIES] = {128, 128, 394, 394, 256};
  int c_y[N_BODIES] = {128, 394, 128, 394, 256};
  int r2cyl = MESH_DIM / 4;
  float u = 0.001;

  forall cells c of heat_mesh {
    h = 0.0f;
    h_next = 0.0f;
    mask = 1.0;

    /* no left/right boundaries 
    if (c.position.x == 0 || c.position.x == (heat_mesh.width-1)) {
      h = MAX_TEMP;
      h_next = MAX_TEMP;
      mask = 0.0;
    }
    */

    if (c.position.y == 0 || c.position.y == (heat_mesh.height-1)) {
      h = MAX_TEMP;
      h_next = MAX_TEMP;
      mask = 0.0;
    }

    for (int i = 0; i < N_BODIES; i++) {
      float r2 = (c.position.x - c_x[i])*(c.position.x - c_x[i]) +
        (c.position.y - c_y[i])*(c.position.y - c_y[i]);
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
  
  const float dx    = 10.0f / heat_mesh.width;
  const float dy    = 10.0f / heat_mesh.height;
  const float alpha = 0.00001f;
  const float dt    = 0.5f * (dx * dx+ dy * dy)/4.0f/alpha;
  
  //printf ("dx = dy = %e, alpha = %e, dt = %e\n", dx, alpha, dt);

  // Time steps loop.
  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {
    
    forall cells c of heat_mesh {
      float ddx = 0.5*(cshift(c.h, 1, 0) - cshift(c.h, -1, 0))/dx;
      float d2dx2 = cshift(c.h, 1, 0) - 2.0f * c.h + cshift(c.h, -1,  0);
      d2dx2 /= dx * dx;

      float d2dy2 = cshift(c.h, 0, 1) - 2.0f * c.h + cshift(c.h,  0, -1);
      d2dy2 /= dy * dy;

      h_next = mask*dt*(alpha * (d2dx2 + d2dy2) - mask*u*ddx) + c.h;
    }


    forall cells c of heat_mesh {
      h = h_next;
    }

    renderall cells c of heat_mesh {
      float norm_h = h / MAX_TEMP;
      float hue = 240.0f - 240.0f * norm_h;
#if (SHOW_SOLID)
      color = hsv(hue, 1.0f, mask);
#else
      color = hsv(hue, 1.0f, 1.0f);
#endif
      }
  }
  
  return 0;
}
