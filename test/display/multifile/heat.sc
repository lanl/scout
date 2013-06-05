//
// The actual heat transfer computations... 
//
#include "mesh.sch"
#include "heat.h"

static const int N_TIME_STEPS = 500;

// ----- heat_xfer
//
void heat_transfer(UniMesh2D& heat_mesh)
{
  const float dx    = 1.0f / heat_mesh.width;
  const float dy    = 1.0f / heat_mesh.height;
  const float alpha = 0.00003;
  const float dt    = 0.4 * (alpha / 4.0f) * ((1.0f / (dx * dx)) + (1.0 / (dy * dy)));

  for(unsigned int t = 0; t < N_TIME_STEPS; ++t) {

    forall cells c of heat_mesh {
        
      if (c.position.x > 0 && c.position.x < (heat_mesh.width-1) &&
          c.position.y > 0 && c.position.y < (heat_mesh.height-1)) {

        float d2dx2 = cshift(c.t1, 1, 0) - 2.0 * c.t1 + cshift(c.t1, -1, 0);
        d2dx2 /= dx * dx;
        
        float d2dy2 = cshift(c.t1, 0, 1) - 2.0 * c.t1 + cshift(c.t1, 0, -1);
        d2dy2 /= dy * dy;

        t2 = (alpha * dt * (d2dx2 + d2dy2)) + t1;
      }
    }

    forall cells c of heat_mesh {
      t1 = t2;
    }

    renderall cells c of heat_mesh {
      float norm_t1 = t1 / MAX_TEMPERATURE;
      float hue     = 240.0 - 240.0 * norm_t1;
      color         = hsv(hue, 1.0, 1.0);
    }
  }
}
