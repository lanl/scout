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
 *
 */

int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 100;
  const float MAX_TEMP      = 100.0f;

  uniform mesh HeatMeshType{
  cells:
    float t1, t2;
  };

  HeatMeshType heat_mesh[512, 512];
  
  forall cells c of heat_mesh {
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
  }

  const float dx    = 1.0f / heat_mesh.width;
  const float dy    = 1.0f / heat_mesh.height;
  const float alpha = 0.0000039f;
  const float dt    = 0.4 * (alpha / 4.0f) * ((1.0f / (dx * dx)) + (1.0f / (dy * dy)));

  // Time steps loop.
  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {

    forall cells c of heat_mesh {

      if (c.position.x > 0 && c.position.x < heat_mesh.width-1 &&
          c.position.y > 0 && c.position.y < heat_mesh.height-1) {

        float d2dx2 = cshift(c.t1, 1, 0) - 2.0f * c.t1 + cshift(c.t1, -1, 0);
        d2dx2 /= dx * dx;

        float d2dy2 = cshift(c.t1, 0, 1) - 2.0f * c.t1 + cshift(c.t1, 0, -1);
        d2dy2 /= dy * dy;

        t2 = (alpha * dt * (d2dx2 + d2dy2)) + c.t1;
      }
    }

    forall cells c of heat_mesh {
      t1 = t2;
    }

    renderall cells c of heat_mesh {
      float norm_t1 = t1 / MAX_TEMP;
      float hue = 240.0f - 240.0f * norm_t1;
      color = hsv(hue, 1.0f, 1.0f);
    }
  }

  return 0;
}
