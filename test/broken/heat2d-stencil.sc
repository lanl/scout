/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * Stencil-based 2D heat transfer example.  
 *
 */


uniform mesh HeatMeshType{
  cells:
  
    float temperature;

    // define a two-dimensional (cell-based) stencil operation:
    //   - x,y are pre-defined for us to simplify neighbor operations (cleaner syntax than shift calls).
    //   - time is as well (thus providing us with a time view of the temperature calculation).
    //
    // The empty range specifiers ([ : ]) denote the two-dimensional nature of the stencil operation...
    //
    stencil[ : ] heatxfer(field f, float dx, float dy, float alpha)  {

      const field f;  // this let's us know that we will only read from 'f' within the stencil.
      
      float d2dx2 = f[x+1,y] - 2.0f * f[x,y] + f[x-1,y];
      d2dx2 /= dx * dx;
  
      float d2dy2 = f[x,y+1] - 2.0f * f[x,y] + f[x,y-1];
      d2dy2 /= dy * dy;

      return (alpha * dt * (d2dx2 + d2dy2)) + f[x,y];
    };
};


int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 5000;
  const float MAX_TEMP      = 100.0f;

  HeatMeshType heat_mesh[512, 512];

  forall cells c of heat_mesh {
    t1 = MAX_TEMP;
    t2 = MAX_TEMP;
    where(c.position.x == 0 || c.position.x == heat_mesh.width-1 ||
          c.position.y == 0 || c.position.y == heat_mesh.height-1 {
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
        // stencil invocation looks function-like...  Need to work on the LHS some more...
        temperature[;t+1] = heatxfer(temperature, dx, dy, alpha);
    }

    renderall cells c of heat_mesh {
      float norm_t1 = temperature / MAX_TEMP;
      float hue = 240.0f - 240.0f * norm_t1;
      color = hsv(hue, 1.0f, 1.0f);
    }
  }

  return 0;
}
