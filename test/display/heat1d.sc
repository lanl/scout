/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * Simplistic 1D heat transfer...
 * 
 */
int main(int argc, char *argv[])
{
  const int NTIME_STEPS     = 300;
  const float MAX_TEMP      = 100.0f;
  
  uniform mesh HeatMeshType{
    cells:
      float t1, t2;
  };

  HeatMeshType heat_mesh[1024];

  // Set initial conditions.
  //
  // A nice shorthand for the forall construct above might be
  // something like this (stealing from Fortran):
  // 
  //  heat_mesh::cells.t1[0:1023:1023] = 0.0f;
  //  heat_mesh::cells.t1[1:1022] = 100.0f;
  forall cells c of heat_mesh {
    if (c.position.x > 0 && c.position.x < 1023)
      t1 = 0.0f;
    else
      t1 = MAX_TEMP;      
  }

  const float therm_conduct = 0.445f;
  const float spec_heat     = 0.113f;
  const float rho           = 7.8f;

  float mat_const = therm_conduct / (spec_heat * rho);

  // Time steps loop. 
  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {

    // The 'position' attribute of a cell is automatically
    // provided to contain the coordinates of the current 
    // cell being processed. 
    //
    // cshift (circular shift) is a built-in function (part of the 
    // standard library) that we use for access to neighboring 
    // cells in the mesh.  For now we're only working with uniform 
    // meshes.  This is like F90 but we shift index values vs. 
    // array duplication (which is a horrid feature in most F90 
    // runtimes -- no wonder the data-parallel features are never
    // used).
    forall cells c of heat_mesh {
      if (c.position.x > 0 && c.position.x < 1023) {
        t2 = t1 + mat_const * (cshift(c.t1,1) + cshift(c.t1,-1) - 2.0f * t1);
      }
    }

    forall cells c of heat_mesh {
      t1 = t2;
    }

    renderall cells c of heat_mesh {
      // Normalize temperatures ranges into the 0...1.0 range and then map into
      // HSV color space with hue running from blue to red for cold to hot. 
      float norm_t1 = t1 / MAX_TEMP;
      float hue = 240.0f - 240.0f * norm_t1;
      color = hsv(hue, 1.0f, 1.0f);
    }
  }

  return 0;
}
