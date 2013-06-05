//
// The routines in this file handle setting the initial conditions of
// the mesh.
//

#include "mesh.sch"

static const float MAX_TEMPERATURE = 100.0f;

// ----- initialize_mesh
//
void initialize_mesh(UniMesh2D& hmesh)
{
  forall cells c of hmesh {
    t1 = MAX_TEMPERATURE;
    t2 = MAX_TEMPERATURE;

    if (c.position.x == 0 || c.position.x == (hmesh.width-1)) {
      t1 = t2 = 0.0;
    }

    if (c.position.y == 0 || c.position.y == (hmesh.height-1)) {
      t1 = t2 = 0.0;
    }
  }
}

