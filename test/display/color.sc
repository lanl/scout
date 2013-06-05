/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * A simple rendering test with no forall stmt's.
 *
 */

int main(int argc, char *argv[]) {
  const int NTIME_STEPS = 1000;

  uniform mesh HeatMeshType{
  cells:
    float h1;
  };

  HeatMeshType heat_mesh[2, 2];

  for(unsigned int t = 0; t < NTIME_STEPS; ++t) {

    forall cells c of heat_mesh {
      h1 = float(t)/NTIME_STEPS;
    }

    renderall cells c of heat_mesh {
      color.r = ((float)position.x / heat_mesh.width)*h1;
      color.g = ((float)position.y / heat_mesh.height)*h1;
      color.b = 0.0f;
      color.a = 1.0f;
    }
  }

  return 0;
}
