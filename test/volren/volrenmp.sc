
/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * Simplistic volume rendering...
 * This version allows user to specify multiple nodes to mpi.
 *
 */

#include <iostream>
#include <stdio.h>
#include "mycolormap.h"
#include <mpi.h>
#include "scout/Runtime/scout.h"

#define DATA_SPHERECUBESIZE 256 
#define SQR(x) ((x) * (x))
#define MAX(x, y) ((x) > (y)? (x) : (y))
#define MIN(x, y) ((x) < (y)? (x) : (y))
#define CLAMP(x, minval, maxval) (MIN(MAX(x, (minval)), (maxval)))

using namespace std;
using namespace scout;

//SC_TODO: temporary workaround for broken scout vectors in camera
typedef float float3d __attribute__((ext_vector_type(3)));

uniform mesh AMeshType{
cells:
  float data;
};


int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <npx> <npy> <npz> \n",
        argv[0]);
    return 1;
  }

  /* set up MPI communication */

  MPI_Comm gcomm;

  MPI_Init(&argc, &argv);

  int ivdim[3] = {atoi(argv[3]), atoi(argv[2]), atoi(argv[1])};
  int ivper[3] = {0, 0, 0};

  MPI_Cart_create(MPI_COMM_WORLD, 3, ivdim, ivper, 0, &gcomm);

  __volren_gcomm = gcomm;

  /* switch x and z axes : convert column-major to row-major */
  int     npz = atoi(argv[1]);
  int     npy = atoi(argv[2]);
  int     npx = atoi(argv[3]);

  /* compute info about where we are within domain */

  int domain_grid_size[3];

  domain_grid_size[0] = DATA_SPHERECUBESIZE;
  domain_grid_size[1] = DATA_SPHERECUBESIZE;
  domain_grid_size[2] = DATA_SPHERECUBESIZE;

  int nx = domain_grid_size[0] / npx;
  int ny = domain_grid_size[1] / npy;
  int nz = domain_grid_size[2] / npz;

  int id;
  MPI_Comm_rank(gcomm, &id);

  int mypz = id /(npx * npy);
  int mypx = (id - mypz * npx * npy) % npx;
  int mypy = (id - mypz * npx * npy) / npx;

  uint64_t start[3];

  start[0] = mypx * nx;
  start[1] = mypy * ny;
  start[2] = mypz * nz;

  float center[3];

  int i;

  /* ----- sphere center ----- */
  for (i = 0; i < 3; i++) {
     center[i] = (float)(DATA_SPHERECUBESIZE-1.0)/2.0;
  }

  // set up mesh

  AMeshType amesh[nx, ny, nz];

  // generate data in mesh

  forall cells c of amesh {
    float p[3];
    p[2] = (float)c.position.x + start[0];
    p[1] = (float)c.position.y + start[1];
    p[0] = (float)c.position.z + start[2];

    c.data = CLAMP((1.0 - sqrt(SQR(p[0] - center[0])+
            SQR(p[1] - center[1])+
            SQR(p[2] - center[2])) 
            / (float)(DATA_SPHERECUBESIZE-1)), 0, 1);

  }

  printf ("Finished setting data\n");
 
  //SC_TODO: temporary workaround for broken scout vectors in camera
  float3d mypos = (float3d){-300.0f, -300.0f, -300.0f};
  float3d mylookat = (float3d){0.0f, 0.0f, 0.0f};
  float3d myup = (float3d){0.0f, 0.0f, -1.0f};

  camera cam {
    near = 70.0;
    far = 100.0;
    fov = 40.0;
    pos = mypos;
    lookat = mylookat;
    up = myup;
  };

  renderall cells c of amesh with cam {
    float val;
    val = data;
    val = (MYCOLORMAP_SIZE-1)*val;
    val = CLAMP(val, 0, (MYCOLORMAP_SIZE-1));
    int index = (int)val;
    index *= 4;

    // return value indexed by colormap
    color.r  =  mycolormap[index];
    color.g  =  mycolormap[index+1];
    color.b  =  mycolormap[index+2];
    color.a  =  mycolormap[index+3];
  }

  MPI_Finalize();

  return 0;
}
