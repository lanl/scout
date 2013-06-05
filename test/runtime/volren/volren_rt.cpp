/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * -----
 * 
 */


#include <iostream>
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/renderall/mpi/RenderallVolume.h"
#include "mycolormap.h"
#include <mpi.h>

#define DATA_SPHERECUBESIZE 64 

using namespace std;
using namespace scout;


static const size_t WINDOW_WIDTH = 768;
static const size_t WINDOW_HEIGHT = 768;

// generates test grid 
void genTestGrid(int id, int npx, int npy, int npz, 
  int* pnx, int* pny, int* pnz, double** ptrx, double** ptry, double** ptrz)
{
    int domain_grid_size[3];

    domain_grid_size[0] = DATA_SPHERECUBESIZE;
    domain_grid_size[1] = DATA_SPHERECUBESIZE;
    domain_grid_size[2] = DATA_SPHERECUBESIZE;

    int nx = domain_grid_size[0] / npx;
    int ny = domain_grid_size[1] / npy;
    int nz = domain_grid_size[2] / npz;

    int mypz = id /(npx * npy);
    int mypx = (id - mypz * npx * npy) % npx;
    int mypy = (id - mypz * npx * npy) / npx;

    uint64_t start[3];

    start[2] = mypx * nx;
    start[1] = mypy * ny;
    start[0] = mypz * nz;

    // set grid, 256 evenly spaced 1-unit ticks on all axes

    double *x, *y, *z;
    x = (double *)calloc(nx, sizeof(double));
    y = (double *)calloc(ny, sizeof(double));
    z = (double *)calloc(nz, sizeof(double));

    int i;

    for (i = 0; i < nx; i++) {
        x[i] = start[2] + i;
    }

    for (i = 0; i < ny; i++) {
        y[i] = start[1] + i;
    }

    for (i = 0; i < nz; i++) {
        z[i] = start[0] + i;
    }

  *pnx = nx;
  *pny = ny;
  *pnz = nz;
  *ptrx = x;
  *ptry = y;
  *ptrz = z;
}

// generates test data for one volume
void genTestData(void **ptest_data,
  int nx, int ny, int nz, double *x, double *y, double *z)
{

    /* volume */
    float *test_data = (float *)malloc(nx * ny * nz * sizeof(float));
    if (test_data == NULL) {
      printf("Out of memory!\n");
    }

    int px, py, pz;
    float c[3], p[3];
    float value;

    int i;

    /* ----- sphere ----- */
    for (i = 0; i < 3; i++) {
       c[i] = (float)(DATA_SPHERECUBESIZE-1.0)/2.0;
    }


   for (pz = 0; pz < nz; pz++) {
        for (py = 0; py < ny; py++) {
            for (px = 0; px < nx; px++) {

                p[2] = x[px];
                p[1] = y[py];
                p[0] = z[pz];

                value = SQR(p[0] - c[0])+
                    SQR(p[1] - c[1])+
                    SQR(p[2] - c[2]);

                value = 1.0 - sqrt(value) / (float)(DATA_SPHERECUBESIZE-1);

                value = CLAMP(value, 0, 1);

                int id = px + py * nx + pz * nx * ny;

                test_data[id] = value;
            }
        }
    }

  *ptest_data = (void*)test_data;
}

void printTestData(void *ptest_data,
  int nx, int ny, int nz, double* x, double* y, double* z)
{

    int px, py, pz;
    float c[3], p[3];
    double value;

    int i;

   for (pz = 0; pz < nz; pz++) {
        for (py = 0; py < ny; py++) {
            for (px = 0; px < nx; px++) {
                int id = px + py * nx + pz * nx * ny;
              printf("%lf ", ((double*)ptest_data)[id]);
            }
            printf("\n");
        }
    }
}

/**
 * my_transfer_func
 *
 */
trans_func_ab_t my_transfer_func = ^ int (block_t* block, point_3d_t* pos, rgba_t& partialcolor)
{

  float val;

  // I think we want transfer function to do its own value getting, because
  // we may want to get more than one data field.  This call just gets a value
  // for one field (the 0th field).
  if (block_get_value(block, 0, pos->x3d, pos->y3d, pos->z3d, &val) == HPGV_FALSE)
  {
    return HPGV_FALSE;
  }

  // now choose partial color

  val = (MYCOLORMAP_SIZE-1)*val;
  val = CLAMP(val, 0, (MYCOLORMAP_SIZE-1));
  int index = (int)val;
  index *= 4;

  partialcolor.r  =  mycolormap[index];
  partialcolor.g  =  mycolormap[index+1];
  partialcolor.b  =  mycolormap[index+2];
  partialcolor.a  =  mycolormap[index+3];

  return true;
};

// only supports serial execution at the moment

int main(int argc, char *argv[])
{

  // set up camera
  glCamera camera;
  camera.near = 70.0;
  camera.far = 100.0;
  camera.fov  = 40.0;
  const glfloat3 pos = glfloat3(-100.0, -100.0, -100.0);
  const glfloat3 lookat = glfloat3(0.0, 0.0, 0.0);
  const glfloat3 up = glfloat3(0.0, 0.0, -1.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  // get processor dimensions
  // switch x and z axes : convert column-major to row-major,
  int c = 1;
  int     npz = 1;
  int     npy = 1;
  int     npx = 1;

  int nx = DATA_SPHERECUBESIZE;
  int ny = DATA_SPHERECUBESIZE;
  int nz = DATA_SPHERECUBESIZE;

  __scrt_renderall_volume_init(MPI_COMM_WORLD, nx, ny, nz,
      WINDOW_WIDTH, WINDOW_HEIGHT, &camera, my_transfer_func);
  printf("finished __scrt_renderall_volume_init\n");

  // this is a duplication of what is done in volume_renderall constructor, but ok for now
  // generates part of grid for this mpi rank
  double *x, *y, *z;
  genTestGrid(0, npx, npy, npz, &nx, &ny, &nz, &x, &y, &z);
  printf("finished genTestGrid\n");

  // Set to the generated data (genTestData allocates space for the data).
  // Generates data for this mpi rank.
  void* data;
  genTestData(&data, nx, ny, nz, x, y, z); 
  printf("finished genTestData\n");

  __scrt_renderall_add_volume((float*)data, 0);

  // makes sure __sc_volume_renderall_data is mapped
  __scrt_renderall_begin();
  printf("finished __scrt_renderall_begin\n");

  // Run volume renderer on current data in __sc_volume_renderall_data
  // and draw.
  __scrt_renderall_end();
  printf("finished __scrt_renderall_end\n");

  sleep(10);

  // destroy
  __scrt_renderall_delete();
  printf("finished __scrt_renderall_delete\n");

  return 0;
}
