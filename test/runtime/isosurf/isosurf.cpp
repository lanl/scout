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
#include "scout/Runtime/isosurf/isosurface_cpp.h"
#include "scout/Runtime/renderall/RenderallSurface.h"
#include "scout/Runtime/isosurf/piston/hsv_color_map.h"
#include "scout/Runtime/isosurf/user_defined_color_func.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include "scout/Runtime/init_mac.h"
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#define GRIDSIZE 64 
#define SQR(x) ((x) * (x))
#define MAX(x, y) ((x) > (y)? (x) : (y))
#define MIN(x, y) ((x) < (y)? (x) : (y))
#define CLAMP(x, minval, maxval) (MIN(MAX(x, (minval)), (maxval)))

using namespace std;
using namespace scout;


static const size_t WINDOW_WIDTH = 1024;
static const size_t WINDOW_HEIGHT = 1024;

// generates test grid 
void genTestGrid(int id, int npx, int npy, int npz, 
  int* pnx, int* pny, int* pnz, double** ptrx, double** ptry, double** ptrz)
{
    int domain_grid_size[3];

    domain_grid_size[0] = GRIDSIZE;
    domain_grid_size[1] = GRIDSIZE;
    domain_grid_size[2] = GRIDSIZE;

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

void genTangle(void **ptest_data, int nx, int ny, int nz)
{

  float *test_data = (float *)malloc(nx * ny * nz * sizeof(float));
  if (test_data == NULL) {
    printf("Out of memory!\n");
  }

  int px, py, pz;

  float xscale = 2.0f/(nx-1);
  float yscale = 2.0f/(ny-1);
  float zscale = 2.0f/(nz-1);

  for (pz = 0; pz < nz; pz++) {
    for (py = 0; py < ny; py++) {
      for (px = 0; px < nx; px++) {

        float x = 3.0f*(px*xscale -1.0f);
        float y = 3.0f*(py*yscale -1.0f);
        float z = 3.0f*(pz*zscale -1.0f);
        const float v = (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;

        int id = px + py * nx + pz * nx * ny;

        test_data[id] = v;
      }
    }
  }

  *ptest_data = (void*)test_data;
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
       c[i] = (float)(GRIDSIZE-1.0)/2.0;
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

                value = 1.0 - sqrt(value) / (float)(GRIDSIZE-1);

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


// only supports serial execution at the moment

int main(int argc, char *argv[])
{

  int     npz = 1;
  int     npy = 1;
  int     npx = 1;

  int nx = GRIDSIZE;
  int ny = GRIDSIZE;
  int nz = GRIDSIZE;

  // baby example, just renders one triangle, doesn't do isosurfacing
  glCamera camera;
  camera.near = 1.0;
  camera.far = 1000.0;
  camera.fov  = 60.0;
  glfloat3 pos = glfloat3(0.0, 0.0, 15.0);
  glfloat3 lookat = glfloat3(5.0, 5.0, 0.0);
  glfloat3 up = glfloat3(0.0, 1.0, 0.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  float verts_simple[12] = {0.0, 0.0, 0.0, 1.0,
    5.0, 5.0, 0.0, 1.0,
    10.0, 0.0, 0.0, 1.0};
  float normals_simple[9] = {0.0, 0.0, -1.0, 0, 0, -1.0, 0.0, 0.0, -1.0};
  float scalars_simple[3] = {.2, .2, .2};
  float colors_simple[12] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

  __scrt_renderall_surface_begin(20, 20, 20, verts_simple, normals_simple, colors_simple, 3, &camera);

  __scrt_renderall_end();

  sleep(5);

  __scrt_renderall_delete();

  // tangle example from PISTON, does isosurfacing
  camera.near = 1.0;
  camera.far = GRIDSIZE*4.0;
  camera.fov  = 60.0;
  pos = glfloat3(0.0, 0.0, GRIDSIZE*1.5);
  lookat = glfloat3(GRIDSIZE/2, GRIDSIZE/2, GRIDSIZE/2);
  up = glfloat3(0.0, 1.0, 0.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  void* data;

  genTangle(&data, nx, ny, nz);

  float isoval = .2f;
  printf("isoval: %f\n", isoval);


  __sc_isosurface(nx, ny, nz, (float*)data, (float*)data, isoval);

  float* verts = __sc_isosurface_cpp->getVertices();
  float* norms = __sc_isosurface_cpp->getNormals();
  float* scalars = __sc_isosurface_cpp->getScalars();

  // allocate colors
  int num_vertices = __sc_isosurface_cpp->getNumVertices();

  thrust::host_vector<psfloat4>* colors_host_vector = new thrust::host_vector<psfloat4>(num_vertices);

  float* colors = (float*)thrust::raw_pointer_cast(&(*colors_host_vector)[0]);

  for (int i = 1; i <= 10; i++) {

    // user-defined color function
    color_func_t my_color_func = ^ psfloat4 (float scalar, float min, float max)
    {
      psfloat4 color;

      // user code here
      color.r = .10*i;
      color.g = 1.0- .10*i;
      color.b = 0.0;
      color.a = 1.0;
      // end user code

      return make_psfloat4(color.r, color.g, color.b, color.a);
    };

    // new way of doing it
    user_defined_color_func ud_color_func(my_color_func, 0.0, 1.0);

    // this will not be interesting, because all scalars are the isoval
    __sc_isosurface_cpp->computeColors2(colors_host_vector, ud_color_func);

    // render surface
    if (__sc_isosurface_cpp->getNumVertices() > 0) {
      __scrt_renderall_surface_begin(nx, ny, nz, __sc_isosurface_cpp->getVertices(),
          __sc_isosurface_cpp->getNormals(), colors, 
          __sc_isosurface_cpp->getNumVertices(), &camera);

	    __scrt_renderall_end();

	    sleep(1);

	    __scrt_renderall_delete();

    }
  }
 
  // cleanup isosurface
  if (__sc_isosurface_cpp) {
    delete __sc_isosurface_cpp;
    __sc_isosurface_cpp = NULL;
  }

  free(data);

  // Example from Hongfeng's volume renderer

  // set up camera
  camera.near = 70.0;
  camera.far = 300.0;
  camera.fov  = 30.0;
  pos = glfloat3(31.5, 31.5, 242.0);
  lookat = glfloat3(31.5, 31.5, 31.5);
  up = glfloat3(0, 1, 0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  double *x, *y, *z;
  genTestGrid(0, npx, npy, npz, &nx, &ny, &nz, &x, &y, &z);

  // Set to the generated data (genTestData allocates space for the data).
  genTestData(&data, nx, ny, nz, x, y, z); 

  // iterate through creating different isosurfaces through the volume
  for (int i = 14; i < 100; i+=3) {

    float isoval = float(i)*.01;
    printf("isoval: %f\n", isoval);

    if (__sc_isosurface_cpp) {
      __sc_recompute_isosurface(isoval);
    } else {
      // good for now but would like to have a different volume for computing scalars
      __sc_isosurface(nx, ny, nz, (float*)data, (float*)data, isoval);
    }

    verts = __sc_isosurface_cpp->getVertices();
    norms = __sc_isosurface_cpp->getNormals();
    scalars = __sc_isosurface_cpp->getScalars();

    // we get control over color computation
    // use this if we get a different volume for computing scalars
    //min_scalar = __sc_isosurface_cpp->getMinScalar();
    //max_scalar = __sc_isosurface_cpp->getMaxScalar();

    // allocate colors
    num_vertices = __sc_isosurface_cpp->getNumVertices();

    colors_host_vector->resize(num_vertices);

    // iterate over colors and assign
    piston::hsv_color_map<float> color_func2(0.0, 1.0);

    __sc_isosurface_cpp->computeColors(colors_host_vector, color_func2);

    colors = (float*)thrust::raw_pointer_cast(&(*colors_host_vector)[0]);

    // render isosurface
    if (__sc_isosurface_cpp->getNumVertices() > 0) {
      __scrt_renderall_surface_begin(nx, ny, nz, __sc_isosurface_cpp->getVertices(),
          __sc_isosurface_cpp->getNormals(), colors, __sc_isosurface_cpp->getNumVertices(), &camera);

      __scrt_renderall_end();
      __scrt_renderall_delete();
    }
    
    if (__sc_isosurface_cpp) {
      delete __sc_isosurface_cpp;
      __sc_isosurface_cpp = NULL;
    }
  }

  delete colors_host_vector;
  free(data);

  return 0;
}
