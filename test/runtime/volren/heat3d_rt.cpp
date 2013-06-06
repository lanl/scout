/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * 3D heat transfer...
 *
 */

#include <iostream>
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/renderall/mpi/RenderallVolume.h"
#include <mpi.h>

#define XDIM 32 
#define YDIM 32 
#define ZDIM 32 
#define PRINTFIELD 0
#define NTIME_STEPS 25
#define MAX_TEMP 100.0f

static const size_t WINDOW_WIDTH = 768;
static const size_t WINDOW_HEIGHT = 768;

using namespace std;
using namespace scout;


void myhsva(float h, float s, float v, float a, float r[4])
{

  r[3] = a;

  int i;
  float f, p, q, t;
  if (s == 0.0f) {
    r[0] = v;
    r[1] = v;
    r[2] = v;
    return;
  }

  h /= 60.0f;
  i = int(h);
  f = h - float(i);
  p = v * (1.0 - s);
  q = v * (1.0 - s * f);
  t = v * (1.0 - s * (1.0 - f));

  switch(i) {

    case 0:
      r[0] = v;
      r[1] = t;
      r[2] = p;
      break;

    case 1:
      r[0] = q;
      r[1] = v;
      r[2] = p;
      break;

    case 2:
      r[0] = p;
      r[1] = v;
      r[2] = t;
      break;

    case 3:
      r[0] = p;
      r[1] = q;
      r[2] = v;
      break;

    case 4:
      r[0] = t;
      r[1] = p;
      r[2] = v;
      break;

    default:
      r[0] = v;
      r[1] = p;
      r[2] = q;
      break;
  }

}


// generates test data for one volume
void genTestData(float **t1, float **t2, int nx, int ny, int nz)
{

    /* volume */
    float *test_data1 = (float *)malloc(nx * ny * nz * sizeof(float));
    if (test_data1 == NULL) {
      printf("Out of memory!\n");
    }
    float *test_data2 = (float *)malloc(nx * ny * nz * sizeof(float));
    if (test_data2 == NULL) {
      printf("Out of memory!\n");
    }

    int px, py, pz;
    float value1, value2;

    int i;

    for (pz = 0; pz < nz; pz++) {
	    for (py = 0; py < ny; py++) {
		    for (px = 0; px < nx; px++) {

			    int id = px + py * nx + pz * nx * ny;

			    if ((px == 0) || (px == nx-1)
					    || (py == 0) || (py == ny-1)
					    || (pz == 0) || (pz == nz-1)) {
				    test_data1[id] = 0;
				    test_data2[id] = 0;
			    } else {
				    test_data1[id] = MAX_TEMP;
				    test_data2[id] = MAX_TEMP;
			    }


		    }
	    }
    }

  *t1 = test_data1;
  *t2 = test_data2;
}

void printTestData(void *ptest_data, int nx, int ny, int nz)
{

    int px, py, pz;

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

#define indexof(x, y, z) ((z*YDIM + y)*XDIM + x)

void do_heat_xfer_step(float* t1, float* t2, int nx, int ny, int nz, 
    float dx, float dy, float dz, float alpha, float dt) {

  int px, py, pz;
  for (pz = 0; pz < nz; pz++) {
    for (py = 0; py < ny; py++) {
      for (px = 0; px < nx; px++) {

        if (px > 0 && px < nx-1 &&
            py > 0 && py < ny-1 &&
            pz > 0 && pz < nz-1) { 

          float t1val = t1[indexof(px, py, pz)];
          float d2dx2 = t1[indexof(px+1, py, pz)] - 2.0f * t1val + t1[indexof(px-1, py, pz)];
          d2dx2 /= dx * dx;

          float d2dy2 = t1[indexof(px, py+1, pz)] - 2.0f * t1val + t1[indexof(px, py-1, pz)];
          d2dy2 /= dy * dy;

          float d2dz2 = t1[indexof(px, py, pz+1)] - 2.0f * t1val + t1[indexof(px, py, pz-1)];
          d2dz2 /= dz * dz;        

          float val = (alpha * dt * (d2dx2 + d2dy2 + d2dz2)) + t1val;

          t2[indexof(px,py,pz)] = val;
          t1[indexof(px,py,pz)] = val;
        }
      }
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

  float norm_t1;
  if (val > MAX_TEMP) {
    norm_t1 = 1.0f;
  } else {
    norm_t1 = val / MAX_TEMP;
  } 

  float hue = 240.0f - 240.0f * norm_t1;
  float thiscolor[4];
  myhsva(hue, 1.0f, 1.0f, norm_t1, thiscolor);

  partialcolor.r = thiscolor[0];
  partialcolor.g = thiscolor[1];
  partialcolor.b = thiscolor[2];
  partialcolor.a = thiscolor[3];

  return true;
};


// only supports serial execution at the moment

int main(int argc, char *argv[])
{

  // set up camera
  glCamera camera;
  camera.near = 64.0;
  camera.far = 200.0;
  camera.fov  = 95.0;
  const glfloat3 pos = glfloat3(32.0, 32.0, 100.0);
  const glfloat3 lookat = glfloat3(32.0, 32.0, 0.0);
  const glfloat3 up = glfloat3(0.0, 1.0, 0.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  // get processor dimensions
  // switch x and z axes : convert column-major to row-major,
  int nx = XDIM;
  int ny = YDIM;
  int nz = ZDIM;

  __scrt_renderall_volume_init(MPI_COMM_WORLD, nx, ny, nz, 
      WINDOW_WIDTH, WINDOW_HEIGHT, &camera, my_transfer_func);
  printf("finished __sc_init_volume_renderall\n");

  // Set to the generated data (genTestData allocates space for the data).
  // Generates data for this mpi rank.
  float* t1, *t2;
  genTestData(&t1, &t2, nx, ny, nz); 
  printf("finished genTestData\n");

  __scrt_renderall_add_volume(t1, 0);

  // compute increments
  const float dx    = 1.0f / XDIM;
  const float dy    = 1.0f / YDIM;
  const float dz    = 1.0f / ZDIM;  
  const float alpha = 0.00039f; // good for 32x32x32
  const float dt  = 0.4 * (alpha / 4.0f) * ((1.0f / (dx * dx)) + (1.0f / (dy * dy)) + (1.0f / (dz * dz)));
  printf("dt: %f\n", dt);

  for (unsigned int t = 0; t < NTIME_STEPS; ++t) {

  // makes sure __sc_volume_renderall_data is mapped
    __scrt_renderall_begin();
    printf("finished __sc_begin_renderall\n");

    do_heat_xfer_step(t1, t2, nx, ny, nz, dx, dy, dz, alpha, dt);
    printf("finished do_heat_xfer_step\n");

    // Run volume renderer on current data in __sc_volume_renderall_data
    // and draw.
    __scrt_renderall_end();
    printf("finished __sc_end_renderall\n");

  }

  // destroy
  __scrt_renderall_delete();
  printf("finished __sc_delete_renderall\n");

  return 0;
}

