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
#include "scout/Runtime/renderall/mpi/RenderallVolume.h"
#include <mpi.h>

using namespace std;
using namespace scout;


static const size_t WINDOW_WIDTH = 768;
static const size_t WINDOW_HEIGHT = 768;

#define COMB_DATA_DIM_X 507
#define COMB_DATA_DIM_Y 400
#define COMB_DATA_DIM_Z 100


void genTestData(int id, void **ptest_data,
  int npx, int npy, int npz, int nx, int ny, int nz)
{
  static int sizes[3] = {COMB_DATA_DIM_X, COMB_DATA_DIM_Y, COMB_DATA_DIM_Z};

  // my processor dimension
  int mypz = id /(npx * npy);
  int mypx = (id - mypz * npx * npy) % npx;
  int mypy = (id - mypz * npx * npy) / npx;

  // the starting point of my data 
  static int start[3];
  start[0] = mypx * nx;
  start[1] = mypy * ny;
  start[2] = mypz * nz;

  // how many data points for each dim
  static int count[3];
  count[0] = nx;
  count[1] = ny;
  count[2] = nz;

  int numelts = count[0] * count[1] * count[2];

  float *test_data = (float *)malloc(numelts*sizeof(float));
  if (test_data == NULL) {
    printf("Out of memory!\n");
  }

  char filename [256];
  sprintf(filename, "/project/ccs7/projects/scout/data/combustion/lifted_Y_OH_0000.dat");

  static MPI_Datatype filetype;
  static int totalsize = 0;

  MPI_File fd;
  MPI_Status status;
  MPI_Datatype datatype = MPI_FLOAT;

  if (MPI_Type_create_subarray(3, sizes, count, start,
        MPI_ORDER_FORTRAN,
        datatype, &filetype) != MPI_SUCCESS )
  {
    HPGV_ABORT("Can not create subarray", HPGV_ERR_IO);
  }

  if (MPI_Type_commit(&filetype) != MPI_SUCCESS) {
    HPGV_ABORT("Can not commit file type", HPGV_ERR_IO);
  }

  if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
        &fd) != MPI_SUCCESS)
  {
    HPGV_ABORT("Can not open file", HPGV_ERR_IO);
  }

  MPI_File_set_view(fd, 0, datatype, filetype, (char *)"native", MPI_INFO_NULL);

  if (MPI_File_read_all(fd, test_data, numelts, datatype,
        &status) != MPI_SUCCESS)
  {
    HPGV_ABORT("Can not read file", HPGV_ERR_IO);
  }
 
  int statuscount;
  MPI_Get_count(&status, datatype, &statuscount);

  HPGV_ASSERT_P(id, statuscount == numelts,
     "Inconsistent read", HPGV_ERR_IO);

  MPI_File_close(&fd);

  *ptest_data = (void*)test_data;
}


void printTestData(int id, void *ptest_data,
  int nx, int ny, int nz, double* x, double* y, double* z)
{

    int px, py, pz;
    float value;

   printf("id: %d nx: %d, ny: %d, nz: %d\n", id, nx, ny, nz);

   int count = 0;
 
   for (pz = 0; pz < nz; pz++) {
        for (py = 0; py < ny; py++) {
            for (px = 0; px < nx; px++) {
                int id = px + py * nx + pz * nx * ny;
                printf("%f ", ((float*)ptest_data)[id]);
                count++;
            }
            printf("\n");
        }
      //if (count >= 1000) return;
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
  if (val > .0005) {
    partialcolor.r    =  1.0;
    partialcolor.g  =  0.0;
    partialcolor.b   =  0.0;
    partialcolor.a  =  1.0;
  } else {
    partialcolor.r    =  0.0;
    partialcolor.g  =  0.0;
    partialcolor.b   =  0.0;
    partialcolor.a  =  0.0;
  }

  return true;
};

// only supports args "1 1 1" at the moment

int main(int argc, char *argv[])
{

#ifdef NOTNOW
  // set up camera
  glCamera camera;
  camera.near = 70.0;
  camera.far = 500.0;
  camera.fov  = 40.0;
  
  const glfloat3 pos = glfloat3(350.0, -100.0, 650.0);
  const glfloat3 lookat = glfloat3(350.0, 200.0, 25.0);
  const glfloat3 up = glfloat3(-1.0, 0.0, 0.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);
#endif

  int     npz = 1;
  int     npy = 1;
  int     npx = 1;

  int nx = COMB_DATA_DIM_X;
  int ny = COMB_DATA_DIM_Y;
  int nz = COMB_DATA_DIM_Z;

  int id = 0;

  printf("%d: genTestGrid...\n", id);

  printf("%d: __scrt_renderall_volume_init...\n", id);
  __scrt_renderall_volume_init(MPI_COMM_WORLD, nx, ny, nz,
      WINDOW_WIDTH, WINDOW_HEIGHT, NULL, my_transfer_func);

  nx = nx / npx;
  ny = ny / npy;
  nz = nz / npz;

  void* data;

  // Set to the generated data (genTestData allocates space for the data).
  genTestData(id, &data, npx, npy, npz, nx, ny, nz); 

  __scrt_renderall_add_volume((float*)data, 0);

  printf("%d: __scrt_renderall_begin...\n", id);
  // makes sure __sc_volume_renderall_data is mapped
  __scrt_renderall_begin();

  printf("%d: genTestData...\n", id);

  // Run volume renderer on current data in __sc_volume_renderall_data
  // and draw.
  printf("%d: __scrt_renderall_end...\n", id);
  __scrt_renderall_end();

  sleep(10);

  // destroy
  printf("id: %d: __scrt_renderall_delete...\n", id);
  __scrt_renderall_delete();


  return 0;
}
