

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
 *
 */
#include <stdio.h>

#define COMB_DATA_DIM_X 507
#define COMB_DATA_DIM_Y 400
#define COMB_DATA_DIM_Z 100


uniform mesh AMeshType{
cells:
  float lifted_Y_OH;
};


int main(int argc, char *argv[])
{

  printf("Reading data\n");

  char filename [256];
  sprintf(filename, "/project/ccs7/projects/scout/data/combustion/lifted_Y_OH_0000.dat");
  float* data = (float*)malloc(sizeof(float)*COMB_DATA_DIM_X*COMB_DATA_DIM_Y*COMB_DATA_DIM_Z);
  FILE *fp = fopen(filename, "r");
  if (fp == 0) return 0;
  size_t n = fread(data, sizeof(float), COMB_DATA_DIM_X*COMB_DATA_DIM_Y*COMB_DATA_DIM_Z, fp);
  if (n == 0) return 0;
  fclose(fp);

  printf("Finished reading data -- now copy into mesh\n");

  // declare a 3d mesh
  AMeshType amesh[COMB_DATA_DIM_X, COMB_DATA_DIM_Y, COMB_DATA_DIM_Z];

  // copy data into mesh  -- really we need an efficient reader for meshes
  forall cells c of amesh {
    // datafile is in col-major order for array A[depth_size][col_size][row_size]
    // ((rowindex*col_size+colindex) * depth_size + depthindex)
    lifted_Y_OH = data[((c.position.z * COMB_DATA_DIM_Y + c.position.y) * 
        COMB_DATA_DIM_X + c.position.x)]; 
  }

  printf ("Finished setting data -- now volume rendering\n");

  float3 mypos = float3(350.0f, -100.0f, 650.0f);
  float3 mylookat = float3(350.0f, 200.0f, 25.0f); 
  float3 myup = float3(-1.0f, 0.0f, 0.0f);
      
  camera cam {
    near = 70.0;
    far = 500.0;
    fov = 40.0;
    pos = mypos;
    lookat = mylookat;
    up = myup;
  };

  // volume render data
  renderall cells c of amesh with cam {

    // choose partial color
    if (lifted_Y_OH > .0005) {
      color.r    =  1.0;
      color.g  =  0.0;
      color.b   =  0.0;
      color.a  =  1.0;
    } else {
      color.r    =  0.0;
      color.g  =  0.0;
      color.b   =  0.0;
      color.a  =  0.0;
    }

  }

  printf("done\n");

  return(0);
}
