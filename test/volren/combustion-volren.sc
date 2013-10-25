/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * Simplistic volume rendering...
 *
 * ##### 
 */ 

#include <stdio.h>

#define COMB_DATA_DIM_X 507
#define COMB_DATA_DIM_Y 400
#define COMB_DATA_DIM_Z 100


uniform mesh AMeshType{
cells:
  float lifted_Y_OH;
};

//SC_TODO: temporary workaround for broken scout vectors in camera
typedef float float3d __attribute__((ext_vector_type(3)));

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
  forall cells c in amesh {
    // datafile is in col-major order for array A[depth_size][col_size][row_size]
    // ((rowindex*col_size+colindex) * depth_size + depthindex)
    lifted_Y_OH = data[((c.position.z * COMB_DATA_DIM_Y + c.position.y) * 
        COMB_DATA_DIM_X + c.position.x)]; 
  }

  printf ("Finished setting data -- now volume rendering\n");

  //SC_TODO: temporary workaround for broken scout vectors in camera
  float3d mypos = (float3d){350.0f, -100.0f, 650.0f};
  float3d mylookat = (float3d){350.0f, 200.0f, 25.0f}; 
  float3d myup = (float3d){-1.0f, 0.0f, 0.0f};
      
  camera cam {
    near = 70.0;
    far = 500.0;
    fov = 40.0;
    pos = mypos;
    lookat = mylookat;
    up = myup;
  };

  // volume render data
  renderall cells c in amesh with cam {
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
