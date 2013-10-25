<<<<<<< HEAD

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
=======
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

>>>>>>> 331f45ad55fb625f198d765bff49b3d4fc0a6ce5
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "mycolormap.h"

#define DATA_SPHERECUBESIZE 64
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

// only supports args "1 1 1" at the moment

int main(int argc, char *argv[])
{

  float center[3];

  int i;

  /* ----- sphere ----- */
  for (i = 0; i < 3; i++) {
     center[i] = (float)(DATA_SPHERECUBESIZE-1.0)/2.0;
  }

  // set up mesh


  AMeshType amesh[DATA_SPHERECUBESIZE,DATA_SPHERECUBESIZE,DATA_SPHERECUBESIZE];

  // generate data in mesh

<<<<<<< HEAD
  forall cells c of amesh {
=======
  forall cells c in amesh {
>>>>>>> 331f45ad55fb625f198d765bff49b3d4fc0a6ce5
    float p[3];
    p[2] = (float)c.position.x;
    p[1] = (float)c.position.y;
    //p[0] = DATA_SPHERECUBESIZE/2;
    p[0] = (float)c.position.z;

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

<<<<<<< HEAD
  renderall cells c of amesh with cam {
=======
  renderall cells c in amesh with cam {
>>>>>>> 331f45ad55fb625f198d765bff49b3d4fc0a6ce5
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
  sleep(3);
  return 0;
}
