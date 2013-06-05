
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

  forall cells c of amesh {
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
  
  float3 mypos = float3(-300.0f, -300.0f, -300.0f);
  float3 mylookat = float3(0.0f, 0.0f, 0.0f);
  float3 myup = float3(0.0f, 0.0f, -1.0f);

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
  sleep(3);
  return 0;
}
