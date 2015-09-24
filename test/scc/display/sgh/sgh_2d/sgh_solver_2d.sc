/* 2D Scout implementation of SGH approach
 * Joe Bottini and Sriram Nagaraj
 */
 
#include "sgh_solver_2d.sch"
#include <stdio.h>
#include <stdlib.h>

//----------------------------------------------------------------------------------
//                                    Main Method
//----------------------------------------------------------------------------------

extern task void initializeMesh(sghMesh *M);
extern void LagrangeStep(sghMesh *M, int count, double Time);
extern void RemapStep(sghMesh *M, int count, double Time);
int main()
{
  // -----------------------------------------------------------------------------------------------
  // Initializations
  // -----------------------------------------------------------------------------------------------

  sghMesh M[xcells, ycells];
  initializeMesh(&M); // Check for interior does not occur in initialize methods, but it does in update

  window win[512,512]; 
  window win2[512,512]; 

  // -----------------------------------------------------------------------------------------------
  // Begin time-marching loop
  // -----------------------------------------------------------------------------------------------

  double Time = 0.;

  int count = 0;

  while ( Time < endtime)
  {
    count++;
    // Increment time
    Time += dt;

    LagrangeStep(&M, count, Time);

    RemapStep(&M, count, Time);

    if (Time <= endtime) { 
      renderall cells c in M to win 
      { 
        //float norm_p = c_p_new / high_pressure;
        //float hue = 240.0f - 240.0f * norm_p;
        float norm_r = c_r_new / high_density;
        float hue = 240.0f - 240.0f * norm_r;
        color = hsv(hue, 1.0f, 1.0f);
      }
      usleep(30000);
    }
  }

  Time = 0.;

  count = 0;

  while ( Time < endtime)
  {
    count++;
    // Increment time
    Time += dt;

    LagrangeStep(&M, count, Time);

    RemapStep(&M, count, Time);

    if (Time <= endtime) { 
      renderall vertices v in M to win2 
      { 
        float norm_v = v_u_new;
        float hue = 240.0f - 240.0f * norm_v;
        color = hsv(hue, 1.0f, 1.0f);
      }
      usleep(30000);
    }
  }

  return 0;
}
