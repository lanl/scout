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
 
  // -----------------------------------------------------------------------------------------------
  // Begin time-marching loop
  // -----------------------------------------------------------------------------------------------

  double Time = 0.;

  int count = 0;
  
  
  FILE *v_l = fopen("verts_L.txt", "w");
  fprintf(v_l, "# Vertex index\n");
  fprintf(v_l, "# Col 1 x\n");
  fprintf(v_l, "# Col 2 y\n");
  fprintf(v_l, "# Col 3 u\n");
  fprintf(v_l, "# Col 4 v\n");
  fprintf(v_l, "# Col 5 U\n");
  fprintf(v_l, "# Col 6 V\n");
  fprintf(v_l, "# Col 7 M\n");
  
  FILE *v_r = fopen("verts_R.txt", "w");
  fprintf(v_l, "# Vertex index\n");
  fprintf(v_r, "# Col 1 x\n");
  fprintf(v_r, "# Col 2 y\n");
  fprintf(v_r, "# Col 3 u\n");
  fprintf(v_r, "# Col 4 v\n");
  fprintf(v_r, "# Col 5 U\n");
  fprintf(v_r, "# Col 6 V\n");
  fprintf(v_r, "# Col 7 M\n");
  
  FILE *c_l = fopen("cells_L.txt", "w");
  fprintf(c_l, "# Cell index\n");
  fprintf(c_l, "# Col 1 x\n");
  fprintf(c_l, "# Col 2 y\n");
  fprintf(c_l, "# Col 3 p\n");
  fprintf(c_l, "# Col 4 rho\n");
  fprintf(c_l, "# Col 5 e\n");
  fprintf(c_l, "# Col 6 M\n");
  
  FILE *c_r = fopen("cells_R.txt", "w");
  fprintf(c_l, "# Cell index\n");
  fprintf(c_r, "# Col 1 x\n");
  fprintf(c_r, "# Col 2 y\n");
  fprintf(c_r, "# Col 3 p\n");
  fprintf(c_r, "# Col 4 rho\n");
  fprintf(c_r, "# Col 5 e\n");
  fprintf(c_r, "# Col 6 M\n");
  
  
  
  while ( Time < endtime)
    {
      count++;
      // Increment time
      Time += dt;
     
      //printf("Before Lagrange\n"); 
      LagrangeStep(&M, count, Time);
      //printf("After Lagrange\n"); 
      
      
      
      fprintf(v_l, "\nAfter %i time steps\n", count);
      fprintf(c_l, "\nAfter %i time steps\n", count);
      forall vertices v in M
      {
          fprintf(v_l, "%d,%d\t %f \t %f \t %f \t %f \t %e \t %f \t %f \n", positionx(), positiony(), v.v_x_new, v.v_y_new, v.v_u_new, v.v_v_new, v.v_U_new, v.v_V_new, v.v_M_new);
      }// end forall vertices loop
      
      forall cells c in M
      {
          fprintf(c_l, "%d,%d\t %f \t %f \t %f \t %f \t %f \t %e \n", positionx(), positiony(), c.c_x_new, c.c_y_new, c.c_p_new, c.c_r_new, c.c_e_new, c.c_M_new);
      }// end forall cells loop
      fprintf(v_l, "\n");
      fprintf(c_l, "\n");
      
      
      //printf("Before RemapStep\n"); 
      RemapStep(&M, count, Time);
      //printf("After RemapStep\n"); 
      
      
      
      
      fprintf(v_r, "\nAfter %i time steps\n", count);
      fprintf(c_r, "\nAfter %i time steps\n", count);
      forall vertices v in M
      {
          fprintf(v_r, "%d,%d\t %f \t %f \t %f \t %f \t %e \t %f \t %f \n", positionx(), positiony(), v.v_x_new, v.v_y_new, v.v_u_new, v.v_v_new, v.v_U_new, v.v_V_new, v.v_M_new);
      }// end forall vertices loop
      
      forall cells c in M
      {
        {
          fprintf(c_r, "%d,%d\t %f \t %f \t %f \t %f \t %f \t %f \t %e \n", positionx(), positiony(), c.c_x_new, c.c_y_new, c.c_p_new, c.c_r_new, c.c_e_new, c.c_M_new, c.c_U);
          //fprintf(temp, "%f %f \n", c.c_x_new, c.c_r_new);
        }
      }// end forall cells loop
      fprintf(v_r, "\n");
      fprintf(c_r, "\n");
     
#if 0
      if (Time <= endtime) { 
        renderall vertices v in M to win 
        { 
          float norm_v = v_u_new;
          float hue = 240.0f - 240.0f * norm_v;
          color = hsv(hue, 1.0f, 1.0f);
        }
        renderall cells c in M to win 
        { 
          //float norm_p = c_p_new / high_pressure;
          //float hue = 240.0f - 240.0f * norm_p;
          float norm_r = c_r_new / high_density;
          float hue = 240.0f - 240.0f * norm_r;
          color = hsv(hue, 1.0f, 1.0f);
        }
        usleep(250000);
      }
#endif
    }

  return 0;
}
