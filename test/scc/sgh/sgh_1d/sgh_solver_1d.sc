/*
This scout code is the 1D scout 
code corresponding to the cpp code
sgh_solver.cpp

Date modified: Jul 6, 2015

*/

#include <stdio.h>
#include <assert.h>

// Global Variable Declaration and Initialization
int    npts_global=6          ,  ncells_global=5;
double length_global=1.0;
double dt = 1.E-4      ,  endtime = 0.5;
double init_density_global=1.0,  init_energy_global=0.01;
double pistonspeed_global=1.0 ,  gmama_global=5.0/3.0;
double q1_global=1.0          ,  q2_global=1.0;


// sghMesh declaration and field declaration
uniform mesh sghMesh{

   cells:
     double c_density_old  , c_density_new;
     double c_energy_old   , c_energy_new;
     double c_pressure_old , c_pressure_new;
     double c_pos_old      , c_pos_new;
     double c_q;
     // Remap variables
     double M, P, E;
     double M_new, P_new, E_new;
     double x_left, x_right;
     
     int c_index;

   vertices:
     double v_pos_old, v_pos_new;
     double v_velocity_old, v_velocity_new;
     double v_force;
     
     int v_index;
};

void printMesh(sghMesh* m){
  forall cells c in *m {
    printf("cell position: %d \n", positionx());
    printf("  c_density_old: %f \n", c_density_old);
    printf("  c_density_new: %f \n", c_density_new);
    printf("  c_energy_old: %f \n", c_energy_old);
    printf("  c_energy_new: %f \n", c_energy_new);
    printf("  c_pressure_old: %f \n", c_pressure_old);
    printf("  c_pressure_new: %f \n", c_pressure_new);
    printf("  c_pos_old: %f \n", c_pos_old);
    printf("  c_pos_new: %f \n", c_pos_new);
    printf("  c_q: %f \n", c_q);
    printf("  M: %f \n", M);
    printf("  P: %f \n", P);
    printf("  E: %f \n", E);
    printf("  M_new: %f \n", M_new);
    printf("  P_new: %f \n", P_new);
    printf("  E_new: %f \n", E_new);
    printf("  x_left: %f \n", x_left);
    printf("  x_right: %f \n", x_right);
    printf("  c_index: %d \n", c_index);
  }
  forall vertices v in *m {
    printf("vertex position: %d \n", positionx());
    printf("  v_pos_old: %f \n", v_pos_old);
    printf("  v_pos_new: %f \n", v_pos_new);
    printf("  v_velocity_old: %f \n", v_velocity_old);
    printf("  v_velocity_neg: %f \n", v_velocity_new);
    printf("  v_force: %f \n", v_force);
    printf("  v_index: %d \n", v_index);
  }
  printf("\n\n");
}

double getPressure(double c_den, double c_en, double gmama){
    return (gmama-1)*c_den*c_en;
}


double getSoundspeed(double c_en, double gmama){
    return gmama*(gmama-1)*c_en;                          // Expression for speed of sound - may be incorrect (sqrt missing)
}

task void setMesh(sghMesh* m, double gmama, double init_energy, double init_density, double dx){

   forall cells c in *m{

     c.c_density_old=init_density;
     c.c_density_new=init_density;

     c.c_energy_old=init_energy;
     c.c_energy_new=init_energy;

     c.c_pos_old=dx/2. + positionx() * dx;
     c.c_pos_new=dx/2. + positionx() * dx;
     
     c.c_pressure_old = getPressure(init_density, init_energy, gmama );
     c.c_pressure_new = getPressure(init_density, init_energy, gmama );
     c.c_q = 0.;
     c.c_index = positionx();
    }
   
   forall vertices v in *m{

     v.v_pos_old=positionx() * dx;
     v.v_pos_new=positionx() * dx;
     
     v.v_velocity_old=0.0;
     v.v_velocity_new=0.0;

     v.v_force=0.0;
     
     v.v_index = positionx();
    }
   //printMesh(m);
    //--------------------------------------------------------------------------- Depends on previous forall loop in task
   forall cells c in *m {
     double temps1[2] = {0, 0};
     forall vertices v in c{
       temps1[positionx()] = v.v_pos_old;
     }
     c.x_left = temps1[0];
     c.x_right = temps1[1];
   }

   //--------------------------------------------------------------------------- Depends on previous forall loop in task
   forall cells c in *m{
     double temps2[2] = {0, 0};
     forall vertices v in c{
       temps2[v.v_index % 2] = v.v_velocity_old/2;
     }
     double avgVel = temps2[0] + temps2[1];
     c.M = c.c_density_old * (c.x_right - c.x_left);
     c.P = c.c_density_old * (c.x_right - c.x_left) * avgVel;
     c.E = c.c_density_old * (c.x_right - c.x_left) * c.c_energy_old;
     //c.E = c.c_density_old * (c.x_right - c.x_left) * (c.c_energy_old + 1./2. * avgVel^2);                                       // Add Conversion Factor - From Scott
   }
}


// Updated method
task void updatePressure(sghMesh* m){
   forall cells c in *m{
     c.c_pressure_old = c.c_pressure_new;
   }
}



task void updateFields(sghMesh* m){
  forall cells c in *m{
    c.c_pos_old = c.c_pos_new;
    c.c_density_old = c.c_density_new;
    c.c_energy_old = c.c_energy_new;
    c.c_pressure_old = c.c_pressure_new;
    c.c_q = 0.;
  }
  forall vertices v in *m{
    v.v_pos_old = v.v_pos_new;
    v.v_velocity_old = v.v_velocity_new;
    v.v_force = 0.;
  }
  //--------------------------------------------------------------------------- Depends on previous forall loop in task
  // Update variables for remap
  forall cells c in *m{
    double temps1[2] = {0, 0};
    forall vertices v in c{
       temps1[positionx()] = v.v_pos_old;
     }
     c.x_left = temps1[0];
     c.x_right = temps1[1];
    
  }
  //--------------------------------------------------------------------------- Depends on previous forall loop in task
  forall cells c in *m{
    double temps2[2] = {0, 0};
    forall vertices v in c{
      temps2[v.v_index % 2] = v.v_velocity_old/2;
    }
    double avgVel = temps2[0] + temps2[1];
    c.M = c.c_density_old * (c.x_right - c.x_left);
    c.P = c.c_density_old * (c.x_right - c.x_left) * avgVel;
    c.E = c.c_density_old * (c.x_right - c.x_left) * c.c_energy_old;
    //c.E = c.c_density_old * (c.x_right - c.x_left) * (c.c_energy_old + 1./2. * avgVel^2);                       // Add Conversion Factor - From Scott
  }
}


task void updateMesh(sghMesh* m, double mydt, double gmama, double q1, double q2, int npts, double pistonspeed){
   // Calculate artificial viscosity contribution in each cell
   forall cells c in *m{
     double du_1[2] = {0.,0.};
     forall vertices v in c{
       du_1[v.v_index % 2] = v.v_velocity_old * (v.v_pos_old - c.c_pos_old) / fabs(v.v_pos_old - c.c_pos_old); // Change when implementation available
                              // Will remain consistent with ^ this notation
     }
     double du = du_1[0] + du_1[1];                                                                              // Change when implementation available
     
     double ss=getSoundspeed(c.c_energy_old, gmama);
     double q=q1*c.c_density_old*fabs(du)*ss        +       q2*c.c_density_old*du*du;
     
     
     if(du>0.0) q=0.0;
     c.c_q = q;
   } // First part done, updated c_q
   
   //--------------------------------------------------------------------------- Depends on previous forall loop in task
   // Calculate force exerted and induced velocity on each vertex
   forall vertices v in *m{
     if (positionx() != 0 && positionx() != npts - 1){
       double totalForce_1[2] = {0., 0.};
       double totalMass_1[2]  = {0., 0.};
       forall cells c in v{
         totalForce_1[c.c_index % 2] = (c.c_pressure_old + c.c_q) * (v.v_pos_old - c.c_pos_old) / fabs(v.v_pos_old - c.c_pos_old); // Change when implementation available
         
         totalMass_1[c.c_index % 2]  = c.M/2;
       }
       
       double totalForce = totalForce_1[0] + totalForce_1[1];                                                    // Change when implementation available
       double totalMass  = totalMass_1[0]  + totalMass_1[1];
       v.v_force = totalForce;
       v.v_velocity_new = v.v_velocity_old + mydt * totalForce / totalMass;
     }
     else if (positionx() == 0){
       v.v_force = 0;              // Useless, not used
       v.v_velocity_new = pistonspeed; // Boundary Condition #1
     }
     else{
       v.v_force = 0;              // Useless, not used
       v.v_velocity_new = 0;           // Boundary Condition #2
     }
   }
   //--------------------------------------------------------------------------- Depends on previous forall loop in task
   // Calculate the change in position for the vertices
   forall vertices v in *m{
     v.v_pos_new = v.v_pos_old + mydt * v.v_velocity_new;
   }
   //--------------------------------------------------------------------------- Depends on previous forall loop in task
   // Calculate change in cell energy, density, position and pressure
   forall cells c in *m{
     double Erate_1[2] = {0., 0.};
     double positionSum_1[2] = {0., 0.};
     double positionDiff_1[2] = {0., 0.};
     double positionDiff_2[2] = {0., 0.};
     forall vertices v in c{
       Erate_1[v.v_index % 2] = (c.c_pressure_old + c.c_q) * -1 * (v.v_pos_old - c.c_pos_old) / fabs(v.v_pos_old - c.c_pos_old)  * v.v_velocity_new; // Sign is right.
       positionSum_1[v.v_index % 2] = v.v_pos_new;
       positionDiff_1[v.v_index % 2] = v.v_pos_new * (v.v_pos_old - c.c_pos_old) / fabs(v.v_pos_old - c.c_pos_old);
       positionDiff_2[v.v_index % 2] = v.v_pos_old * (v.v_pos_old - c.c_pos_old) / fabs(v.v_pos_old - c.c_pos_old);
     }                                                                                                           // Change when implementation available
     double Erate = Erate_1[0] + Erate_1[1];
     double positionSum = positionSum_1[0] + positionSum_1[1];
     double positionDiff1 = positionDiff_1[0] + positionDiff_1[1];
     double positionDiff2 = positionDiff_2[0] + positionDiff_2[1];
	   //printf("updateMesh: The energy_old at cell position %f is %f \n", c.c_pos_new, c.c_energy_old);
     c.c_energy_new = c.c_energy_old + mydt * Erate / c.M;
     c.c_pos_new    = positionSum / 2.;
     c.c_density_new = c.c_density_old * positionDiff2 / positionDiff1;
   }
   //--------------------------------------------------------------------------- Depends on previous forall loop in task
   forall cells c in *m{
     c.c_pressure_new = getPressure(c.c_density_new, c.c_energy_new, gmama);
   }
}

task void remap(sghMesh* m, double T, double gmama, int npts, double length, double pistonspeed, int ncells){
    double length_1 = length - pistonspeed * T;
    double dx_1 = length_1 / (npts - 1);
    forall cells c in *m{
      c.c_pos_new = pistonspeed * T + dx_1/2. + dx_1 * positionx();
    }
    forall vertices v in *m{
      v.v_pos_new = pistonspeed*T + dx_1 * positionx();
    }
    
    forall cells c in *m{
      // Check overlap from neighboring 2 cells
      double tempM = 0;
      double tempP = 0;
      double tempE = 0;
      double tempLength1 = 0; // Right  cell
      double tempLength2 = 0; // Center cell
      double tempLength3 = 0; // Left   cell
      if (positionx() == 0){
        // Right cell
        tempLength1 = cshift(c.x_right, 1) - cshift(c.x_left, 1);
        if (cshift(c.x_left, 1) < c.c_pos_new + dx_1/2.){
          tempM += cshift(c.M, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
          tempP += cshift(c.P, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
          tempE += cshift(c.E, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
        }
        
        // Center cell
        tempLength2 = c.x_right - c.x_left;
        if (c.x_right > c.c_pos_new - dx_1/2. && c.x_left < c.c_pos_new + dx_1/2.){
          tempM += c.M * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempP += c.P * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempE += c.E * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
        }
      }
      // if the last cell to the right
      else if (positionx() == (ncells - 1)){

        // Left cell contribution
        tempLength3 = cshift(c.x_right, -1) - cshift(c.x_left, -1);
        if (cshift(c.x_right, -1) > c.c_pos_new - dx_1/2.){
          tempM += cshift(c.M, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
          tempP += cshift(c.P, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
          tempE += cshift(c.E, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
        } 

        // Center cell contribution
        tempLength2 = c.x_right - c.x_left;
        if (c.x_right > c.c_pos_new - dx_1/2. && c.x_left < c.c_pos_new + dx_1/2.){
          tempM += c.M * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempP += c.P * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempE += c.E * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
        }
      }
      else{
        // Right cell
        tempLength1 = cshift(c.x_right, 1) - cshift(c.x_left, 1);
        if (cshift(c.x_left, 1) < c.c_pos_new + dx_1/2.){
          tempM += cshift(c.M, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
          tempP += cshift(c.P, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
          tempE += cshift(c.E, 1) * (c.c_pos_new + dx_1/2. - cshift(c.x_left, 1)) / tempLength1;
        }
        // Left cell
        tempLength3 = cshift(c.x_right, -1) - cshift(c.x_left, -1);
        if (cshift(c.x_right, -1) > c.c_pos_new - dx_1/2.){
          tempM += cshift(c.M, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
          tempP += cshift(c.P, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
          tempE += cshift(c.E, -1) * (c.c_pos_new - dx_1/2. - cshift(c.x_right, -1)) / tempLength3;
        }
        // Center cell
        tempLength2 = c.x_right - c.x_left;
        if (c.x_right > c.c_pos_new - dx_1/2. && c.x_left < c.c_pos_new + dx_1/2.){
          tempM += c.M * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempP += c.P * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
          tempE += c.E * (fmin(c.x_right, c.c_pos_new + dx_1/2.) - fmax(c.x_left, c.c_pos_new - dx_1/2.)) / tempLength2;
        }
      }
      
      c.c_density_new  = tempM / dx_1;
      c.c_energy_new   = tempE / tempM;
      c.c_pressure_new = getPressure(tempM / dx_1, tempE/tempM, gmama);
      
      c.M_new = tempM;
      c.P_new = tempP;
      c.E_new = tempE;
      
    }
    
    forall vertices v in *m{
      if (positionx() != 0 && positionx() != npts - 1){
        double stagM_1[2] = {0., 0.};
        double stagP_1[2] = {0., 0.};
        forall cells c in v{
          stagM_1[c.c_index % 2] = c.M_new/2.;
          stagP_1[c.c_index % 2] = c.P_new/2.;
        }
        double stagM = stagM_1[0] + stagM_1[1];
        double stagP = stagP_1[0] + stagP_1[1];
        v.v_velocity_new = stagP/stagM;
      }
      
      else if (positionx() == 0)
        v.v_velocity_new = pistonspeed;
      else
        v.v_velocity_new = 0;
    }
}


int main(){
  double dx =((double) length_global)/(npts_global-1);
  sghMesh m[ncells_global];
  setMesh(&m, gmama_global, init_energy_global, init_density_global, dx);
  updateFields(&m);
  double time=0;

  while(time<endtime){
    time+=dt;
    updateMesh(&m, dt/2, gmama_global, q1_global, q2_global, npts_global, pistonspeed_global);
    updatePressure(&m);
    updateMesh(&m, dt, gmama_global, q1_global, q2_global, npts_global, pistonspeed_global);
    updateFields(&m);

    remap(&m, time, gmama_global, npts_global, length_global, pistonspeed_global, ncells_global);
    updateFields(&m);

  }
  float expected[] = 
  {0.550090, 0.057845, 0.650070, 0.057142, 0.750050, 0.056944, 0.850030, 0.056849, 0.950010, 0.056763};

  int i = 0;
  forall cells c in m{

    float tmpfloat;
    char floatval[10];
    sprintf(floatval, "%f", c.c_pos_new);
    sscanf(floatval, "%f", &tmpfloat);
    assert(tmpfloat == expected[i] && "unexpected position value");
    ++i;
    sprintf(floatval, "%f", c.c_energy_new);
    sscanf(floatval, "%f", &tmpfloat);
    assert(tmpfloat ==  expected[i] && "unexpected energy value");
    ++i;
  }

  return 0;
}
