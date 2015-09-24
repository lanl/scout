#ifndef __SGH_SOLVER_2D_SCH__
#define __SGH_SOLVER_2D_SCH__

#include <stdio.h>

//----------------------------------------------------------------------------------
//                           Global Variable Declarations
//----------------------------------------------------------------------------------
// In the case of axial symmetry with regards to I.C., xpts should be odd
// Mesh Parameters
#define xcells 52             // Change with number of points: xpts - 1
#define ycells 52              // Change with number of points: ypts - 1
#define ncells xcells*ycells           // Change with number of points: xcells * ycells
#define xpts   xcells+1             // Change with number of points
#define ypts   ycells+1              // Change with number of points
#define npts   xpts*ypts           // Change with number of points: xpts * ypts


// Physical Properties and Parameters
#define gamma_1       5./3.
//#define endtime       0.002
#define endtime       0.150
#define dt            0.002
#define low_density   0.125
#define high_density  1.000
#define low_pressure  0.100
#define high_pressure 1.000
#define length_x      0.25
#define length_y      0.25
#define dx            0.25/(xcells-2) // Change with number of points: length_x/(xcells-2)
#define dy            0.25/(ycells-2)  // Change with number of points: length_y/(ycells-2)
#define low_energy    0.100 / ((5./3. - 1.) * 0.125) // Change with initial parameters: density, pressure, gamma
#define high_energy   1.000 / ((5./3. - 1.) * 1.000) // Change with initial parameters: density, pressure, gamma
#define q1            1.500
#define q2            1.500

#define stencilSize   9

//----------------------------------------------------------------------------------
//                             Point Structure Declaration
//----------------------------------------------------------------------------------
struct Point{
  double x;
  double y;
}; // end Point structure

//----------------------------------------------------------------------------------
//                             Mesh and Mesh Properties
//----------------------------------------------------------------------------------
uniform mesh sghMesh {
  // Extensive properties have capitals; intensive are all lower case
  cells:
    // Old properties
    double c_x_old;   double c_y_old;     // Cell position field
    double c_p_old;   double c_r_old;     // Cell pressure and density fields
    double c_e_old;   double c_q_old;     // Cell S.I.E. and q fields
    double c_M_old;   double c_E_old;     // Cell mass and energy fields
    double c_A_old;                       // Cell area field
    
    // New properties
    double c_x_new;   double c_y_new;     // Cell position field
    double c_p_new;   double c_r_new;     // Cell pressure and density fields
    double c_e_new;   double c_q_new;     // Cell S.I.E. and q fields
    double c_M_new;   double c_E_new;     // Cell mass and energy fields
    double c_A_new;                       // Cell area field
    
    double c_U;       double c_V;         // Cell momentum, transferred from staggered cells - Remap variables
    int    c_interior;                    // Cell position is interior or exterior
    int    c_index;                       // Cell index, so positionx() and positiony() don't have to be used (as much)
    
  vertices:
    // Old properties
    double v_x_old;   double v_y_old;     // Vertex position field
    double v_u_old;   double v_v_old;     // Vertex velocity fields
    double v_f_x_old; double v_f_y_old;   // Vertex force fields
    double v_U_old;   double v_V_old;     // Staggered momentum fields
    double v_M_old;                       // Staggered mass field
    
    // New properties
    double v_x_new;   double v_y_new;     // Vertex position field
    double v_u_new;   double v_v_new;     // Vertex velocity fields
    double v_f_x_new; double v_f_y_new;   // Vertex force fields
    double v_U_new;   double v_V_new;     // Staggered momentum fields
    double v_M_new;                       // Staggered mass field
    
    double v_A;                           // Vertex area - Remap variable
    int    v_interior;                    // Vertex position is interior or exterior
    int    v_index;                       // Vertex index, so positionx() and positiony() don't have to be used (as much)
}; // end sghMesh declaration


#endif
