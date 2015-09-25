#include "sgh_solver_2d.sch"

extern task void updateStaggeredMass(sghMesh *M);
extern task void updateAll(sghMesh *M);

// 2 Methods: initializeExtensiveCells() and initializeMesh()

//----------------------------------------------------------------------------------
//                              Initialize Mesh Methods
//----------------------------------------------------------------------------------


/* initializeExtensiveCells()
 * Initializes the extensive properties in all cells.
 * Properties include total mass and total internal energy, old and new.
 * Separate subtask because properties called are initialized in the task initializeMesh().
 */
task void initializeExtensiveCells(sghMesh *M)
{
  forall cells c in *M
  {
    c.c_M_old = c.c_r_old * c.c_A_old;             c.c_M_new = c.c_r_new * c.c_A_new;
    c.c_E_old = c.c_e_old * c.c_A_old * c.c_r_old; c.c_E_new = c.c_e_new * c.c_A_new * c.c_r_new;
  }// end cells loop over mesh M to initialize extensive properties
}// end initializeExtensiveCells() method



/* initializeMesh()
 * Initializes all properties in all cells and vertices. Should be called once at the beginning of main() method.
 * Properties include:
 *   - Cells:    Position, pressure, density, SIE, q, area, index, mass, internal energy, interior
 *   - Vertices: Position, velocity, force, momentum, index, staggered mass, interior
 * Calls two subtasks for extensive properties: one for cells, one for vertices.
 */
task void initializeMesh(sghMesh *M)
{
  forall cells c in *M
  {
    c.c_x_old = (dx * positionx() + dx/2.) - dx;  c.c_x_new = (dx * positionx() + dx/2.) - dx;
    c.c_y_old = (dy * positiony() + dy/2.) - dy;  c.c_y_new = (dy * positiony() + dy/2.) - dy;
    
                                                                             // Criteria for initial condition
    if (positionx() < xcells / 2)
    {
      // On the left: has higher properties
      c.c_p_old = high_pressure;               c.c_p_new = high_pressure;
      c.c_r_old = high_density;                c.c_r_new = high_density;
      c.c_e_old = high_energy;                 c.c_e_new = high_energy;
    }// end if statement for initial condition
    else
    {
      // On the right: has lower properties
      c.c_p_old = low_pressure;                c.c_p_new = low_pressure;
      c.c_r_old = low_density;                 c.c_r_new = low_density;
      c.c_e_old = low_energy;                  c.c_e_new = low_energy;
    }// end else statement for initial condition
    c.c_q_old = 0.;                            c.c_q_new = 0.;
    c.c_A_old = dx * dy;                       c.c_A_new = dx * dy; // Not intensive, but used extensively in extensive properties loop
    
    c.c_index = positionx() + xcells * positiony();
    
    // Check for exterior cells. At this point, exterior cells are simply placeholders due to the malfunctioning of cshift()
    if ((positionx() == 0) || (positionx() == (xcells - 1)) || (positiony() == 0) || (positiony() == (ycells - 1)))
      c.c_interior = 0;
    else
      c.c_interior = 1;
  }// end cells loop over mesh M to initialize intensive properties
  
  initializeExtensiveCells(M);
  
  forall vertices v in *M
  {
    v.v_x_old = dx * positionx() - dx;         v.v_x_new = dx * positionx() - dx;
    v.v_y_old = dy * positiony() - dy;         v.v_y_new = dy * positiony() - dy;
    v.v_u_old = 0.;                            v.v_u_new = 0.;
    v.v_v_old = 0.;                            v.v_v_new = 0.;
    v.v_f_x_old = 0.;                          v.v_f_x_new = 0.;
    v.v_f_y_old = 0.;                          v.v_f_y_new = 0.;
    v.v_U_old = 0.;                            v.v_U_new = 0.;
    v.v_V_old = 0.;                            v.v_V_new = 0.;
    v.v_index = positionx() + xpts * positiony();
    
    // Check for exterior vertices. At this point, exterior vertices are simply placeholders due to the malfunctioning of cshift()
    if ((positionx() == 0) || (positionx() == (xpts - 1)) || (positiony() == 0) || (positiony() == (ypts - 1)))
      v.v_interior = 0;
    else
      v.v_interior = 1;
  }// end vertices loop over mesh M
  
  updateStaggeredMass(M); // Initializes the one extensive vertex property: staggered mass
  updateAll(M);
}// end initializeMesh() method
