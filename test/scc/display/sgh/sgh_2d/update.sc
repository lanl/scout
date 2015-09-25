#include "sgh_solver_2d.sch"

extern struct Point getIntersect(struct Point p1, struct Point p2, struct Point p3, struct Point p4);
extern struct Point getMidpoint(struct Point p1, struct Point p2);
extern double getArea(struct Point p1, struct Point p2, struct Point p3, struct Point p4);




//----------------------------------------------------------------------------------
//                              Update Mesh Methods
//----------------------------------------------------------------------------------
/* updateStaggeredMass()
 * This method is different from the following update methods.
 * Using _new quantities, the staggered mass v_M_new is calculated for each vertex.
 * Assumes c_r_new, c_x_new, c_y_new, v_x_new, and v_y_new are accurate
 */

task void updateStaggeredMass(sghMesh *M)
{
  // Goal: Initialize v_M_new for all vertices
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      // Must check all cases: 4 corners, 4 walls, interior
      int xpos = positionx(); // Global position indices of the vertex
      int ypos = positiony();
      
      struct Point vertex_pos = {v.v_x_new, v.v_y_new}; // Current vertex position
      
      struct Point vertex_south = {cshift(v.v_x_new,  0, -1), cshift(v.v_y_new,  0, -1)}; // Adjacent vertex positions
      struct Point vertex_east  = {cshift(v.v_x_new,  1,  0), cshift(v.v_y_new,  1,  0)};
      struct Point vertex_north = {cshift(v.v_x_new,  0,  1), cshift(v.v_y_new,  0,  1)};
      struct Point vertex_west  = {cshift(v.v_x_new, -1,  0), cshift(v.v_y_new, -1,  0)};
      
      struct Point vertex_midpoint_south = getMidpoint(vertex_pos, vertex_south); // Midpoint positions between vertex and adjacent vertices
      struct Point vertex_midpoint_east  = getMidpoint(vertex_pos, vertex_east );
      struct Point vertex_midpoint_north = getMidpoint(vertex_pos, vertex_north);
      struct Point vertex_midpoint_west  = getMidpoint(vertex_pos, vertex_west );
      
      bool valid_cells[4] = {1, 1, 1, 1}; // Cells that are valid for the considered vertex: indexed by 0-SW, 1-SE, 2-NW, 3-NE
      
      if (     xpos == (0+1)        && ypos == (0+1))       { // SW corner
        valid_cells[2] = 0;        valid_cells[3] = 1;
        valid_cells[0] = 0;        valid_cells[1] = 0;
      }// end if statement for SW corner
      
      else if (xpos == (xpts - 1-1) && ypos == (0+1))       { // SE corner
        valid_cells[2] = 1;        valid_cells[3] = 0;
        valid_cells[0] = 0;        valid_cells[1] = 0;
      }// end elseif statement for SE corner
      
      else if (xpos == (xpts - 1-1) && ypos == (ypts - 1-1)){ // NE corner
        valid_cells[2] = 0;        valid_cells[3] = 0;
        valid_cells[0] = 1;        valid_cells[1] = 0;
      }// end elseif statement for NE corner
      
      else if (xpos == (0+1)        && ypos == (ypts - 1-1)){ // NW corner
        valid_cells[2] = 0;        valid_cells[3] = 0;
        valid_cells[0] = 0;        valid_cells[1] = 1;
      }// end elseif statement for NW corner
      
      else if (ypos == (0+1))                               { // South wall
        valid_cells[2] = 1;        valid_cells[3] = 1;
        valid_cells[0] = 0;        valid_cells[1] = 0;
      }// end elseif statement for South wall
      
      else if (xpos == (xpts - 1-1))                        { // East wall
        valid_cells[2] = 1;        valid_cells[3] = 0;
        valid_cells[0] = 1;        valid_cells[1] = 0;
      }// end elseif statement for East wall
      
      else if (ypos == (ypts - 1-1))                        { // North wall
        valid_cells[2] = 0;        valid_cells[3] = 0;
        valid_cells[0] = 1;        valid_cells[1] = 1;
      }// end elseif statement for North wall
      
      else if (xpos == (0+1))                               { // West wall
        valid_cells[2] = 0;        valid_cells[3] = 1;
        valid_cells[0] = 0;        valid_cells[1] = 1;
      }// end elseif statement for West wall
      
      else                                                  { // Interior point
        valid_cells[2] = 1;        valid_cells[3] = 1;
        valid_cells[0] = 1;        valid_cells[1] = 1;
      }// end else statement for Interior point
      
      double temp_mass[4] = {0., 0., 0., 0.}; // Mass from each surrounding cell
      forall cells c in v
      {
        int local_index = 2 * positiony() + positionx(); // Same indexing as above: 0-SW, 1-SE, 2-NW, 3-NE
        if (valid_cells[local_index])
        {
          struct Point cell_pos = {c.c_x_new, c.c_y_new};
          switch (local_index)
          {
            case 0: // SW cell
              temp_mass[local_index] = c.c_r_new * getArea(vertex_pos, vertex_midpoint_south, cell_pos, vertex_midpoint_west);
              break;
            case 1: // SE cell
              temp_mass[local_index] = c.c_r_new * getArea(vertex_pos, vertex_midpoint_south, cell_pos, vertex_midpoint_east);
              break;
            case 2: // NW cell
              temp_mass[local_index] = c.c_r_new * getArea(vertex_pos, vertex_midpoint_north, cell_pos, vertex_midpoint_west);
              break;
            case 3: // NE cell
              temp_mass[local_index] = c.c_r_new * getArea(vertex_pos, vertex_midpoint_north, cell_pos, vertex_midpoint_east);
              break;
          }// end switch statement to check what cell we're in
        }// end if statement to check for valid cell
      }// end forall cells in vertex loop
      v.v_M_new = temp_mass[0] + temp_mass[1] + temp_mass[2] + temp_mass[3];
    }// end if statement for interior cell
    
    else // Exterior cell
      v.v_M_new = 0.;
  }// end forall vertices loop
}// end updateStaggeredMass() method


// Following update methods initialize old quantities from new quantities

/* updatePressure()
 * Initializes old cell pressure from new cell pressure
 * Properties include:
 *   - Cells:    Pressure
 */
task void updatePressure(sghMesh *M)
{
  forall cells c in *M
  {
    c.c_p_old = c.c_p_new;
  }// end forall cells loop
}// end updatePressure() method

/* updateAll()
 * Initializes old cell and vertex properties from new properties
 * Properties include:
 *   - Cells:    Position, pressure, density, SIE, q, mass, internal energy, area
 *   - Vertices: Position, velocity, force, momentum, mass, area
 */
task void updateAll(sghMesh *M)
{
  forall cells c in *M
  {
    c.c_x_old = c.c_x_new;
    c.c_y_old = c.c_y_new;
    c.c_p_old = c.c_p_new;
    c.c_r_old = c.c_r_new;
    c.c_e_old = c.c_e_new;
    c.c_q_old = c.c_q_new;
    c.c_M_old = c.c_M_new;
    c.c_E_old = c.c_E_new;
    c.c_A_old = c.c_A_new;
  }// end forall cells loop
  
  forall vertices v in *M
  {
    v.v_x_old   = v.v_x_new;
    v.v_y_old   = v.v_y_new;
    v.v_u_old   = v.v_u_new;
    v.v_v_old   = v.v_v_new;
    v.v_f_x_old = v.v_f_x_new;
    v.v_f_y_old = v.v_f_y_new;
    v.v_U_old   = v.v_U_new;
    v.v_V_old   = v.v_V_new;
    v.v_M_old   = v.v_M_new;
  }// end forall vertices loop
}// end updateAll() method
