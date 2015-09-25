#include "math.h"
#include "sgh_solver_2d.sch"
#include "stdio.h"

extern task void updatePressure(sghMesh *M);
extern task void updateAll(sghMesh *M);
extern task void updateStaggeredMass(sghMesh *M);


/* getIntersect()
 * Helper function for sgh_force(). Returns the point of intersection of four points which define the endpoints of two line segments.
 * The points p1 and p2 define one line segment. Points p3 and p4 define the second line segment.
 * Prints error statement for parallel lines. Deals with vertical lines explicitly; deals with horizontal implicitly.
 ********* METHOD NEEDS TO BE VERIFIED
 */
struct Point getIntersect(struct Point p1, struct Point p2, struct Point p3, struct Point p4)
{
  double m1; // Slope of line 1
  double m2; // Slope of line 2
  double b1; // y-intercept of line 1
  double b2; // y-intercept of line 2
  struct Point result = {0., 0.};
  if (p1.x == p2.x && p3.x == p4.x)
  {
    result.x = -999;
    result.y = -999;
    //printf("ERROR in get_intersect(): parallel lines constructed\n");
    return result;
  }// end if statement to check for same x locations in both pairs of points
  
  else if (p1.x == p2.x) // Vertical line for first line segment
  {
    m2 = (p4.y - p3.y) / (p4.x - p3.x); // Slope of second line segment
    b2 = p3.y - m2 * p3.x;              // y-intercept of second line segment
    result.x = p1.x;
    result.y = m2 * p1.x + b2;
    return result;
  }// end if statement for vertical line for first line segment
  
  else if (p3.x == p4.x) // Vertical line for second line segment
  {
    m1 = (p2.y - p1.y) / (p2.x - p1.x); // Slope of first  line segment
    b1 = p1.y - m1 * p1.x;              // y-intercept of first  line segment
    result.x = p3.x;
    result.y = m1 * p3.x + b1;
    return result;
  }// end else if statement for vertical line for second line segment
  
  // General case: non-vertical lines
  m1 = (p2.y - p1.y) / (p2.x - p1.x); // Slope of first  line segment
  m2 = (p4.y - p3.y) / (p4.x - p3.x); // Slope of second line segment
  
  b1 = p1.y - m1 * p1.x;              // y-intercept of first  line segment
  b2 = p3.y - m2 * p3.x;              // y-intercept of second line segment
  if (m1 == m2)
  {
    result.x = -999;
    result.y = -999;
    //printf("ERROR in get_intersect(): parallel lines constructed\n");
    return result;
  }// end if statement for parallel lines
  result.x = -1 * (b1 - b2) / (m1 - m2);
  result.y = m1 * result.x + b1;
  return result;
}// end getIntersect() method









/* getArea()
 * Helper function for a number of methods. Returns the area of the quadrilateral defined by the four vertices p1, p2, p3 and p4.
 * The points must be passed in a circular or anti-circular fashion. In other words, p1 and p3 define one diagonal and p2 and p4
 * define the other diagonal. THE METHOD WILL NOT WORK IF POINTS ARE PASSED OUT OF ORDER.
 * Also assumes a convex quadrilateral.
 * The area is calculated by taking half of the cross product of the two diagonals.
 * Reference: http://mathworld.wolfram.com/Quadrilateral.html
 ********* METHOD NEEDS TO BE VERIFIED
 */
double getArea(struct Point p1, struct Point p2, struct Point p3, struct Point p4)
{
  // Will not use points in this method; these are vectors, avoid confusion
  // Diagonal 1 is p1 -> p3
  // Vector 1: <x1, y1>
  double x1 = p3.x - p1.x;
  double y1 = p3.y - p1.y;
  // Diagonal 2 is p2 -> p4
  // Vector 2: <x2, y2>
  double x2 = p4.x - p2.x;
  double y2 = p4.y - p2.y;
  
  double area = 1./2. * fabs(x1 * y2 - x2 * y1);
  return area;
}// end getArea() method









/* getCentroid()
 * Helper function for sgh_c_area(). Returns the centroid of the quadrilateral with uniform density
 * defined by the four vertices p1, p2, p3 and p4.
 * The points must be passed in a circular or anti-circular fashion. In other words, p1 and p3 define one diagonal and p2 and p4
 * define the other diagonal. THE METHOD WILL NOT WORK IF POINTS ARE PASSED OUT OF ORDER.
 * Also assumes a convex quadrilateral.
 * The centroid is found by finding the intersection of the bimedians.
 * Reference: http://mathworld.wolfram.com/Quadrilateral.html
 ********* METHOD NEEDS TO BE VERIFIED
 */
struct Point getCentroid(struct Point p1, struct Point p2, struct Point p3, struct Point p4)
{
  // Must find the points of the four medians.
  struct Point m1 = {0., 0.}; // Median point between p1 and p2
  struct Point m2 = {0., 0.}; // Median point between p2 and p3
  struct Point m3 = {0., 0.}; // Median point between p3 and p4
  struct Point m4 = {0., 0.}; // Median point between p4 and p1
  
  m1.x = (p1.x + p2.x) / 2.;
  m1.y = (p1.y + p2.y) / 2.;
  
  m2.x = (p2.x + p3.x) / 2.;
  m2.y = (p2.y + p3.y) / 2.;
  
  m3.x = (p3.x + p4.x) / 2.;
  m3.y = (p3.y + p4.y) / 2.;
  
  m4.x = (p4.x + p1.x) / 2.;
  m4.y = (p4.y + p1.y) / 2.;
  
  // One bimedian is defined by m1, m3; the other by m2, m4
  // Must find intersection of two line segments
  return getIntersect(m1, m3, m2, m4);
}// end getCentroid() method









/* getMidpoint()
 * Helper function for staggered cells. Returns the midpoint between two points p1 and p2.
 */
struct Point getMidpoint(struct Point p1, struct Point p2)
{
  struct Point result;
  result.x = (p1.x + p2.x) / 2.;
  result.y = (p1.y + p2.y) / 2.;
  return result;
}// end getMidpoint() method









//----------------------------------------------------------------------------------
//                              Lagrange Step Methods
//----------------------------------------------------------------------------------




/* sgh_q()
 * Subtask to calculate cell quantity q from surrounding vertices
 ******* For this particular implementation, we assume shock goes from left to right. Only in x-direction.
 ******* Will take the average difference of velocities for calculating q. Scott's C++ code does lower right minus lower left.
 */
task void sgh_q(sghMesh *M)
{
  forall cells c in *M {
    //if (c.c_interior) {
    forall vertices v in c {
      //printf("1:positionx(): %d\n", positionx());
    }
  //}
  }

  // Goal: Initialize c_q_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      double du_vec[4] = {0., 0., 0., 0.};
      forall vertices v in c
      {
        // forall to calculate difference of x-component of velocity, v.v_u. East - West, then average.
        //printf("2:positionx(): %d\n", positionx());
        if (positionx() == 1) // Vertex to east
          du_vec[2 * positiony() + positionx()] =  1 * v.v_u_old;
        else                  // Vertex to west
          du_vec[2 * positiony() + positionx()] = -1 * v.v_u_old;
      }// end forall vertices in cell loop
      
      double du = (du_vec[0] + du_vec[1] + du_vec[2] + du_vec[3]) / 2.; // Average compression rate in x-direction
      double ss = gamma_1 * (gamma_1 - 1.) * c.c_e_old; // Sound speed; need to take sqrt
      if (ss > 0) ss = sqrt(ss); else ss = 0.000001; // Defensive programming; all factors above should be positive, but just in case, double check
      
      double q = 0.;
      q += q1 * c.c_r_old * fabs(du) * ss;     // Linear    term
      q += q2 * c.c_r_old *    du    * du;     // Quadratic term
      if (du > 0.) q = 0.;
      
      c.c_q_new = q; // end goal - update c_q_new for all cells
    }// end if statement for interior cell
  }// end forall cells loop
}// end sgh_q() method









/* sgh_force()
 * Updates x- and y- components of forces on vertices.
 * This task depends on previous task, sgh_q, since c_q_new is used.
 ******* The dividing lines between staggered cells are the line segments connecting cell centroids, c_x and c_y, to edge midpoints.
 */
task void sgh_force(sghMesh *M)
{
  // Goal: Initialize v_f_x_new and v_f_y_new for all vertices
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      // Must check all cases: 4 corners, 4 walls, interior
      int xpos = positionx();
      int ypos = positiony();
      
      struct Point vertex_pos = {v.v_x_old, v.v_y_old}; // Current vertex position
      
      struct Point vertex_south = {cshift(v_x_old,  0, -1), cshift(v_y_old,  0, -1)}; // Adjacent vertex positions
      struct Point vertex_east  = {cshift(v_x_old,  1,  0), cshift(v_y_old,  1,  0)};
      struct Point vertex_north = {cshift(v_x_old,  0,  1), cshift(v_y_old,  0,  1)};
      struct Point vertex_west  = {cshift(v_x_old, -1,  0), cshift(v_y_old, -1,  0)};
      
      
      struct Point vertex_midpoint_south = getMidpoint(vertex_pos, vertex_south); // Midpoint positions between vertex and adjacent vertices
      struct Point vertex_midpoint_east  = getMidpoint(vertex_pos, vertex_east );
      struct Point vertex_midpoint_north = getMidpoint(vertex_pos, vertex_north);
      struct Point vertex_midpoint_west  = getMidpoint(vertex_pos, vertex_west );
      
      
      
      // B.C.'s: Edges and corners are fixed in position.
      // Accomplished by setting forces equal to 0. Only loss of information is stress on boundaries.
      // However, focus is on what's going on inside; don't need to know stress on boundaries.
      // Boundary conditions applied at end of method.
      
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
      
      
      double temp_x_force[4] = {0., 0., 0., 0.}; // x-component of force from each surrounding cell
      double temp_y_force[4] = {0., 0., 0., 0.}; // y-component of force from each surrounding cell
      forall cells c in v
      {
        int local_index = 2 * positiony() + positionx(); // Same indexing as above: 0-SW, 1-SE, 2-NW, 3-NE
        if (valid_cells[local_index])
        {
          double force_x1 = 0.;         double force_x2 = 0.; // x-forces from vertical and horizontal line segments
          double force_y1 = 0.;         double force_y2 = 0.; // y-forces from vertical and horizontal line segments
          struct Point cell_pos = {c.c_x_old, c.c_y_old};
          
          switch (local_index)
          {
            case 0: // SW cell
              force_x1 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_south.y);
              force_x2 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_west.y );
              force_y1 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_south.x);
              force_y2 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_west.x );
              
              temp_x_force[local_index] = force_x1 + force_x2;
              temp_y_force[local_index] = force_y1 + force_y2;
              break;
            case 1: // SE cell
              force_x1 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_south.y);
              force_x2 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_east.y );
              force_y1 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_south.x);
              force_y2 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_east.x );
              
              temp_x_force[local_index] = force_x1 + force_x2;
              temp_y_force[local_index] = force_y1 + force_y2;
              break;
            case 2: // NW cell
              force_x1 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_north.y);
              force_x2 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_west.y );
              force_y1 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_north.x);
              force_y2 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_west.x );
              
              temp_x_force[local_index] = force_x1 + force_x2;
              temp_y_force[local_index] = force_y1 + force_y2;
              break;
            case 3: // NE cell
              force_x1 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_north.y);
              force_x2 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.y - vertex_midpoint_east.y );
              force_y1 = -1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_north.x);
              force_y2 =  1 * (c.c_p_old + c.c_q_new) * (cell_pos.x - vertex_midpoint_east.x );
              
              temp_x_force[local_index] = force_x1 + force_x2;
              temp_y_force[local_index] = force_y1 + force_y2;
              break;
          }// end switch statement to check what cell we're in
        }// end if statement to check for valid cell
      }// end forall cells in vertex loop
      
      v.v_f_x_new = temp_x_force[0] + temp_x_force[1] + temp_x_force[2] + temp_x_force[3];
      v.v_f_y_new = temp_y_force[0] + temp_y_force[1] + temp_y_force[2] + temp_y_force[3];
      
      // Boundary conditions
      if (xpos == (0+1) || xpos == (xpts - 1-1))
        v.v_f_x_new = 0.;
      if (ypos == (0+1) || ypos == (ypts - 1-1))
        v.v_f_y_new = 0.;
      
    }// end if statement for interior vertices
  }// end forall vertices loop
}// end sgh_force() method









/* sgh_velocity()
 * Updates the two components of velocity for every vertex on the mesh.
 * Uses v_f_x_new and v_f_y_new to update the velocity components.
 * ******** This method assumes the staggered masses v.v_M_old on the vertices is correct
 */
task void sgh_velocity(sghMesh *M, double my_dt)
{
// Goal: Initialize v_u_new and v_v_new for all vertices
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      v.v_u_new = v.v_u_old + (v.v_f_x_new / v.v_M_old) * my_dt;
      v.v_v_new = v.v_v_old + (v.v_f_y_new / v.v_M_old) * my_dt;
      
    }// end if statement for interior vertex
  }// end forall vertices loop
}// end sgh_velocity() method









/* sgh_v_pos()
 * Updates the two components of position for every vertex on the mesh.
 * Uses v_u_new and v_v_new to update the position components.
 */
task void sgh_v_pos(sghMesh *M, double my_dt)
{
// Goal: Initialize v_x_new and v_y_new for all vertices
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      v.v_x_new = v.v_x_old + my_dt * v.v_u_new;
      v.v_y_new = v.v_y_old + my_dt * v.v_v_new;
    }// end if statement for interior vertex
  }// end forall vertices loop
}// end sgh_v_pos() method









/* sgh_c_pos()
 * Updates the two components of position for every cell on the mesh.
 * Uses v_x_new and v_y_new to update the position components.
 */
task void sgh_c_pos(sghMesh *M)
{
// Goal: Initialize c_x_new and c_y_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      // Need to find the centroid of the quadrilateral defined by the surrounding vertices
      struct Point sur_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}; // Positions of surrounding vertices
      forall vertices v in c
      {
        sur_vertices[2 * positiony() + positionx()].x = v.v_x_new;
        sur_vertices[2 * positiony() + positionx()].y = v.v_y_new;
      }// end forall vertices in cell loop
      
      /* Explanation:
       * So, the method for finding centroids only works when points are passed in a circular or anti-circular fashion.
       * The way the indexing works above, passing the points with indices 0, 1, 2, 3 is ALWAYS out of order - it goes
       * across a diagonal. However, the order 0, 1, 3, 2 is ALWAYS in order! Same as 1, 3, 2, 0;     3, 2, 0, 1;   etc.
       * So, to find the centroid of the quadrilateral, the points will be passed [0, 1, 3, 2]. If in doubt, work it out
       * for oneself. Passed [SW, SE, NE, NW].
       */
      struct Point temp_pos = getCentroid(sur_vertices[0], sur_vertices[1], sur_vertices[3], sur_vertices[2]);
      c_x_new = temp_pos.x;
      c_y_new = temp_pos.y;
    }// end if statement for interior cells
  }// end forall cells loop
}// end sgh_c_pos() method









/* sgh_c_area()
 * Updates the area associated with each cell.
 * Uses the surrounding vertex positions.
 */
task void sgh_c_area(sghMesh *M)
{
// Goal: Initialize c_A_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      struct Point sur_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
      forall vertices v in c
      {
        sur_vertices[2 * positiony() + positionx()].x = v.v_x_new;
        sur_vertices[2 * positiony() + positionx()].y = v.v_y_new;
      }// end forall vertices in cell loop
      // If looking for explanation on passing order, check explanation in sgh_c_pos() method
      c.c_A_new = getArea(sur_vertices[0], sur_vertices[1], sur_vertices[3], sur_vertices[2]);
    }// end if statement for interior cell
  }// end forall cells loop
}// end sgh_c_area() method









/* sgh_density()
 * Updates the density associated with each cell.
 * Uses the cell area as well as old quantities
 */
task void sgh_density(sghMesh *M)
{
// Goal: Initialize c_r_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
      c.c_r_new = c.c_r_old * c.c_A_old / c.c_A_new;
  }// end forall cells loop
}// end sgh_density() method









/* sgh_energy()
 * Updates the specific internal energy associated with each cell.
 * Uses the vertex velocities as well as vertex forces.
 */
task void sgh_energy(sghMesh *M, double my_dt)
{
// Goal: Initialize c_e_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      double E_rate_vec[4] = {0., 0., 0., 0.};
      forall vertices v in c
      {
        int sign_x = 1;
        int sign_y = 1;
        if (positionx() == 1)
          sign_x = -1;
        if (positiony() == 1)
          sign_y = -1;
        E_rate_vec[2 * positiony() + positionx()] = sign_x * v.v_f_x_new * v.v_u_new + sign_y * v.v_f_y_new * v.v_v_new; // E_rate is equal to F dot v
      }// end forall vertices in cell loop
      double E_rate = E_rate_vec[0] + E_rate_vec[1] + E_rate_vec[2] + E_rate_vec[3];
      c.c_e_new = c.c_e_old + E_rate * my_dt / c.c_M_old; // c.c_M_new hasn't been initialized, but all cells maintain the same mass in a lagrange step, so this works
      
    }// end if statement for interior cell
  }// end forall cells loop
}// end sgh_energy() method









/* sgh_pressure()
 * Updates the pressure associated with each cell.
 * Uses the cell S.I.E. as well as cell density. EOS in this method
 */
task void sgh_pressure(sghMesh *M)
{
// Goal: Initialize c_p_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      c.c_p_new = (gamma_1 - 1) * c.c_e_new * c.c_r_new;
    }
  }// end forall cells loop
}// end sgh_pressure() method









/* sgh_extensive()
 * Updates the extensive quantities associated with each cell and vertex.
 * Uses several cell and vertex quantities.
 * Updates c_M_new, c_E_new, v_M_new, v_U_new, v_V_new
 */
task void sgh_extensive(sghMesh *M)
{
// Goal: Initialize c_M_new, c_E_new, v_M_new, v_U_new, v_V_new for all cells and vertices
  forall cells c in *M
  {
    if (c.c_interior)
    {
      c.c_M_new = c.c_r_new * c.c_A_new;
      
      // c.c_E_new = c.c_r_new * c.c_A_new * c.c_e_new; // for E = Internal Energy
      
      double avg_speed_squared_vec[4] = {0., 0., 0., 0.};
      forall vertices v in c
      {
        avg_speed_squared_vec[2 * positiony() + positionx()] = (v.v_u_new * v.v_u_new + v.v_v_new * v.v_v_new) / 4.;
      }// end forall vertices in cell loop
      double avg_speed_squared = avg_speed_squared_vec[0] + avg_speed_squared_vec[1] + avg_speed_squared_vec[2] + avg_speed_squared_vec[3];
      
      c.c_E_new = c.c_r_new * c.c_A_new * c.c_e_new;//    +    c.c_r_new * c.c_A_new * avg_speed_squared / 2.; // Internal + Kinetic = Total; for E = Total Energy
    }// end if statement for interior cell
  }// end forall cells loop
  
  // Update v_M_new
  updateStaggeredMass(M);
  
  // Calculate staggered momentum; update v_U_new, v_V_new
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      v.v_U_new = v.v_M_new * v.v_u_new;
      v.v_V_new = v.v_M_new * v.v_v_new;
    }// end if statement for interior vertex
  }// end forall vertices loop
  
}// end sgh_extensive() method









/* sghAdvance()
 * Single advance step. New properties are stored to _new variables, reassigned in different method.
 * Subtasks, in order:
 *   - Calculate q for cells                                   - sgh_q()
 *   - Calculate forces for vertices                           - sgh_force()
 *   - Calculate velocities for vertices                       - sgh_velocity()
 *   - Calculate positions for vertices                        - sgh_v_pos()
 *   - Calculate positions for cells                           - sgh_c_pos()
 *   - Calculate areas for cells                               - sgh_c_area()
 *   - Calculate densities for cells                           - sgh_density()
 *   - Calculate energies for cells                            - sgh_energy()
 *   - Calculate pressures for cells                           - sgh_pressure()
 *   - Calculate extensive properties for cells and vertices   - sgh_extensive()
 */
task void sghAdvance(sghMesh *M, double my_dt)
{
  sgh_q(M);               // c.c_q_new initialized
  sgh_force(M);           // v.v_f_x_new, v.v_f_y_new initialized
  sgh_velocity(M, my_dt); // v.v_u_new, v.v_v_new initialized
  sgh_v_pos(M, my_dt);    // v.v_x_new, v.v_y_new initialized
  sgh_c_pos(M);           // c.c_x_new, c.c_y_new initialized
  sgh_c_area(M);          // c.c_A_new initialized
  sgh_density(M);         // c.c_r_new initialized
  sgh_energy(M, my_dt);   // c.c_e_new initialized
  sgh_pressure(M);        // c.c_p_new initialized
  sgh_extensive(M);       // c.c_M_new, c.c_E_new, v.v_M_new, v.v_U_new, v.v_V_new initialized
}// end sghAdvance() method









/* LagrangeStep()
 * Performs the entirety of a single lagrange step without remap
 * Carries out a half-step, updates pressure, carries out a full-step, updates all properties
 * Mostly just separates tasks and calls methods
 */
void LagrangeStep(sghMesh *M, int count, double Time)
{
  sghAdvance(M, dt/2.);
  updatePressure(M);
  sghAdvance(M, dt);
  updateAll(M);
}// end LagrangeStep() method
