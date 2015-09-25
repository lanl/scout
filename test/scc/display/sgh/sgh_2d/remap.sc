#include "sgh_solver_2d.sch"

extern struct Point getIntersect(struct Point p1, struct Point p2, struct Point p3, struct Point p4);
extern struct Point getMidpoint(struct Point p1, struct Point p2);
extern double getArea(struct Point p1, struct Point p2, struct Point p3, struct Point p4);
extern task void sgh_c_area(sghMesh *M);
extern task void updateStaggeredMass(sghMesh *M);
extern task void sgh_pressure(sghMesh *M);
extern task void updateAll(sghMesh *M);


/* remap_pos()
 * Initialize c_x_new, c_y_new, c_A_new v_x_new, v_y_new.
 * Assumes boundaries are fixed, and creates a regularly spaced mesh.
 */
task void remap_pos(sghMesh *M)
{
// Goal: Initialize c_x_new, c_y_new, v_x_new, v_y_new for all cells and vertices
  forall cells c in *M
  {
    if (c.c_interior)
    {
      c.c_x_new = (dx * positionx() + dx/2.) - dx;
      c.c_y_new = (dy * positiony() + dy/2.) - dy;
      c.c_A_new = dx * dy;
    }// end if statement for checking for interior cell
  }// end forall cells loop
  
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      v.v_x_new = (dx * positionx()) - dx;
      v.v_y_new = (dy * positiony()) - dy;
    }// end if statement for checking for interior vertex
  }// end forall vertices loop
}// end remap_pos() method









/* Calculates the area of overlap from the stencil of 9 deformed cells about the regular cell, represented by c.
 * Intended for quadrilateral cells.
 * Designed to map from the 9 cells around this cell will be considered, enumerated as follows:
 * 
 *    -------------------------------------------------------------
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |         6         |         7         |         8         |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |-------------------|-------------------|-------------------|
 *    |                   |                   |                   |
 *    |                   |      ccccccc      |                   |
 *    |                   |      c     c      |                   |
 *    |         3         |      c  4  c      |         5         |
 *    |                   |      c     c      |                   |
 *    |                   |      ccccccc      |                   |
 *    |                   |                   |                   |
 *    |-------------------|-------------------|-------------------|
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |         0         |         1         |         2         |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    |                   |                   |                   |
 *    -------------------------------------------------------------
 *
 * Cell number 4 is the cell mapping extensive quantities to; it is the parameter c.
 * 
 * The variable 'areas' is the result of the clipping. It is an array of 9 doubles representing the 9 intersecting areas.
 * 
 * The variable 'validCells' is an array of 9 booleans representing the validity of the surrounding cells.
 * For example, the corner cells have 4 valid cells to map to the new cell. The wall cells have 6 valid cells,
 * and the interior cells have 9 valid cells. The need for 'validCells' is due to the circular nature of the mesh.
 */




/* remap_mass()
 * Initialize c_M_new, c_r_new.
 * Take old mesh with old properties and map extensive mass property (c_M_old)
 * to new mesh (c_M_new). From calculation of c_M_new, update c_r_new.
 * Also calculates c_E_new since the clipping is already done in this method.
 * Might as well use areas[] to calculate c_E_new. c_e_new is calculated in the remap_energy() method.
 */
task void remap_mass(sghMesh *M)
{
// Goal: Initialize c_M_new, c_r_new for all cells

  // Mass calculation
  forall cells c in *M
  {
    if (c.c_interior)
    {
      bool theSame = true;
      forall vertices v in c
        if (v.v_x_new != v.v_x_old || v.v_y_new != v.v_y_old) theSame = false;
      
      if (theSame)
      {
        c.c_M_new = c.c_M_old;
        c.c_E_new = c.c_E_old;
      }// end if statement for simple case of no movement
      
      else // Non-trivial case, vertex movement
      {
        // Must clip() from each surrounding cell
        // For corner,   that's 4 cells
        // For edge,     that's 6 cells
        // For interior, that's 9 cells
        double areas[stencilSize] = {0., 0., 0.,    0., 0., 0.,    0., 0., 0.};
        int xpos = positionx();
        int ypos = positiony();
        
        struct Point reg_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}; // The vertices around the regular cell (index 4)
        forall vertices v in c
        {
          reg_vertices[2 * positiony() + positionx()].x = v.v_x_new; // _new for regular positions
          reg_vertices[2 * positiony() + positionx()].y = v.v_y_new; // _new for regular positions
        }// end forall vertices around regular cell
        
        // Every time this indexing is performed, the order is as follows: 0-SW, 1-SE, 2-NW, 3-NE
        // Every time, the indexing will be reassigned so it's CCW:        0-SW, 1-SE, 2-NE, 3-NW
        
        struct Point temp_point = {reg_vertices[2].x, reg_vertices[2].y};
        reg_vertices[2].x = reg_vertices[3].x;          reg_vertices[2].y = reg_vertices[3].y;
        reg_vertices[3].x = temp_point.x;               reg_vertices[3].y = temp_point.y;
        
        bool validCells[stencilSize] = {1, 1, 1,  1, 1, 1,  1, 1, 1}; // Which surrounding cells to check
        
        if      (xpos == (0+1)         && ypos == (0+1)){           // SW corner
          validCells[6] = 0;                                      
          validCells[3] = 0;                                      
          validCells[0] = 0; validCells[1] = 0; validCells[2] = 0;
        }
        else if (xpos == (xcells - 1-1) && ypos == (0+1)){          // SE corner
                                                validCells[8] = 0;
                                                validCells[5] = 0;
          validCells[0] = 0; validCells[1] = 0; validCells[2] = 0;
        }
        else if (xpos == (xcells - 1-1) && ypos == (ycells - 1-1)){ // NE corner
          validCells[6] = 0; validCells[7] = 0; validCells[8] = 0;
                                                validCells[5] = 0;
                                                validCells[2] = 0;
        }
        else if (xpos == (0+1)          && ypos == (ycells - 1-1)){ // NW corner
          validCells[6] = 0; validCells[7] = 0; validCells[8] = 0;
          validCells[3] = 0;                                      
          validCells[0] = 0;                                      
        }
        else if (ypos == (0+1)){                                    // South wall
                                                                  
                                                                  
          validCells[0] = 0; validCells[1] = 0; validCells[2] = 0;
        }
        else if (xpos == (xcells - 1-1)){                           // East  wall
                                                validCells[8] = 0;
                                                validCells[5] = 0;
                                                validCells[2] = 0;
        }
        else if (ypos == (ycells - 1-1)){                           // North wall
          validCells[6] = 0; validCells[7] = 0; validCells[8] = 0;
                                                                  
                                                                  
        }
        else if (xpos == (0+1)){                                    // West  wall
          validCells[6] = 0;                                      
          validCells[3] = 0;                                      
          validCells[0] = 0;                                      
        }
        else{                                                       // Interior
          
        }
        
        
        
        
        // This could be done using a 2nd mesh, but will be implemented using arrays.
        for (int i = 0; i < stencilSize; i++)      // for loop, to loop through the 9 cells in stencil
        {
          if (validCells[i])
          {
            int xdir = 0;
            int ydir = 0;
            
            // Switch statement to determine the direction to go from the cell to the considered deformed cell
            switch (i)
            {
              case 0: xdir = -1; ydir = -1; break;
              case 1: xdir =  0; ydir = -1; break;
              case 2: xdir =  1; ydir = -1; break;
              case 3: xdir = -1; ydir =  0; break;
              case 4: xdir =  0; ydir =  0; break;
              case 5: xdir =  1; ydir =  0; break;
              case 6: xdir = -1; ydir =  1; break;
              case 7: xdir =  0; ydir =  1; break;
              case 8: xdir =  1; ydir =  1; break;
            }// end switch statement for initializing the directions for the cell considered
            
            struct Point def_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}; // Points for deformed cell
            
            forall vertices v in c
            {
              def_vertices[2 * positiony() + positionx()].x = cshift(v_x_old, xdir, ydir); // Old for deformed positions
              def_vertices[2 * positiony() + positionx()].y = cshift(v_y_old, xdir, ydir); // Old for deformed positions
            }// end forall vertices in deformed cell
            
            // Change indexing so points are in correct order
            
            temp_point.x = def_vertices[2].x;           temp_point.y = def_vertices[2].y;
            def_vertices[2].x = def_vertices[3].x;      def_vertices[2].y = def_vertices[3].y;
            def_vertices[3].x = temp_point.x;           def_vertices[3].y = temp_point.y;
            
            // Now we have the regular vertices in reg_vertices, and the deformed vertices in def_vertices.
            // Must find the area overlap between these two quadrilaterals.
            
            struct Point verts_old[21]; // Vertices of the intersecting polygon - old
            struct Point verts_new[21]; // Vertices of the intersecting polygon - new
            
            for (int j = 0; j < 21; j++) // Initialize the arrays of polygon vertices
            {
              verts_old[j].x = -999;   verts_old[j].y = -999;
              verts_new[j].x = -999;   verts_new[j].y = -999;
            }// end for loop to initialize vertex locations
            
            int vertex_counter = 4;
            
            for (int j = 0; j < vertex_counter; j++)
            {
              verts_old[j].x = def_vertices[j].x;  verts_old[j].y = def_vertices[j].y;
              verts_new[j].x = def_vertices[j].x;  verts_new[j].y = def_vertices[j].y;
            }// end for loop to initialize the first four points of the polygon vertices
            
            
            
            for (int j = 0; j < 4; j++) // Iterate through windows; 4 for the 4 windows
            {
              // Reminder: {0, 1, 2, 3} = {SW, SE, NE, NW}
              // For j = 0 & 2, the windows are horizontal; for j = 1 & 3, the windows are vertical.
              // So, for checking j = 0, just compare y values. For checking j = 1, compare x values.
              
              bool inside[vertex_counter]; // Boolean array for vertices inside window
              
              for (int k = 0; k < vertex_counter; k++) // Initialize inside array to all false
                inside[k] = false;
              
              for (int k = 0; k < vertex_counter; k++) // Iterate through the deformed vertices to determine inside or outside.
              {
                switch(j) // Switch statement to determine what window on the regular cell we're on
                {
                  case 0: // Horizontal window on south part of cell: Inside is to the north
                    if (verts_old[k].y > reg_vertices[j].y)
                      inside[k] = true; // False otherwise
                    break;
                  case 1: // Vertical   window on east  part of cell: Inside is to the west
                    if (verts_old[k].x < reg_vertices[j].x)
                      inside[k] = true; // False otherwise
                    break;
                  case 2: // Horizontal window on north part of cell: Inside is to the south
                    if (verts_old[k].y < reg_vertices[j].y)
                      inside[k] = true; // False otherwise
                    break;
                  case 3: // Vertical   window on west  part of cell: Inside is to the east
                    if (verts_old[k].x > reg_vertices[j].x)
                      inside[k] = true; // False otherwise
                    break;
                }// end switch statement for determining the window
              }// end for loop to iterate through vertices in intersecting polygon
              
              /* Now we know the sequence of inside/outside for each vertex relative to the considered window.
               * Based on the sequence of inside/outside, we determine which points will be added to the array verts_new
               * There are 4 cases:
               *   1) Outside -> Inside
               *      - Add the intersection between window and line segment connecting two points, as well as inside point
               *   2) Inside  -> Inside
               *      - Add the second inside point, neglect the first point
               *   3) Inside  -> Outside
               *      - Add the intersection between window and line segment connecting the two points
               *   4) Outside -> Outside
               *      - Add no points
               *
               *   Credit for algorithm attributed to Sutherland-Hodgeman Paper: #include <citation>
               */
               
              int temp_counter = 0;        // Number of vertices accumulated from this window
              for (int k = 0; k < vertex_counter; k++)
              {
                // Begin if statements for the 4 cases
                if      (! inside[k] &&   inside[(k+1) % vertex_counter]) // Case 1
                {
                  struct Point temp_pt = getIntersect(reg_vertices[j], reg_vertices[(j+1) % 4], verts_old[k], verts_old[(k+1) % vertex_counter]);
                  verts_new[temp_counter].x = temp_pt.x;                             verts_new[temp_counter++].y = temp_pt.y;
                  verts_new[temp_counter].x = verts_old[(k+1) % vertex_counter].x;   verts_new[temp_counter++].y = verts_old[(k+1) % vertex_counter].y;
                }// end if statement for case 1
                
                else if (  inside[k] &&   inside[(k+1) % vertex_counter]) // Case 2
                {
                  verts_new[temp_counter].x = verts_old[(k+1) % vertex_counter].x;   verts_new[temp_counter++].y = verts_old[(k+1) % vertex_counter].y;
                }// end else if statement for case 2
                
                else if (  inside[k] && ! inside[(k+1) % vertex_counter]) // Case 3
                {
                  struct Point temp_pt = getIntersect(reg_vertices[j], reg_vertices[(j+1) % 4], verts_old[k], verts_old[(k+1) % vertex_counter]);
                  verts_new[temp_counter].x = temp_pt.x;                             verts_new[temp_counter++].y = temp_pt.y;
                }// end else if statement for case 3
                
                else if (! inside[k] && ! inside[(k+1) % vertex_counter]) // Case 4
                {
                  
                }// end else if statement for case 4
              }// end for loop to iterate through the existing polygon vertices to add vertices to the new polygon
              
              for (int k = 0; k < temp_counter; k++)
              {
                verts_old[k].x = verts_new[k].x;
                verts_old[k].y = verts_new[k].y;
              }// end for loop to reassign new vertices to old vertices
              
              vertex_counter = temp_counter;
            }// end for loop to iterate through windows in the regular cell
            
            
            if (vertex_counter > 2) // WE HAVE A POLYGON!!
            {
              double result = 0.;
              for (int j = 0; j < vertex_counter; j++)
                result += verts_new[j].x * verts_new[(j+1) % vertex_counter].y - verts_new[j].y * verts_new[(j+1) % vertex_counter].x;
              areas[i] = 1./2. * fabs(result);
            }// end if statement for nonzero overlapping area
            else                    // No Polygon :(
              areas[i] = 0.;
            
            
          }// end if statement to check for valid cells
          else
            areas[i] = 0.;
        }// end for loop to loop through all cells in the stencil
        
        double temp_mass   = 0.;
        double temp_energy = 0.;
        for (int i = 0; i < stencilSize; i++)
        {
          int xdir = 0;  int ydir = 0;
          switch (i)
          {
            case 0: xdir = -1; ydir = -1; break;
            case 1: xdir =  0; ydir = -1; break;
            case 2: xdir =  1; ydir = -1; break;
            case 3: xdir = -1; ydir =  0; break;
            case 4: xdir =  0; ydir =  0; break;
            case 5: xdir =  1; ydir =  0; break;
            case 6: xdir = -1; ydir =  1; break;
            case 7: xdir =  0; ydir =  1; break;
            case 8: xdir =  1; ydir =  1; break;
          }// end switch statement for initializing the directions for the cell considered
          temp_mass   += areas[i] / cshift(c_A_old, xdir, ydir)     *    cshift(c_M_old, xdir, ydir);
          temp_energy += areas[i] / cshift(c_A_old, xdir, ydir)     *    cshift(c_E_old, xdir, ydir);
          
        }// end for loop to accumulate mass from surrounding cells
        
        c.c_M_new = temp_mass;
        c.c_E_new = temp_energy;
      }// end else statement for non-trivial case: vertex movement
    }// end if statement for interior cells
  }// end forall cells loop
  
  // Density calculation
  forall cells c in *M
  {
    if (c.c_interior)
      c.c_r_new = c.c_M_new / c.c_A_new;
  }// end forall cells loop
}// end remap_mass() method









/* calc_vert_area
 * Initializes the vertex quantity, v_A, which is the deformed area of the staggered cell.
 * The deformed area is needed for the momentum remap method.
 * The deformed area is calculated by summing the 4 areas of the quadrilaterals from each surrounding cell.
 */
task void calc_vert_area(sghMesh *M)
{
// Goal: Initialize v_A for internal vertices
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      int xpos = positionx(); // Vertex global position
      int ypos = positiony();
      bool valid_cells[4] = {1, 1, 1, 1}; // 4 Surrounding cells around vertex
      double areas[4] = {0., 0., 0., 0.}; // Areas from each surrounding cell
      
      struct Point vertex_pos = {v.v_x_old, v.v_y_old}; // Current vertex position
      
      struct Point vertex_south = {cshift(v_x_old,  0, -1), cshift(v_y_old,  0, -1)}; // Adjacent vertex positions
      struct Point vertex_east  = {cshift(v_x_old,  1,  0), cshift(v_y_old,  1,  0)};
      struct Point vertex_north = {cshift(v_x_old,  0,  1), cshift(v_y_old,  0,  1)};
      struct Point vertex_west  = {cshift(v_x_old, -1,  0), cshift(v_y_old, -1,  0)};
      
      struct Point vertex_midpoint_south = getMidpoint(vertex_pos, vertex_south); // Midpoint positions between vertex and adjacent vertices
      struct Point vertex_midpoint_east  = getMidpoint(vertex_pos, vertex_east );
      struct Point vertex_midpoint_north = getMidpoint(vertex_pos, vertex_north);
      struct Point vertex_midpoint_west  = getMidpoint(vertex_pos, vertex_west );
      
      // Cells arranged 0-SW, 1-SE, 2-NW, 3-NE. Same indexing for valid_cells
      if      (xpos == (0+1)        && ypos == (0+1)){          // SW corner
        valid_cells[2] = 0;            valid_cells[3] = 1;
        valid_cells[0] = 0;            valid_cells[1] = 0;
      }
      else if (xpos == (xpts - 1-1) && ypos == (0+1)){          // SE corner
        valid_cells[2] = 1;            valid_cells[3] = 0;
        valid_cells[0] = 0;            valid_cells[1] = 0;
      }
      else if (xpos == (xpts - 1-1) && ypos == (ypts - 1-1)){   // NE corner
        valid_cells[2] = 0;            valid_cells[3] = 0;
        valid_cells[0] = 1;            valid_cells[1] = 0;
      }
      else if (xpos == (0+1)        && ypos == (ypts - 1-1)){   // NW corner
        valid_cells[2] = 0;            valid_cells[3] = 0;
        valid_cells[0] = 0;            valid_cells[1] = 1;
      }
      else if (ypos == (0+1)){                                  // South wall
        valid_cells[2] = 1;            valid_cells[3] = 1;
        valid_cells[0] = 0;            valid_cells[1] = 0;
      }
      else if (xpos == (xpts - 1-1)){                           // East  wall
        valid_cells[2] = 1;            valid_cells[3] = 0;
        valid_cells[0] = 1;            valid_cells[1] = 0;
      }
      else if (ypos == (ypts - 1-1)){                           // North wall
        valid_cells[2] = 0;            valid_cells[3] = 0;
        valid_cells[0] = 1;            valid_cells[1] = 1;
      }
      else if (xpos == (0+1)){                                  // West  wall
        valid_cells[2] = 0;            valid_cells[3] = 1;
        valid_cells[0] = 0;            valid_cells[1] = 1;
      }
      else{                                                     // Interior
        valid_cells[2] = 1;            valid_cells[3] = 1;
        valid_cells[0] = 1;            valid_cells[1] = 1;
      }
      
      forall cells c in v
      {
        int local_index = 2 * positiony() + positionx(); // Local cell position
        if (valid_cells[local_index])
        {
          struct Point cell_pos = {c.c_x_old, c.c_y_old}; // Cell position
          switch (local_index)
          {
            case 0: // SW cell
              areas[local_index] = getArea(vertex_pos, vertex_midpoint_south, cell_pos, vertex_midpoint_west);
              break;
            case 1: // SE cell
              areas[local_index] = getArea(vertex_pos, vertex_midpoint_south, cell_pos, vertex_midpoint_east);
              break;
            case 2: // NW cell
              areas[local_index] = getArea(vertex_pos, vertex_midpoint_north, cell_pos, vertex_midpoint_west);
              break;
            case 3: // NE cell
              areas[local_index] = getArea(vertex_pos, vertex_midpoint_north, cell_pos, vertex_midpoint_east);
              break;
          }// end switch statement to check which cell we're in
        }// end if statement to check for valid cell
      }// end forall cells in vertex loop
      
      v.v_A = areas[0] + areas[1] + areas[2] + areas[3];
    }// end if statement to check for interior vertices
  }// end forall vertices loop
}// end calc_vert_area() method









/* map_mom_to_cell()
 * Initializes c_U and c_V for all cells
 * The cell based momentum makes it easier to remap in the remap_momentum() method
 * Finds the momentum by summing over each surrounding vertex on a cell
 */

task void map_mom_to_cell(sghMesh *M)
{
// Goal: Initialize c_U, c_V for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      struct Point cell_pos = {c.c_x_old, c.c_y_old};
      struct Point sur_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}; // Position of the surrounding vertices
      forall vertices v in c
      {
        sur_vertices[2 * positiony() + positionx()].x = v.v_x_old;
        sur_vertices[2 * positiony() + positionx()].y = v.v_y_old;
      }// end forall vertices in cell loop
      // Vertices are indexed: 0-SW, 1-SE, 2-NW, 3-NE
      
      struct Point south_midpoint = getMidpoint(sur_vertices[0], sur_vertices[1]);
      struct Point  east_midpoint = getMidpoint(sur_vertices[1], sur_vertices[3]);
      struct Point north_midpoint = getMidpoint(sur_vertices[2], sur_vertices[3]);
      struct Point  west_midpoint = getMidpoint(sur_vertices[0], sur_vertices[2]);
      
      double U_moms[4] = {0., 0., 0., 0.};
      double V_moms[4] = {0., 0., 0., 0.};
      forall vertices v in c
      {
        int local_index = 2 * positiony() + positionx();
        switch (local_index)
        {
          case 0: // SW vertex
            U_moms[local_index] = v.v_U_old * getArea(cell_pos, south_midpoint, sur_vertices[local_index], west_midpoint) / v.v_A;
            V_moms[local_index] = v.v_V_old * getArea(cell_pos, south_midpoint, sur_vertices[local_index], west_midpoint) / v.v_A;
            break;
          case 1: // SE vertex
            U_moms[local_index] = v.v_U_old * getArea(cell_pos, south_midpoint, sur_vertices[local_index], east_midpoint) / v.v_A;
            V_moms[local_index] = v.v_V_old * getArea(cell_pos, south_midpoint, sur_vertices[local_index], east_midpoint) / v.v_A;
            break;
          case 2: // NW vertex
            U_moms[local_index] = v.v_U_old * getArea(cell_pos, north_midpoint, sur_vertices[local_index], west_midpoint) / v.v_A;
            V_moms[local_index] = v.v_V_old * getArea(cell_pos, north_midpoint, sur_vertices[local_index], west_midpoint) / v.v_A;
            break;
          case 3: // NE vertex
            U_moms[local_index] = v.v_U_old * getArea(cell_pos, north_midpoint, sur_vertices[local_index], east_midpoint) / v.v_A;
            V_moms[local_index] = v.v_V_old * getArea(cell_pos, north_midpoint, sur_vertices[local_index], east_midpoint) / v.v_A;
            break;
          
        }// end switch statement to check what vertex we're in
      }// end forall vertices in cell loop
      c.c_U = U_moms[0] + U_moms[1] + U_moms[2] + U_moms[3];
      c.c_V = V_moms[0] + V_moms[1] + V_moms[2] + V_moms[3];
    }// end if statement to check for interior cell
  }// end forall cells loop
}// end map_mom_to_cell() method




/* remap_momentum()
 * Initialize v_U_new, v_V_new, v_u_new, v_v_new.
 * Take old mesh with old properties and map extensive momentum property (c_U, c_V)
 * to new mesh (v_U_new, v_V_new). From calculation of v_U_new, v_V_new, calculate v_u_new, v_v_new.
 */
task void remap_momentum(sghMesh *M)
{
// Goal: Initialize v_U_new, v_V_new, v_u_new, v_v_new for all vertices
  // Momentum calculation
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      bool theSame = true;
      forall cells c in v
        if (c.c_x_new != c.c_x_old || c.c_y_new != c.c_y_old) theSame = false;
      if (theSame)
      {
        v.v_U_new = v.v_U_old;
        v.v_V_new = v.v_V_old;
      }
      
      else
      {
        // Must clip() from each surrounding cell
        // For corner,   that's 1 cell
        // For edge,     that's 2 cells
        // For interior, that's 4 cells
        double areas[4] = {0., 0., 0., 0.};
        int xpos = positionx();
        int ypos = positiony();
        
        bool validCells[4] = {1, 1, 1, 1}; // Indexed as 0-SW, 1-SE, 2-NW, 3-NE
        bool not_A_Corner = true; // If corner, both components of momentum are 0 due to B.C.'s. Simplifies setting up staggered cells
        
        struct Point reg_cell_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}}; // The points defining the deformed cell
        
        forall cells c in v
        {
          reg_cell_vertices[2 * positiony() + positionx()].x = c.c_x_new; // _new for regular positions
          reg_cell_vertices[2 * positiony() + positionx()].y = c.c_y_new; // _new for regular positions
        }// end forall vertices around regular cell
        
        // Every time this indexing is performed, the order is as follows: 0-SW, 1-SE, 2-NW, 3-NE
        // Every time, the indexing will be reassigned so it's CCW:        0-SW, 1-SE, 2-NE, 3-NW
        
        struct Point temp_point = {reg_cell_vertices[2].x, reg_cell_vertices[2].y};
        reg_cell_vertices[2].x = reg_cell_vertices[3].x;  reg_cell_vertices[2].y = reg_cell_vertices[3].y;
        reg_cell_vertices[3].x = temp_point.x;            reg_cell_vertices[3].y = temp_point.y;
        
        struct Point vertex_pos = {v.v_x_new, v.v_y_new}; // Current vertex position
        
        struct Point vertex_south = {cshift(v_x_new,  0, -1), cshift(v_y_new,  0, -1)}; // Adjacent vertex positions
        struct Point vertex_east  = {cshift(v_x_new,  1,  0), cshift(v_y_new,  1,  0)};
        struct Point vertex_north = {cshift(v_x_new,  0,  1), cshift(v_y_new,  0,  1)};
        struct Point vertex_west  = {cshift(v_x_new, -1,  0), cshift(v_y_new, -1,  0)};
        
        struct Point vertex_midpoint_south = getMidpoint(vertex_pos, vertex_south); // Midpoint positions between vertex and adjacent vertices
        struct Point vertex_midpoint_east  = getMidpoint(vertex_pos, vertex_east );
        struct Point vertex_midpoint_north = getMidpoint(vertex_pos, vertex_north);
        struct Point vertex_midpoint_west  = getMidpoint(vertex_pos, vertex_west );
        
        if      (xpos == (0+1)        && ypos == (0+1)){            // SW corner
          v.v_U_new = 0.;         v.v_V_new = 0.;
          not_A_Corner = false;
        }
        else if (xpos == (xpts - 1-1) && ypos == (0+1)){            // SE corner
          v.v_U_new = 0.;         v.v_V_new = 0.;
          not_A_Corner = false;
        }
        else if (xpos == (xpts - 1-1) && ypos == (ypts - 1-1)){     // NE corner
          v.v_U_new = 0.;         v.v_V_new = 0.;
          not_A_Corner = false;
        }
        else if (xpos == (0+1)        && ypos == (ypts - 1-1)){     // NW corner    
          v.v_U_new = 0.;         v.v_V_new = 0.;
          not_A_Corner = false;
        }
        
        else if (ypos == (0+1)){                                    // South wall
          validCells[2] = 1;      validCells[3] = 1;
          validCells[0] = 0;      validCells[1] = 0;
          reg_cell_vertices[0].x = vertex_midpoint_west.x;
          reg_cell_vertices[0].y = vertex_midpoint_west.y;
          reg_cell_vertices[1].x = vertex_midpoint_east.x;
          reg_cell_vertices[1].y = vertex_midpoint_east.y;
        }
        else if (xpos == (xpts - 1-1)){                             // East  wall
          validCells[2] = 1;      validCells[3] = 0;
          validCells[0] = 1;      validCells[1] = 0;
          reg_cell_vertices[1].x = vertex_midpoint_south.x;
          reg_cell_vertices[1].y = vertex_midpoint_south.y;
          reg_cell_vertices[2].x = vertex_midpoint_north.x;
          reg_cell_vertices[2].y = vertex_midpoint_north.y;
        }
        else if (ypos == (ypts - 1-1)){                             // North wall
          validCells[2] = 0;      validCells[3] = 0;
          validCells[0] = 1;      validCells[1] = 1;
          reg_cell_vertices[2].x = vertex_midpoint_east.x;
          reg_cell_vertices[2].y = vertex_midpoint_east.y;
          reg_cell_vertices[3].x = vertex_midpoint_west.x;
          reg_cell_vertices[3].y = vertex_midpoint_west.y;
        }
        else if (xpos == (0+1)){                                    // West  wall
          validCells[2] = 0;      validCells[3] = 1;
          validCells[0] = 0;      validCells[1] = 1;
          reg_cell_vertices[3].x = vertex_midpoint_north.x;
          reg_cell_vertices[3].y = vertex_midpoint_north.y;
          reg_cell_vertices[0].x = vertex_midpoint_south.x;
          reg_cell_vertices[0].y = vertex_midpoint_south.y;
        }
        else{                                                       // Interior
          
        }
        
        struct Point def_vertex_pos = {v.v_x_old, v.v_y_old}; // Current vertex position
        
        struct Point def_vertex_south = {cshift(v_x_old,  0, -1), cshift(v_y_old,  0, -1)}; // Adjacent vertex positions
        struct Point def_vertex_east  = {cshift(v_x_old,  1,  0), cshift(v_y_old,  1,  0)};
        struct Point def_vertex_north = {cshift(v_x_old,  0,  1), cshift(v_y_old,  0,  1)};
        struct Point def_vertex_west  = {cshift(v_x_old, -1,  0), cshift(v_y_old, -1,  0)};
        
        struct Point def_vertex_sw    = {cshift(v_x_old, -1, -1), cshift(v_y_old, -1, -1)};
        struct Point def_vertex_se    = {cshift(v_x_old,  1, -1), cshift(v_y_old,  1, -1)};
        struct Point def_vertex_ne    = {cshift(v_x_old,  1,  1), cshift(v_y_old,  1,  1)};
        struct Point def_vertex_nw    = {cshift(v_x_old, -1,  1), cshift(v_y_old, -1,  1)};
        
        if (not_A_Corner)
        {
          forall cells c in v
          {
            int local_index = 2 * positiony() + positionx();
            if (validCells[local_index])
            {
              struct Point def_vertices[4] = {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
              def_vertices[0].x = vertex_pos.x;      def_vertices[0].y = vertex_pos.y;
              switch (local_index)
              {
                case 0: // SW cell
                  def_vertices[1].x = def_vertex_west.x;   def_vertices[1].y = def_vertex_west.y;  // West
                  def_vertices[2].x = def_vertex_sw.x;     def_vertices[2].y = def_vertex_sw.y;    // SW
                  def_vertices[3].x = def_vertex_south.x;  def_vertices[3].y = def_vertex_south.y; // South
                  break;
                case 1: // SE cell
                  def_vertices[1].x = def_vertex_east.x;   def_vertices[1].y = def_vertex_east.y;  // East
                  def_vertices[2].x = def_vertex_se.x;     def_vertices[2].y = def_vertex_se.y;    // SE
                  def_vertices[3].x = def_vertex_south.x;  def_vertices[3].y = def_vertex_south.y; // South
                  break;
                case 2: // NW cell
                  def_vertices[1].x = def_vertex_west.x;   def_vertices[1].y = def_vertex_west.y;  // West
                  def_vertices[2].x = def_vertex_nw.x;     def_vertices[2].y = def_vertex_nw.y;    // NW
                  def_vertices[3].x = def_vertex_north.x;  def_vertices[3].y = def_vertex_north.y; // North
                  break;
                case 3: // NE cell
                  def_vertices[1].x = def_vertex_east.x;   def_vertices[1].y = def_vertex_east.y;  // East
                  def_vertices[2].x = def_vertex_ne.x;     def_vertices[2].y = def_vertex_ne.y;    // NE
                  def_vertices[3].x = def_vertex_north.x;  def_vertices[3].y = def_vertex_north.y; // North
                  break;
              }// end switch statement to check which cell we're in
              // Now def_vertices is initialized, reg_cell_vertices is initialized, we're on an interior point,
              // and we're not on a corner. Just need to find the area of intersection between the two quads.
              
              struct Point verts_old[21]; // Vertices of the intersecting polygon - old
              struct Point verts_new[21]; // Vertices of the intersecting polygon - new
            
              for (int j = 0; j < 21; j++) // Initialize the arrays of polygon vertices
              {
                verts_old[j].x = -999;   verts_old[j].y = -999;
                verts_new[j].x = -999;   verts_new[j].y = -999;
              }// end for loop to initialize vertex locations
              
              int vertex_counter = 4;
              
              for (int j = 0; j < vertex_counter; j++)
              {
                verts_old[j].x = def_vertices[j].x;  verts_old[j].y = def_vertices[j].y;
                verts_new[j].x = def_vertices[j].x;  verts_new[j].y = def_vertices[j].y;
              }// end for loop to initialize the first four points of the polygon vertices
              
              
              
              for (int j = 0; j < 4; j++) // Iterate through windows; 4 for the 4 windows
              {
                // Reminder: {0, 1, 2, 3} = {SW, SE, NE, NW}
                // For j = 0 & 2, the windows are horizontal; for j = 1 & 3, the windows are vertical.
                // So, for checking j = 0, just compare y values. For checking j = 1, compare x values.
                
                bool inside[vertex_counter]; // Boolean array for vertices inside window
                
                for (int k = 0; k < vertex_counter; k++) // Initialize inside array to all false
                inside[k] = false;
              
                for (int k = 0; k < vertex_counter; k++) // Iterate through the deformed vertices to determine inside or outside.
                {
                  switch(j) // Switch statement to determine what window on the regular cell we're on
                  {
                    case 0: // Horizontal window on south part of cell: Inside is to the north
                      if (verts_old[k].y > reg_cell_vertices[j].y)
                        inside[k] = true; // False otherwise
                      break;
                    case 1: // Vertical   window on east  part of cell: Inside is to the west
                      if (verts_old[k].x < reg_cell_vertices[j].x)
                      inside[k] = true; // False otherwise
                    break;
                    case 2: // Horizontal window on north part of cell: Inside is to the south
                      if (verts_old[k].y < reg_cell_vertices[j].y)
                        inside[k] = true; // False otherwise
                      break;
                    case 3: // Vertical   window on west  part of cell: Inside is to the east
                      if (verts_old[k].x > reg_cell_vertices[j].x)
                        inside[k] = true; // False otherwise
                      break;
                  }// end switch statement for determining the window
                }// end for loop to iterate through vertices in intersecting polygon
              
                /* Now we know the sequence of inside/outside for each vertex relative to the considered window.
                 * Based on the sequence of inside/outside, we determine which points will be added to the array verts_new
                 * There are 4 cases:
                 *   1) Outside -> Inside
                 *      - Add the intersection between window and line segment connecting two points, as well as inside point
                 *   2) Inside  -> Inside
                 *      - Add the second inside point, neglect the first point
                 *   3) Inside  -> Outside
                 *      - Add the intersection between window and line segment connecting the two points
                 *   4) Outside -> Outside
                 *      - Add no points
                 *
                 *   Credit for algorithm attributed to Sutherland-Hodgeman Paper: #include <citation>
                 */
               
                int temp_counter = 0;        // Number of vertices accumulated from this window
                for (int k = 0; k < vertex_counter; k++)
                {
                  // Begin if statements for the 4 cases
                  if      (! inside[k] &&   inside[(k+1) % vertex_counter]) // Case 1
                  {
                    struct Point temp_pt = getIntersect(reg_cell_vertices[j], reg_cell_vertices[(j+1) % 4], verts_old[k], verts_old[(k+1) % vertex_counter]);
                    verts_new[temp_counter].x = temp_pt.x;                             verts_new[temp_counter++].y = temp_pt.y;
                    verts_new[temp_counter].x = verts_old[(k+1) % vertex_counter].x;   verts_new[temp_counter++].y = verts_old[(k+1) % vertex_counter].y;
                  }// end if statement for case 1
                  
                  else if (  inside[k] &&   inside[(k+1) % vertex_counter]) // Case 2
                  {
                    verts_new[temp_counter].x = verts_old[(k+1) % vertex_counter].x;   verts_new[temp_counter++].y = verts_old[(k+1) % vertex_counter].y;
                  }// end else if statement for case 2
                  
                  else if (  inside[k] && ! inside[(k+1) % vertex_counter]) // Case 3
                  {
                    struct Point temp_pt = getIntersect(reg_cell_vertices[j], reg_cell_vertices[(j+1) % 4], verts_old[k], verts_old[(k+1) % vertex_counter]);
                    verts_new[temp_counter].x = temp_pt.x;                             verts_new[temp_counter++].y = temp_pt.y;
                  }// end else if statement for case 3
                  
                  else if (! inside[k] && ! inside[(k+1) % vertex_counter]) // Case 4
                  {
                    
                  }// end else if statement for case 4
                }// end for loop to iterate through the existing polygon vertices to add vertices to the new polygon
                
                for (int k = 0; k < temp_counter; k++)
                {
                  verts_old[k].x = verts_new[k].x;
                  verts_old[k].y = verts_new[k].y;
                }// end for loop to reassign new vertices to old vertices
                
                vertex_counter = temp_counter;
              }// end for loop to iterate through windows in the regular cell
              
              if (vertex_counter > 2) // WE HAVE A POLYGON!!
              {
                double result = 0.;
                for (int j = 0; j < vertex_counter; j++)
                  result += verts_new[j].x * verts_new[(j+1) % vertex_counter].y - verts_new[j].y * verts_new[(j+1) % vertex_counter].x;
                areas[local_index] = 1./2. * fabs(result);
              }// end if statement for nonzero overlapping area
              else                    // No Polygon :(
                areas[local_index] = 0.;
            }// end if statement to check for valid cells
          }// end forall cells in vertex loop
          
          double total_U = 0.;
          double total_V = 0.;
          
          forall cells c in v
          {
            int local_index = 2 * positiony() + positionx();
            total_U += c.c_U * areas[local_index] / c.c_A_old;
            total_V += c.c_V * areas[local_index] / c.c_A_old;
          }// end forall cells in vertex loop
          
          v.v_U_new = total_U;
          v.v_V_new = total_V;
        }// end if statement to ensure not dealing with a corner stag. cell
      }// end else statement to check for non-trivial case of non-moving cells
    }// end if statement to check for interior vertex
  }// end forall vertices loop
  
  
  
  
  
  
  
  
  
  
  
  
  
  // Velocity calculation
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      v.v_u_new = v.v_U_new / v.v_M_new; // v_M_new has already been initialized in updateStaggeredMass()
      v.v_v_new = v.v_V_new / v.v_M_new; //      method prior to call to remap_Momentum().
    }// end if statement to check for interior vertices
  }// end forall vertices loop
  
  // Ensure boundary conditions transferred properly
  forall vertices v in *M
  {
    if (v.v_interior)
    {
      if (positionx() == (0+1) || positionx() == (xpts - 1-1))
      {
        v.v_u_new = 0.;    v.v_U_new = 0.;
      }// end if statement to check for boundary condition
      
      if (positiony() == (0+1) || positiony() == (ypts - 1-1))
      {
        v.v_v_new = 0.;    v.v_V_new = 0.;
      }// end if statement to apply boundary condition
    }// end if statement to check for interior vertices
  }// end forall vertices loop
  
}// end remap_momentum() method









/* remap_energy()
 * Initialize c_e_new.
 * Take old mesh with old properties and map extensive energy property (c_E_old)
 * to new mesh (c_E_new). From calculation of c_E_new, calculate c_e_new.
 * c_E_new is already calculated. Simply initialize c_e_new.
 */
task void remap_energy(sghMesh *M)
{
// Goal: Initialize c_e_new for all cells
  forall cells c in *M
  {
    if (c.c_interior)
    {
      // Internal energy approach
      c.c_e_new = c.c_E_new / c.c_M_new;
      
      // Total energy approach
      //double avg_speed_squared_vec[4] = {0., 0., 0., 0.};
      //forall vertices v in c
      //{
      //  avg_speed_squared_vec[2 * positiony() + positionx()] = (v.v_u_new * v.v_u_new + v.v_v_new * v.v_v_new) / 4.;
      //}// end forall vertices in cell loop
      //double avg_speed_squared = avg_speed_squared_vec[0] + avg_speed_squared_vec[1] + avg_speed_squared_vec[2] + avg_speed_squared_vec[3];
      //c.c_e_new = (c.c_E_new - c.c_M_new * avg_speed_squared / 2.) / c.c_M_new;
      
    }// end if statement to check for interior cell
  }// end forall cells loop
}// end remap_energy() method









/* RemapStep()
 * Performs the entirety of a mesh remap.
 * Mostly just separates tasks and calls methods.
 * Here are the steps:
 *   - Calculate the cell and vertex positions                  - remap_pos()
 *   - Calculate the cell area                                  - sgh_c_area()
 *   - Calculate the cell mass and density (cons of mass)       - remap_mass()
 *   - Calculate the vertex mass                                - updateStaggeredMass()
 *   - Calculate the vertex area                                - calc_vert_area()
 *   - Calculate the cell-based momentum                        - map_mom_to_cell()
 *   - Calculate the vertex momentum and velocity (cons of mom) - remap_momentum()
 *   - Calculate the cell total energy and SIE (cons of energy) - remap_energy()
 *   - Calculate the cell pressure                              - sgh_pressure()
 * No need to update c_q, v_f_x, v_f_y
 */
void RemapStep(sghMesh *M, int count, double Time)
{
  remap_pos(M);           // Initialize c_x_new, c_y_new, v_x_new, v_y_new
  sgh_c_area(M);          // Initialize c_A_new
  remap_mass(M);          // Initialize c_M_new, c_r_new
  updateStaggeredMass(M); // Initialize v_M_new
  calc_vert_area(M);      // Initialize v_A for momentum remap
  map_mom_to_cell(M);     // Initialize c_U, c_V for momentum remap
  remap_momentum(M);      // Initialize v_U_new, v_V_new, v_u_new, v_v_new
  remap_energy(M);        // Initialize c_E_new, c_e_new
  sgh_pressure(M);        // Initialize c_p_new
  
  updateAll(M);           // Reassign all new properties to old properties
}// end RemapStep() method
