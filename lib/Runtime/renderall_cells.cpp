/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#include <stdlib.h>

#ifndef SCOUT_RENDERALL_CELLS_H_
#define SCOUT_RENDERALL_CELLS_H_


// ----- alloc_static_geom
//
void *alloc_static_geom(size_t nbytes);

// ----- free_static_geom
//
void free_static_geom(void* geom_p);


// ----- alloc_cell_colors
//
void *alloc_colors(size_t ncells);


// ----- free_cell_colors
//
void free_colors(void* colors_p);


// ----- renderall_cells_1d
//
void renderall_uni_cells_1d(...)
{
  /*
  if (mesh->geom_p == 0) {
    mesh->geom_p = alloc_static_geom(sizeof(float) * 2 * 4);
  }
    
    glVertexBufferObject* vbo = new glVertexBuffer();
    vbo->alloc(sizeof(float) * 2 * 4, GL_STATIC_DRAW);  // 4 corners, (x,y) coord per corner.
    mesh->geom_p = (void*)vbo;
  }
  */
  // Steps:
  //
  //    1. Build a pixel buffer object to hold the array of colors
  //       (mapping to cell count).
  //
  //    2. Build a vertex buffer object that represents the 1d mesh.
  //       Perhaps a thin quad vs. a line for better visibility. 
  //
  //
  //    3. Call the 'renderall' loop body to fill the PBO array w/
  //       colors.
  //
  //    4. Render the VBO with the PBO data as a texture.
  //
  //    5. Swap buffers.
  //

  // Notes:
  //
  //   * We should support 'renderall' loop body on both the CPU and
  //     GPU.  If using CUDA we'll need to add the details to do
  //     OpenGL and CUDA interop.
  //
  //   * Can we find a way to reuse the VBO, PBO and texture for other
  //     invocations of the loop construct (or correctly handle the
  //     repeated invocation of the loop).
}


// ----- renderall_cells_2d
//
void renderall_cells_2d(...)
{
  // Steps:
  //
  //    1. Build a pixel buffer object to hold the array of colors
  //       (mapping to cell count and mesh shape).
  //  
  //    2. Build a vertex buffer object that represents the 2d mesh.
  //
  //    3. Call the 'renderall' loop body to fill the PBO array w/
  //       colors.
  //
  //    4. Render the VBO with the PBO data as a texture.
  //
  //    5. Swap buffers.

  // Notes:
  //
  //   * We should support 'renderall' loop body on both the CPU and
  //     GPU.  If using CUDA we'll need to add the details to do
  //     OpenGL and CUDA interop.
  //
  //   * Can we find a way to reuse the VBO, PBO and texture for other
  //     invocations of the loop construct (or correctly handle the
  //     repeated invocation of the loop).
}


/*
// ----- renderall_cells_3d
// volume raycaster goes here...
void renderall_cells_3d(...)
{
  // need to look at Sujin's code for how to do this. 
}
*/

#endif 


