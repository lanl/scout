/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *----- 
 * 
 */
#include <iostream>
#include "runtime/opengl/glCamera.h"

using namespace scout;


// ---- glCamera
//
glCamera::glCamera()
{
  fov          = 50.0f;
  focal_length = 70.0f;
  eye_sep      = focal_length / 20.0f;  
  
  near  = 1.0;
  far   = 1000.0;

  position[0] =   0.0;
  position[1] =   0.0;
  position[2] =  -focal_length;
  
  look_at[0] = -position[0];
  look_at[1] = -position[1];
  look_at[2] = -position[2];

  up[0] = 0.0;
  up[1] = 1.0;
  up[2] = 0.0;

  rotation_point[0] = 0.0;
  rotation_point[1] = 0.0;
  rotation_point[2] = 0.0;
  
  win_width  = 0;
  win_height = 0;
  aspect     = 0.0;
}


// ---- glCamera
// 
glCamera::glCamera(float field_of_view, 
                   const glfloat3& pos,
                   const glfloat3& at,
                   const glfloat3& up_vec,
                   float near_pos, float far_pos)
    : position(pos),
      look_at(at),
      up(up_vec)
{
  fov   = field_of_view;
  near  = near_pos;
  far   = far_pos;

  // We don't initialize the camera's window width and height.
  // Instead that should be handled by window resize events.
  win_width  = 0;
  win_height = 0;

  focal_length = 70.0f;
  eye_sep      = focal_length / 30.0f;  
}


// ----- resize
//
bool glCamera::resize(int width, int height)
{
  if (win_width  != width || win_height != height) {
    win_width  = float(width);
    win_height = float(height);
    aspect = win_width / win_height;
    return true;
  } else {
    return false;
  }
}


namespace scout
{
  std::ostream& operator<<(std::ostream& os, const glCamera& camera)
  {
    os << "*** camera:\n";
    os << "  near  : " << camera.near   << ", far: " << camera.far << "\n";
    os << "  aspect: " << camera.aspect << "\n";
    os << "  field of view: " << camera.fov << "\n";
    os << "  position: ("
       << camera.position[0] << ", " 
       << camera.position[1] << ", "
       << camera.position[2] << ")\n";
    os << "  look at: ("
       << camera.look_at[0] << ", " 
       << camera.look_at[1] << ", "
       << camera.look_at[2] << ")\n";
    os << "  up: ("
       << camera.up[0] << ", " 
       << camera.up[1] << ", "
       << camera.up[2] << ")\n";    
    os << "***\n";
    
    return os;
  }
}
