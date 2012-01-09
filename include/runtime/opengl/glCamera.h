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

#ifndef SCOUT_GL_CAMERA_H_
#define SCOUT_GL_CAMERA_H_

#include "runtime/opengl/opengl.h"
#include "runtime/opengl/vectors.h"

namespace scout
{
  
  // ..... glCamera
  //
  class glCamera {

   public:
    glCamera();
    
    glCamera(float aperture, 
             const glfloat3& pos,
             const glfloat3& direction,
             const glfloat3& up,
             float near, float far);
    
    ~glCamera() 
    { /* no-op for now */  }

    bool resize(int width, int height);

    float aspectRatio() const
    { return (float)win_width / win_height; }

    void setPosition(const glfloat3& pt)
    { position = pt; }

    void setLookAt(const glfloat3& pt)
    { look_at = pt; }

    void setRotationPoint(const glfloat3& pt)
    { rotation_point = pt; }    
    
    float        near, far;
    float        aspect;    
    float        fov;
    float        win_width, win_height;
    float        eye_sep;
    float        focal_length;
    glfloat3   position;
    glfloat3   look_at;
    glfloat3   up;
    glfloat3   rotation_point;
  };

  extern std::ostream& operator<<(std::ostream& os, const glCamera& entry);  
}

#endif
