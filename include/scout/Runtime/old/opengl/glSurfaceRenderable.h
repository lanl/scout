#ifndef GL_SURFACE_RENDERABLE_H_
#define GL_SURFACE_RENDERABLE_H_

#include "scout/Runtime/opengl/glRenderable.h"
#include "scout/Runtime/opengl/glCamera.h"

// ----- glSurfaceRenderable 
// 

namespace scout
{

  class glSurfaceRenderable: public glRenderable {

    public:

      glSurfaceRenderable(int nx, int ny, int nz,
          float* vertices, float* normals, float* colors, int num_vertices,
          glCamera* camera);

      void initialize(glCamera* camera);

      void draw(glCamera* camera);

    private:
      int _nx;
      int _ny;
      int _nz;
      float* _vertices;
      float* _normals;
      float* _colors;
      int    _num_vertices;
  };

}
#endif
