
/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * -----
 * 
 */

#ifndef SCOUT_RENDERALL_SURFACE_H_
#define SCOUT_RENDERALL_SURFACE_H_

#include <cstdlib>

#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/renderall/renderall_base.h"
#include "scout/Runtime/opengl/glSurfaceRenderable.h"
#include "scout/Runtime/opengl/glCamera.h"

namespace scout{

class renderall_surface_rt : public renderall_base_rt {
  public:
    renderall_surface_rt(size_t width, size_t height, size_t depth,
        float* vertices, float* normals, float* colors, int num_vertices,
        glCamera* camera);

    ~renderall_surface_rt();

    void exec();

    void begin();

    void end();

    void addVolume(void* dataptr, unsigned volumenum){}

  private:
    glSDL* _glsdl;
    glSurfaceRenderable* _renderable;
    glCamera* _camera;
    float* _vertices, *_normals, *_colors;
    int _num_vertices;
    bool _localcamera;
};

} // end namespace scout

extern void __sc_begin_renderall_surface(size_t width, size_t height, size_t depth,
    float* vertices, float* normals, float* colors, size_t num_vertices, scout::glCamera* cam);

#endif // SCOUT_RENDERALL_SURFACE_H_

