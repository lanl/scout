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

#ifndef SCOUT_GL_QUAD_RENDERABLE_VA_H_
#define SCOUT_GL_QUAD_RENDERABLE_VA_H_

#include "scout/Runtime/opengl/vectors.h"
#include "scout/Runtime/opengl/glRenderable.h"
#include "scout/Runtime/opengl/glTexture.h"
#include "scout/Runtime/opengl/glVertexBuffer.h"
#include "scout/Runtime/opengl/glTextureBuffer.h"
#include "scout/Runtime/opengl/glTexCoordBuffer.h"

namespace scout
{
  // ---- glQuadRenderable
  //
  class glQuadRenderableVA: public glRenderable
  {
   public:
    glQuadRenderableVA(const glfloat3 &min_pt, const glfloat3 &max_pt);
    ~glQuadRenderableVA();

    void initialize(glCamera* camera);
    GLuint get_buffer_object_id();
    float4* map_colors();
    float4* map_vertex_colors();
    float4* map_edge_colors();
    void alloc_texture();
    void alloc_vertex_texture();
    void alloc_edge_texture();
    void unmap_colors();
    void unmap_vertex_colors();
    void unmap_edge_colors();

    void draw(glCamera* camera);

    void setMinPoint(glfloat3 pt)
    { _min_pt = pt; }

    void setMaxPoint(glfloat3 pt)
    { _max_pt = pt; }

    void setTexture(glTexture *texture)
    { _texture = texture; }

    glTexture* texture() const
    { return _texture; }

    void setTexCoordScale(float s)
    { _tex_coord_scale = s; }

  private:
    void destroy();
    void glQuadRenderableVA_1D();
    void glQuadRenderableVA_2D();
    void fill_vbo(float x0, float y0, float x1, float y1);
    void fill_mvbo();
    void fill_edges();
    void fill_tcbo2d(float x0, float y0, float x1, float y1);
    void fill_tcbo1d(float start, float end);

   private:
      glVertexBuffer* _vbo;
      glVertexBuffer* _mvbo;
      unsigned* _edges;
      glTexture* _texture;
      glTextureBuffer* _pbo;
      glTextureBuffer* _vpbo;
      glTextureBuffer* _epbo;
      glTexCoordBuffer* _tcbo;
      unsigned short _ntexcoords;
      unsigned int _nverts;
      glfloat3 _min_pt;
      glfloat3 _max_pt;
      float _tex_coord_scale;
      size_t _width;
      size_t _height;
  };

}

#endif
