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
#include "scout/Runtime/opengl/glVertexBuffer.h"
#include "scout/Runtime/opengl/glColorBuffer.h"

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
    
    float4* map_colors();
    float4* map_vertex_colors();
    float4* map_edge_colors();
    
    void unmap_colors();
    void unmap_vertex_colors();
    void unmap_edge_colors();
    
    void draw(glCamera* camera);
    
    void setMinPoint(glfloat3 pt)
    { _min_pt = pt; }
    
    void setMaxPoint(glfloat3 pt)
    { _max_pt = pt; }
    
  private:
    void destroy();
    
    void glQuadRenderableVA_1D();
    void glQuadRenderableVA_2D();
    
    void fill_vbo();
    void fill_cvbo();
    void fill_evbo();
    
  private:
    glfloat3 _min_pt;
    glfloat3 _max_pt;

    size_t _xdim;
    size_t _ydim;
    size_t _xdim1;
    size_t _ydim1;
    size_t _numCells;
    size_t _numVertices;
    size_t _numEdges;
    
    glVertexBuffer* _vbo;
    glVertexBuffer* _cvbo;
    glVertexBuffer* _evbo;
    
    glColorBuffer* _pbo;
    glColorBuffer* _vpbo;
    glColorBuffer* _epbo;

    float4* _edgeColors;
    float4* _cellColors;
    
    bool _drawCells;
    bool _drawVertices;
    bool _drawEdges;
  };
  
}

#endif
