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

#ifndef SCOUT_GL_UNIFORM_RENDERABLE_H_
#define SCOUT_GL_UNIFORM_RENDERABLE_H_

#include "scout/Runtime/opengl/vectors.h"
#include "scout/Runtime/opengl/glRenderable.h"
#include "scout/Runtime/opengl/glVertexBuffer.h"
#include "scout/Runtime/opengl/glColorBuffer.h"
#include "scout/Runtime/opengl/vmath.h"

namespace scout{

  class glUniformRenderable : public glRenderable{
  public:
    glUniformRenderable(size_t width, size_t height);
    
    ~glUniformRenderable();

    void initialize(glCamera* camera);
    
    float4* map_colors();
    float4* map_vertex_colors();
    float4* map_edge_colors();
    
    void unmap_colors();
    void unmap_vertex_colors();
    void unmap_edge_colors();
    
    void draw(glCamera* camera);

  private:
    size_t width_;
    size_t height_;
    
    size_t numCells_;
    size_t numVertices_;
    size_t numEdges_;

    glVertexBuffer* cellPoints_;
    glVertexBuffer* vertexPoints_;
    glVertexBuffer* edgeLinesAdj_;
    
    glColorBuffer* cellColors_;
    glColorBuffer* vertexColors_;
    glColorBuffer* edgeColors_;

    bool drawCells_;
    bool drawVertices_;
    bool drawEdges_;

    GLuint cellProgram_;
    GLuint cellVAO_;

    GLuint vertexProgram_;
    GLuint vertexVAO_;

    GLuint edgeProgram_;
    GLuint edgeVAO_;

    vmath::mat4 mvp_;

    GLint mvpCellLoc_;
    GLint mvpEdgeLoc_;
    GLint mvpVertexLoc_;
    GLint edgeWidthEdgeLoc_;
    GLint pointSizeVertexLoc_;
  };

} // end namespace scout

#endif // SCOUT_GL_UNIFORM_RENDERABLE_VA_H_
