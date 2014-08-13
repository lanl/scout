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

#include <unistd.h>

#include <iostream>
#include "scout/Runtime/opengl/glQuadRenderableVA.h"
#include "scout/Runtime/opengl/glTexture1D.h"
#include "scout/Runtime/opengl/glTexture2D.h"

//#define WITH_VERTICES_EDGES

using namespace std;
using namespace scout;

namespace{

  static const float CELL_SIZE = 1;

} // end namespace

glQuadRenderableVA::glQuadRenderableVA(const glfloat3 &min_pt, const glfloat3 &max_pt)
{
  setMinPoint(min_pt);
  setMaxPoint(max_pt);

  if ((max_pt.y - min_pt.y) == 0) {
    glQuadRenderableVA_1D();
  } else {
    glQuadRenderableVA_2D();
  }

}


void glQuadRenderableVA::glQuadRenderableVA_1D()
{
  size_t xdim = _max_pt.x - _min_pt.x;

  _ntexcoords = 1;
  _texture = new glTexture1D(xdim);
  _texture->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  _texture->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _texture->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _pbo = new glTextureBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * xdim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);  // we use a quad for 1D meshes...
  fill_vbo(_min_pt.x, _min_pt.x, _max_pt.x, _max_pt.x);
  _vbo->release();
  _nverts = 4;

  _tcbo = new glTexCoordBuffer;
  _tcbo->bind();
  _tcbo->alloc(sizeof(float) * 4, GL_STREAM_DRAW_ARB);  // one-dimensional texture coordinates.
  fill_tcbo1d(0.0f, 1.0f);
  _tcbo->release();

  oglErrorCheck();

}

void glQuadRenderableVA::glQuadRenderableVA_2D()
{
  size_t xdim = _max_pt.x - _min_pt.x;
  size_t ydim = _max_pt.y - _min_pt.y;
  size_t xdim1 = xdim + 1;
  size_t ydim1 = ydim + 1;
  size_t numVertices = xdim1 * ydim1;
  size_t numEdges = xdim * ydim1 + xdim1 * ydim;

  _ntexcoords = 2;
  _texture = new glTexture2D(xdim, ydim);
  _texture->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  _texture->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _texture->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _pbo = new glTextureBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * xdim * ydim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _vpbo = new glTextureBuffer;
  _vpbo->bind();
  _vpbo->alloc(sizeof(float) * 4 * numVertices, GL_STREAM_DRAW_ARB);
  _vpbo->release();
  
  float4* vc = map_vertex_colors();
  for(size_t i = 0; i < numVertices; ++i){
    vc[i].x = 1.0;
    vc[i].y = 0.0;
    vc[i].z = 0.0;
    vc[i].w = 1.0;
  }
  unmap_vertex_colors();

  _epbo = new glTextureBuffer;
  _epbo->bind();
  _epbo->alloc(sizeof(float) * 4 * numEdges, GL_STREAM_DRAW_ARB);
  _epbo->release();

  float4* ec = map_edge_colors();
  for(size_t i = 0; i < numVertices; ++i){
    ec[i].x = 0.0;
    ec[i].y = 0.0;
    ec[i].z = 1.0;
    ec[i].w = 1.0;
  }
  unmap_edge_colors();

  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);
  fill_vbo(_min_pt.x, _min_pt.y, _max_pt.x, _max_pt.y);
  _vbo->release();
  _nverts = 4;

#ifdef WITH_VERTICES_EDGES
  _mvbo = new glVertexBuffer;
  _mvbo->bind();
  _mvbo->alloc(sizeof(float) * 3 * numVertices, GL_STREAM_DRAW_ARB);
  fill_mvbo();
  _mvbo->release();

  _edges = (unsigned*)malloc(sizeof(unsigned) * numEdges * 2);
  fill_edges();
#endif

  _tcbo = new glTexCoordBuffer;
  _tcbo->bind();
  _tcbo->alloc(sizeof(float) * 8, GL_STREAM_DRAW_ARB);  // two-dimensional texture coordinates.
  fill_tcbo2d(0.0f, 0.0f, 1.0f, 1.0f);
  _tcbo->release();

  oglErrorCheck();
}


void glQuadRenderableVA::destroy()
{
  if (_texture != 0) delete _texture;
  if (_pbo != 0) delete _pbo;
  if (_vpbo != 0) delete _vpbo;
  if (_epbo != 0) delete _epbo;
  if (_vbo != 0) delete _vbo;
  if (_mvbo != 0) delete _mvbo;
  if (_tcbo != 0) delete _tcbo;
  _texture = NULL;
  _pbo = NULL;
  _vpbo = NULL;
  _epbo = NULL;
  _vbo = NULL;
  _mvbo = NULL;
  _tcbo = NULL;
}

glQuadRenderableVA::~glQuadRenderableVA()
{
  destroy();
}


void glQuadRenderableVA::initialize(glCamera* camera) 
{
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  _width = _max_pt.x - _min_pt.x;
  _height = _max_pt.y - _min_pt.y;
  
  static const float pad = 0.05;

  if(_height == 0){
    float px = pad * _width;
    gluOrtho2D(-px, _width + px, -px, _width + px);

  }
  else{
    if(_width >= _height){
      float px = pad * _width;
      float py = (1 - float(_height)/_width) * _width * 0.50;
      gluOrtho2D(-px, _width + px, -py - px, _width - py + px);
    }
    else{
      float py = pad * _height;
      float px = (1 - float(_width)/_height) * _height * 0.50;
      gluOrtho2D(-px - py, _width + px + py, -py, _height + py);
    }

  }

  glMatrixMode(GL_MODELVIEW);

  glLoadIdentity();

  glClearColor(0.5, 0.55, 0.65, 0.0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}


void glQuadRenderableVA::fill_vbo(float x0,
    float y0,
    float x1,
    float y1)
{

  float* verts = (float*)_vbo->mapForWrite();

  verts[0] = x0;
  verts[1] = y0;
  verts[2] = 0.0f;

  verts[3] = x1;
  verts[4] = y0;
  verts[5] = 0.f;

  verts[6] = x1;
  verts[7] = y1;
  verts[8] = 0.0f;

  verts[9] = x0;
  verts[10] = y1;
  verts[11] = 0.0f;

  _vbo->unmap();
}

void glQuadRenderableVA::fill_mvbo()
{
  float xdim1 = _max_pt.x - _min_pt.x + 1;
  float ydim1 = _max_pt.y - _min_pt.y + 1;

  float* verts = (float*)_mvbo->mapForWrite();

  size_t i = 0;
  for(float y = 0; y < ydim1; y++) {  
    for(float x = 0; x < xdim1; x++) {
      verts[i++] = x;
      verts[i++] = y;
      verts[i++] = 0.0f;
    }
  }

  _mvbo->unmap();
}

void glQuadRenderableVA::fill_edges()
{
  unsigned xdim = _max_pt.x - _min_pt.x;
  unsigned ydim = _max_pt.y - _min_pt.y;
  unsigned xdim1 = xdim + 1.0f;

  size_t i = 0;
  for(unsigned y = 0; y <= ydim; ++y) {
    for(unsigned x = 0; x < xdim; ++x) {
      _edges[i++] = y * xdim1 + x;
      _edges[i++] = y * xdim1 + x + 1;
    }
  }

  for(unsigned x = 0; x <= xdim; ++x) {
    for(unsigned y = 0; y < ydim; ++y) {
      _edges[i++] = y * xdim1 + x;
      _edges[i++] = (y + 1) * xdim1 + x;
    }
  }
}

void glQuadRenderableVA::fill_tcbo2d(float x0,
    float y0,
    float x1,
    float y1)
{

  float* coords = (float*)_tcbo->mapForWrite();

  coords[0] = x0;
  coords[1] = y0;

  coords[2] = x1;
  coords[3] = y0;

  coords[4] = x1;
  coords[5] = y1;

  coords[6] = x0;
  coords[7] = y1;

  _tcbo->unmap();

}


void glQuadRenderableVA::fill_tcbo1d(float start, float end)
{

  float* coords = (float*)_tcbo->mapForWrite();

  coords[0] = start;
  coords[1] = end;
  coords[2] = end;
  coords[3] = start;

  _tcbo->unmap();
}


void glQuadRenderableVA::alloc_texture()
{
  _pbo->bind();
  _texture->initialize(0);
  _pbo->release();
}

void glQuadRenderableVA::alloc_vertex_texture()
{
  _vpbo->bind();
  _texture->initialize(0);
  _vpbo->release();
}

void glQuadRenderableVA::alloc_edge_texture()
{
  _epbo->bind();
  _texture->initialize(0);
  _epbo->release();
}

GLuint glQuadRenderableVA::get_buffer_object_id()
{
  return _pbo->id();
}


float4* glQuadRenderableVA::map_colors()
{
  return (float4*)_pbo->mapForWrite();
}

float4* glQuadRenderableVA::map_vertex_colors()
{
  return (float4*)_vpbo->mapForWrite();
}

float4* glQuadRenderableVA::map_edge_colors()
{
  return (float4*)_epbo->mapForWrite();
}

void glQuadRenderableVA::unmap_colors()
{
  _pbo->unmap();
  _pbo->bind();
  _texture->initialize(0);
  _pbo->release();
}

void glQuadRenderableVA::unmap_vertex_colors()
{
  _vpbo->unmap();
  _vpbo->bind();
  _texture->initialize(0);
  _vpbo->release();
}

void glQuadRenderableVA::unmap_edge_colors()
{
  _epbo->unmap();
  _epbo->bind();
  _texture->initialize(0);
  _epbo->release();
}

void glQuadRenderableVA::draw(glCamera* camera)
{
  size_t xdim = _max_pt.x - _min_pt.x;
  size_t ydim = _max_pt.y - _min_pt.y;
  size_t xdim1 = xdim + 1;
  size_t ydim1 = ydim + 1;
  size_t numVertices = xdim1 * ydim1;
  size_t numEdges = xdim * ydim1 + xdim1 * ydim;

  _pbo->bind();
  _texture->enable();
  _texture->update(0);
  _pbo->release();

  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  _tcbo->bind();
  glTexCoordPointer(_ntexcoords, GL_FLOAT, 0, 0);

  glEnableClientState(GL_VERTEX_ARRAY);
  _vbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);

  oglErrorCheck();

  glDrawArrays(GL_POLYGON, 0, _nverts);

  glDisableClientState(GL_VERTEX_ARRAY);
  _vbo->release();

  _tcbo->release();
  _texture->disable();

  glDisableClientState(GL_TEXTURE_COORD_ARRAY);

#ifdef WITH_VERTICES_EDGES
  glEnableClientState(GL_VERTEX_ARRAY);
  _mvbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0); 

  glLineWidth(5.0);
  _epbo->bind();
  _texture->enable();
  _texture->update(0);
  _epbo->release();

  glDrawElements(GL_LINES, numEdges*2, GL_UNSIGNED_INT, _edges);

  glPointSize(5.0);
  _vpbo->bind();
  _texture->enable();
  _texture->update(0);
  _vpbo->release();

  glDrawArrays(GL_POINTS, 0, numVertices);
  glDisableClientState(GL_VERTEX_ARRAY);
  _mvbo->release();
#endif
  
  oglErrorCheck();

  sleep(1);
}
