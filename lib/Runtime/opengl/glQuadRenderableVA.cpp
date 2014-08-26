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
#include <cstdlib>
#include <iostream>
#include "scout/Runtime/opengl/glQuadRenderableVA.h"
#include "scout/Runtime/opengl/glTexture1D.h"
#include "scout/Runtime/opengl/glTexture2D.h"

//#define WITH_VERTICES_EDGES

using namespace std;
using namespace scout;

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

  _pbo = new glColorBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * xdim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  oglErrorCheck();
}

void glQuadRenderableVA::glQuadRenderableVA_2D()
{
  size_t xdim = _max_pt.x - _min_pt.x;
  size_t ydim = _max_pt.y - _min_pt.y;
  size_t xdim1 = xdim + 1;
  size_t ydim1 = ydim + 1;
  size_t numCells = xdim * ydim;
  size_t numVertices = xdim1 * ydim1;
  size_t numEdges = xdim * ydim1 + xdim1 * ydim;

  _pbo = new glColorBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * numCells, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _cells = (unsigned*)malloc(sizeof(unsigned) * numCells * 4);
  fill_cells();

  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 2 * numVertices, GL_STREAM_DRAW_ARB);
  fill_vbo();
  _vbo->release();

#ifdef WITH_VERTICES_EDGES
  _vpbo = new glColorBuffer;
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

  _epbo = new glColorBuffer;
  _epbo->bind();
  _epbo->alloc(sizeof(float) * 4 * numEdges, GL_STREAM_DRAW_ARB);
  _epbo->release();

  float4* ec = map_edge_colors();
  for(size_t i = 0; i < numEdges; ++i){
    ec[i].x = 1.0;
    ec[i].y = 1.0;
    ec[i].z = 1.0;
    ec[i].w = 1.0;
  }
  unmap_edge_colors();

  _edges = (unsigned*)malloc(sizeof(unsigned) * numEdges * 2);

  fill_edges();
#endif

  oglErrorCheck();
}


void glQuadRenderableVA::destroy()
{
  if (_pbo != 0) delete _pbo;
  if (_vpbo != 0) delete _vpbo;
  if (_epbo != 0) delete _epbo;
  if (_vbo != 0) delete _vbo;
  _pbo = NULL;
  _vpbo = NULL;
  _epbo = NULL;
  _vbo = NULL;
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

void glQuadRenderableVA::fill_vbo()
{
  float xdim1 = _max_pt.x - _min_pt.x + 1;
  float ydim1 = _max_pt.y - _min_pt.y + 1;

  float* verts = (float*)_vbo->mapForWrite();

  size_t i = 0;
  for(float y = 0; y < ydim1; y++) {
    for(float x = 0; x < xdim1; x++) {
      verts[i++] = x;
      verts[i++] = y;
    }
  }

  _vbo->unmap();
}

void glQuadRenderableVA::fill_edges()
{
  unsigned xdim = _max_pt.x - _min_pt.x;
  unsigned ydim = _max_pt.y - _min_pt.y;
  unsigned xdim1 = xdim + 1;

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

void glQuadRenderableVA::fill_cells()
{
  unsigned xdim = _max_pt.x - _min_pt.x;
  unsigned ydim = _max_pt.y - _min_pt.y;

  size_t i = 0;
  for(unsigned y = 0; y < ydim; y++) {
    for(unsigned x = 0; x < xdim; x++) {
      unsigned i0, i1;
      i0 = y * (xdim + 1) + x;
      i1 = (y + 1) * (xdim + 1) + x;
      _cells[i++] = i1;
      _cells[i++] = i0;
      _cells[i++] = i0 + 1;
      _cells[i++] = i1 + 1;
    }
  }
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
}

void glQuadRenderableVA::unmap_vertex_colors()
{
  _vpbo->unmap();
}

void glQuadRenderableVA::unmap_edge_colors()
{
  _epbo->unmap();
}

void glQuadRenderableVA::draw(glCamera* camera)
{
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  size_t xdim = _max_pt.x - _min_pt.x;
  size_t ydim = _max_pt.y - _min_pt.y;
  size_t xdim1 = xdim + 1;
  size_t ydim1 = ydim + 1;
  size_t numCells = xdim * ydim;
  size_t numVertices = xdim1 * ydim1;
  size_t numEdges = xdim * ydim1 + xdim1 * ydim;
  
  float s = 80.0/max(xdim, ydim);

  float* verts =(float*)_vbo->mapForRead();
  float4* colors = (float4*)_pbo->mapForRead();

  size_t ci = 0;
  for(unsigned y = 0; y < ydim; y++) {
    for(unsigned x = 0; x < xdim; x++) {
      float4 color = colors[ci];

      glColor4f(color.x, color.y, color.z, color.w);
      ++ci;

      glBegin(GL_QUADS);

      int i = (y * (xdim+1) * 2) + (x*2);

      glVertex2f(verts[i], verts[i+1]);
      glVertex2f(verts[i+2], verts[i+3]);

      i = ((y+1) * (xdim+1) * 2) + (x*2);
      glVertex2f(verts[i+2], verts[i+3]);  
      glVertex2f(verts[i], verts[i+1]);        
      
      glEnd();
    }
  }

  _vbo->unmap();
  _pbo->unmap();

#ifdef WITH_VERTICES_EDGES
  _vbo->bind();
  glVertexPointer(2, GL_FLOAT, 0, 0);
  _vbo->release();

  _epbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0); 
  _epbo->release();

  glLineWidth(s);
  glDrawElements(GL_LINES, numEdges * 2, GL_UNSIGNED_INT, _edges);

  _vpbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0); 
  _vpbo->release();

  glPointSize(s);  
  glDrawArrays(GL_POINTS, 0, numVertices);
#endif

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  oglErrorCheck();
}
