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

using namespace std;
using namespace scout;

glQuadRenderableVA::glQuadRenderableVA(const glfloat3 &min_pt,
                                       const glfloat3 &max_pt)
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
  _xdim = _max_pt.x - _min_pt.x;

  _pbo = new glColorBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * _xdim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  oglErrorCheck();
}

void glQuadRenderableVA::glQuadRenderableVA_2D()
{
  _drawCells = false;
  _drawVertices = false;
  _drawEdges = false;
  
  _xdim = _max_pt.x - _min_pt.x;
  _ydim = _max_pt.y - _min_pt.y;
  _xdim1 = _xdim + 1;
  _ydim1 = _ydim + 1;
  _numCells = _xdim * _ydim;
  _numVertices = _xdim1 * _ydim1;
  _numEdges = _xdim * _ydim1 + _xdim1 * _ydim;

  _pbo = new glColorBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * _numCells * 4, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _cellColors = (float4*)malloc(sizeof(float4) * _numCells);
  
  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * _numVertices, GL_STREAM_DRAW_ARB);
  fill_vbo();
  _vbo->release();
  
  _cvbo = new glVertexBuffer;
  _cvbo->bind();
  _cvbo->alloc(sizeof(float) * 3 * _numCells * 4, GL_STREAM_DRAW_ARB);
  fill_cvbo();
  _cvbo->release();

  _vpbo = new glColorBuffer;
  _vpbo->bind();
  _vpbo->alloc(sizeof(float) * 4 * _numVertices, GL_STREAM_DRAW_ARB);
  _vpbo->release();

  float4* vc = (float4*)_vpbo->mapForWrite();
  for(size_t i = 0; i < _numVertices; ++i){
    vc[i].x = 1.0;
    vc[i].y = 0.0;
    vc[i].z = 0.0;
    vc[i].w = 1.0;
  }
  _vpbo->unmap();
  
  _edgeColors = (float4*)malloc(sizeof(float4) * _numEdges);
  
  _epbo = new glColorBuffer;
  _epbo->bind();
  _epbo->alloc(sizeof(float) * 4 * _numEdges * 2, GL_STREAM_DRAW_ARB);
  _epbo->release();

  float4* ec = (float4*)_epbo->mapForWrite();
  for(size_t i = 0; i < _numEdges * 2; i += 2){
    ec[i].x = 1.0;
    ec[i].y = 1.0;
    ec[i].z = 1.0;
    ec[i].w = 1.0;
    
    ec[i+1].x = 1.0;
    ec[i+1].y = 1.0;
    ec[i+1].z = 1.0;
    ec[i+1].w = 1.0;
  }
  _epbo->unmap();
  
  _edgeIndices = (unsigned*)malloc(sizeof(unsigned) * _numEdges * 2);
  fill_edge_indices();

  oglErrorCheck();
}


void glQuadRenderableVA::destroy()
{
  if (_vbo) delete _vbo;
  if (_cvbo) delete _cvbo;
  if (_pbo) delete _pbo;
  if (_vpbo) delete _vpbo;
  if (_epbo) delete _epbo;
  if (_edgeColors) free(_edgeColors);

  _vbo = NULL;
  _cvbo = NULL;
  _pbo = NULL;
  _vpbo = NULL;
  _epbo = NULL;
  _edgeColors = NULL;
}

glQuadRenderableVA::~glQuadRenderableVA()
{
  destroy();
}


void glQuadRenderableVA::initialize(glCamera* camera) 
{
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  static const float pad = 0.05;

  if(_ydim == 0){
    float px = pad * _xdim;
    gluOrtho2D(-px, _xdim + px, -px, _xdim + px);

  }
  else{
    if(_xdim >= _ydim){
      float px = pad * _xdim;
      float py = (1 - float(_ydim)/_xdim) * _xdim * 0.50;
      gluOrtho2D(-px, _xdim + px, -py - px, _xdim - py + px);
    }
    else{
      float py = pad * _ydim;
      float px = (1 - float(_xdim)/_ydim) * _ydim * 0.50;
      gluOrtho2D(-px - py, _xdim + px + py, -py, _ydim + py);
    }

  }

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glClearColor(0.5, 0.55, 0.65, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void glQuadRenderableVA::fill_vbo()
{
  float* verts = (float*)_vbo->mapForWrite();

  size_t i = 0;
  for(float y = 0; y < _ydim1; y++) {
    for(float x = 0; x < _xdim1; x++) {
      verts[i++] = x;
      verts[i++] = y;
      verts[i++] = 0.0f;
    }
  }

  _vbo->unmap();
}

void glQuadRenderableVA::fill_cvbo()
{
  float* verts = (float*)_cvbo->mapForWrite();

  size_t i = 0;
  for(float y = 0.0; y < _ydim; y++) {
    for(float x = 0.0; x < _xdim; x++) {
      verts[i++] = x;
      verts[i++] = y + 1.0f;
      verts[i++] = 0.0f;

      verts[i++] = x;
      verts[i++] = y;
      verts[i++] = 0.0f;

      verts[i++] = x + 1.0f;
      verts[i++] = y;
      verts[i++] = 0.0f;
      
      verts[i++] = x + 1.0f;
      verts[i++] = y + 1.0f;
      verts[i++] = 0.0f;
    }
  }
  
  _cvbo->unmap();
}

void glQuadRenderableVA::fill_edge_indices()
{
  size_t i = 0;
  for(unsigned y = 0; y <= _ydim; ++y) {
    for(unsigned x = 0; x < _xdim; ++x) {
      _edgeIndices[i++] = y * _xdim1 + x;
      _edgeIndices[i++] = y * _xdim1 + x + 1;
    }
  }

  for(unsigned x = 0; x <= _xdim; ++x) {
    for(unsigned y = 0; y < _ydim; ++y) {
      _edgeIndices[i++] = y * _xdim1 + x;
      _edgeIndices[i++] = (y + 1) * _xdim1 + x;
    }
  }
}

GLuint glQuadRenderableVA::get_buffer_object_id()
{
  return _pbo->id();
}


float4* glQuadRenderableVA::map_colors()
{
  _drawCells = true;
  return _cellColors;
}

float4* glQuadRenderableVA::map_vertex_colors()
{
  _drawVertices = true;
  return (float4*)_vpbo->mapForWrite();
}

float4* glQuadRenderableVA::map_edge_colors()
{
  _drawEdges = true;
  return _edgeColors;
}

void glQuadRenderableVA::unmap_colors()
{
  if(!_drawCells){
    return;
  }
  
  float4* pb = (float4*)_pbo->mapForWrite();
  
  size_t j = 0;
  for(size_t i = 0; i < _numCells; ++i){
    pb[j++] = _cellColors[i];
    pb[j++] = _cellColors[i];
    pb[j++] = _cellColors[i];
    pb[j++] = _cellColors[i];
  }
  
  _pbo->unmap();
}

void glQuadRenderableVA::unmap_vertex_colors()
{
  if(!_drawVertices){
    return;
  }
  
  _vpbo->unmap();
}

void glQuadRenderableVA::unmap_edge_colors()
{
  if(!_drawEdges){
    return;
  }
  
  float4* eb = (float4*)_epbo->mapForWrite();
  
  size_t j = 0;
  for(size_t i = 0; i < _numEdges; ++i){
    eb[j] = _edgeColors[i];
    eb[++j] = _edgeColors[i];
  }
  
  _epbo->unmap();
}

void glQuadRenderableVA::draw(glCamera* camera)
{
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  
  float s = 80.0/max(_xdim, _ydim);

  _cvbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);
  _cvbo->release();
  
  _pbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0);
  _pbo->release();
  
  glDrawArrays(GL_QUADS, 0, _numCells * 4);
  
  //glDrawElements(GL_QUADS, _numCells * 4, GL_UNSIGNED_INT, 0);
  
  /*
  _epbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0);
  _epbo->release();
  
  glDrawElements(GL_LINES, numEdges*2, GL_UNSIGNED_INT, _edgeIndices);
  
  _vpbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0);
  _vpbo->release();
  
  glPointSize(s);  
  glDrawArrays(GL_POINTS, 0, _numVertices);
  */
   
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  
  oglErrorCheck();
}
