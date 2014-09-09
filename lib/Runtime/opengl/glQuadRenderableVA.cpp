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
  _drawCells = false;
  _drawVertices = false;
  _drawEdges = false;
  
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

  _cellColors = (float4*)malloc(sizeof(float4) * _numCells);
  
  _pbo = new glColorBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * _numCells * 4, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _cvbo = new glVertexBuffer;
  _cvbo->bind();
  _cvbo->alloc(sizeof(float) * 3 * _numCells * 4, GL_STREAM_DRAW_ARB);
  fill_cvbo();
  _cvbo->release();
  
  

  
  _edgeColors = (float4*)malloc(sizeof(float4) * _numEdges);
  
  _epbo = new glColorBuffer;
  _epbo->bind();
  _epbo->alloc(sizeof(float) * 4 * _numEdges * 2, GL_STREAM_DRAW_ARB);
  _epbo->release();
  
  _evbo = new glVertexBuffer;
  _evbo->bind();
  _evbo->alloc(sizeof(float) * 3 * _numEdges * 2, GL_STREAM_DRAW_ARB);
  fill_evbo();
  _evbo->release();

  
  
  
  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * _numVertices, GL_STREAM_DRAW_ARB);
  fill_vbo();
  _vbo->release();

  _vpbo = new glColorBuffer;
  _vpbo->bind();
  _vpbo->alloc(sizeof(float) * 4 * _numVertices, GL_STREAM_DRAW_ARB);
  _vpbo->release();
  
  oglErrorCheck();
}


void glQuadRenderableVA::destroy()
{
  if (_vbo) delete _vbo;
  if (_cvbo) delete _cvbo;
  if (_evbo) delete _evbo;
  if (_pbo) delete _pbo;
  if (_vpbo) delete _vpbo;
  if (_epbo) delete _epbo;
  if (_cellColors) free(_cellColors);
  if (_edgeColors) free(_edgeColors);

  _vbo = NULL;
  _cvbo = NULL;
  _evbo = NULL;
  _pbo = NULL;
  _vpbo = NULL;
  _epbo = NULL;
  _edgeColors = NULL;
  _cellColors = NULL;
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

void glQuadRenderableVA::fill_evbo()
{
  float* edges = (float*)_evbo->mapForWrite();
  
  size_t i = 0;
  for(float y = 0; y <= _ydim; ++y) {
    for(float x = 0; x < _xdim; ++x) {
      edges[i++] = x;
      edges[i++] = y;
      edges[i++] = 0.0f;
      
      edges[i++] = x + 1.0f;
      edges[i++] = y;
      edges[i++] = 0.0f;
    }
  }

  for(float x = 0; x <= _xdim; ++x) {
    for(float y = 0; y < _ydim; ++y) {
      edges[i++] = x;
      edges[i++] = y;
      edges[i++] = 0.0f;
      
      edges[i++] = x;
      edges[i++] = y + 1.0f;
      edges[i++] = 0.0f;
    }
  }
  
  _evbo->unmap();
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
    eb[j++] = _edgeColors[i];
    eb[j++] = _edgeColors[i];
  }
  
  _epbo->unmap();
}

void glQuadRenderableVA::draw(glCamera* camera)
{
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  
  float s = 80.0/max(_xdim, _ydim);

  if(_drawCells){
    _cvbo->bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    _cvbo->release();
    
    _pbo->bind();
    glColorPointer(4, GL_FLOAT, 0, 0);
    _pbo->release();
    
    glDrawArrays(GL_QUADS, 0, _numCells * 4);
  }
 
  if(_drawEdges){
    _evbo->bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    _evbo->release();
    
    _epbo->bind();
    glColorPointer(4, GL_FLOAT, 0, 0);
    _epbo->release();
    
    glLineWidth(s);
    glDrawArrays(GL_LINES, 0, _numEdges * 2);
  }

  if(_drawVertices){
    _vbo->bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    _vbo->release();
    
    _vpbo->bind();
    glColorPointer(4, GL_FLOAT, 0, 0);
    _vpbo->release();
    
    glPointSize(s/2.0f);
    glDrawArrays(GL_POINTS, 0, _numVertices);
  }
  
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  
  oglErrorCheck();
}
