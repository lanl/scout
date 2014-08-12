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

  _ntexcoords = 2;
  _texture = new glTexture2D(xdim, ydim);
  _texture->addParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  _texture->addParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _texture->addParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);


  _pbo = new glTextureBuffer;
  _pbo->bind();
  _pbo->alloc(sizeof(float) * 4 * xdim * ydim, GL_STREAM_DRAW_ARB);
  _pbo->release();

  _vbo = new glVertexBuffer;
  _vbo->bind();
  _vbo->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);
  fill_vbo(_min_pt.x, _min_pt.y, _max_pt.x, _max_pt.y);
  _vbo->release();
  _nverts = 4;

#ifdef WITH_VERTICES_EDGES
  _mvbo = new glVertexBuffer;
  _mvbo->bind();
  _mvbo->alloc(sizeof(float) * 3 * 4, GL_STREAM_DRAW_ARB);
  fill_mvbo(_min_pt.x, _min_pt.y, _max_pt.x, _max_pt.y);
  _mvbo->release();

  _evbo = new glVertexBuffer;
  _evbo->bind();
  _evbo->alloc(sizeof(float) * 3 * 8, GL_STREAM_DRAW_ARB);
  fill_evbo(_min_pt.x, _min_pt.y, _max_pt.x, _max_pt.y);
  _evbo->release();
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
  if (_vbo != 0) delete _vbo;
  if (_tcbo != 0) delete _tcbo;
  _texture = NULL;
  _pbo = NULL;
  _vbo = NULL;
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

  size_t width = _max_pt.x - _min_pt.x;
  size_t height = _max_pt.y - _min_pt.y;

  static const float pad = 0.05;

  if(height == 0){
    float px = pad * width;
    gluOrtho2D(-px, width + px, -px, width + px);

  }
  else{
    if(width >= height){
      float px = pad * width;
      float py = (1 - float(height)/width) * width * 0.50;
      gluOrtho2D(-px, width + px, -py - px, width - py + px);
    }
    else{
      float py = pad * height;
      float px = (1 - float(width)/height) * height * 0.50;
      gluOrtho2D(-px - py, width + px + py, -py, height + py);
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

void glQuadRenderableVA::fill_mvbo(float x0,
    float y0,
    float x1,
    float y1)
{

  float* verts = (float*)_mvbo->mapForWrite();

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

  _mvbo->unmap();
}

void glQuadRenderableVA::fill_evbo(float x0,
    float y0,
    float x1,
    float y1)
{

  float* verts = (float*)_evbo->mapForWrite();

  verts[0] = x0;
  verts[1] = y0;
  verts[2] = 0.0f;

  verts[3] = x0;
  verts[4] = y1;
  verts[5] = 0.f;

  verts[6] = x0;
  verts[7] = y1;
  verts[8] = 0.f;

  verts[9] = x1;
  verts[10] = y1;
  verts[11] = 0.0f;

  verts[12] = x1;
  verts[13] = y1;
  verts[14] = 0.0f;

  verts[15] = x1;
  verts[16] = y0;
  verts[17] = 0.0f;

  verts[18] = x1;
  verts[19] = y0;
  verts[20] = 0.0f;

  verts[21] = x0;
  verts[22] = y0;
  verts[23] = 0.0f;

  _evbo->unmap();
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


GLuint glQuadRenderableVA::get_buffer_object_id()
{
  return _pbo->id();
}


float4* glQuadRenderableVA::map_colors()
{
  return (float4*)_pbo->mapForWrite();
}


void glQuadRenderableVA::unmap_colors()
{
  _pbo->unmap();
  _pbo->bind();
  _texture->initialize(0);
  _pbo->release();
}


void glQuadRenderableVA::draw(glCamera* camera)
{
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

#ifdef WITH_VERTICES_EDGES
  glEnableClientState(GL_VERTEX_ARRAY);
  _evbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);

  oglErrorCheck();

  glLineWidth(5.0);
  glColor4f(0.7, 0.7, 0.7, 0.0);
  glDrawArrays(GL_LINES, 0, 8);


  glEnableClientState(GL_VERTEX_ARRAY);
  _mvbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);

  oglErrorCheck();

  glPointSize(5.0);
  glColor4f(1.0, 1.0, 1.0, 0.0);
  glDrawArrays(GL_POINTS, 0, 4);

  glDisableClientState(GL_VERTEX_ARRAY);
  _mvbo->release();
#endif

  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}
