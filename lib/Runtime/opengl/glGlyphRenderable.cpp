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
#include <fstream>
#include <limits>
#include <stdlib.h>

#include "runtime/opengl/glGlyphRenderable.h"
#include "sphere_cast_vs.h"
#include "sphere_cast_fs.h"


using namespace scout;
using namespace std;


// ---- glGlyphRenderable
//
glGlyphRenderable::glGlyphRenderable(size_t npoints)
    : glRenderable(), _npoints(npoints)
{
}


// ---- glGlyphRenderable
//
glGlyphRenderable::~glGlyphRenderable()
{
  if (_abo != NULL) {
    delete _abo;
  }

}


// ---- initialize
//
void glGlyphRenderable::initialize(glCamera* camera)
{
  baseInitialize();

  _abo = new glAttributeBuffer;

  loadShaders(camera);

  allocateBuffer();
}


// ---- setData
//
void glGlyphRenderable::allocateBuffer()
{
  assert((_npoints > 0) && (_abo != NULL));
  _abo->bind();
  _abo->alloc(sizeof(glyph_vertex) * _npoints, GL_STREAM_DRAW_ARB);
  _abo->release();

  OpenGLErrorCheck();
}


// ---- loadShaders
//
void glGlyphRenderable::loadShaders(const glCamera* camera)
{

  // vertex shader

  glVertexShader *vshader = new glVertexShader;
  vshader->setSource(sphere_cast_vs);
  if (vshader->compile() == false) {
    cerr << "scout: internal runtime error -- failed to compile glyph vertex shader!\n";
    cerr << vshader->compileLog() << endl;
    abort();
    // TODO: At some point we can probably guard our renderall
    // constructs with a check for a valid renderall type and
    // simply skip rendering vs. aborting the entire program.
    // For now, we'll slap ourselves around for letting back
    // code sneak into the runtime...
  }
  attachShader(vshader);

  // fragment shader

  glFragmentShader *fshader = new glFragmentShader;
  fshader->setSource(sphere_cast_fs);
  if (fshader->compile() == false) {
    cerr << "scout: internal runtime error -- failed to compile glyph fragment shader!\n";
    cerr << fshader->compileLog() << endl;
    abort();
    // TODO: At some point we can probably guard our renderall
    // constructs with a check for a valid renderall type and
    // simply skip rendering vs. aborting the entire program.
    // For now, we'll slap ourselves around for letting back
    // code sneak into the runtime...    
  }
  attachShader(fshader);

  glBindAttribLocation(shader_prog->id(), 1, "radius");
  OpenGLErrorCheck();
  glBindAttribLocation(shader_prog->id(), 7, "color");
  OpenGLErrorCheck();

  if (shader_prog->link() == false) {
    cerr << "scout: internal runtime error -- failed to link -- " << shader_prog->linkLog() << endl;
  }

  shader_prog->bindUniformValue("windowWidth", &camera->win_width);
  shader_prog->bindUniformValue("near", &camera->near);
  shader_prog->bindUniformValue("far", &camera->far);

}


#define BUFFER_OFFSET(i) ((char *)NULL + (i))

// ---- draw
//
void glGlyphRenderable::draw(glCamera* camera)
{
  if (_npoints != 0) {


    shader_prog->enable();

    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

    glClearColor(0.5, 0.55, 0.65, 0.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _abo->bind();


    // vertices
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex),
        BUFFER_OFFSET(0));

    // radiuses
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex),
        BUFFER_OFFSET(sizeof(float) * 3));

    // colors
    glEnableVertexAttribArray(7);
    glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex),
        BUFFER_OFFSET(sizeof(float) * 4));

    //  glPointSize(30.0); // for debugging
    glDrawArrays(GL_POINTS, 0, _npoints);

    _abo->release();
    shader_prog->disable();
  }
}


