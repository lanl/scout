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

#include "runtime/opengl/glRenderable.h"
#include <iostream>

using namespace std;
using namespace scout;

// ----- glRenderable
//
glRenderable::glRenderable()
{
  _hide = false;
  shader_prog = NULL;
}


// ----- baseInitialize
//
void glRenderable::baseInitialize()
{
  shader_prog = new glProgram();
}


// ----- render
//
void glRenderable::render(glCamera* camera)
{
  draw(camera);
}

