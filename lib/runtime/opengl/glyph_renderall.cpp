/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */
#include <iostream>
#include <stdlib.h>

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/types.h"
#include "runtime/opengl/glProgram.h"
#include "runtime/opengl/glVertexShader.h"
#include "runtime/opengl/glFragmentShader.h"
#include "runtime/opengl/glColorBuffer.h"
#include "runtime/opengl/glVertexBuffer.h"

namespace scout 
{

  // NOTE: There are still some times when the CMake build system
  // refuses to generate the following headers correctly (grrr...).
  // The only workaround we've found to date is to make sure you
  // start from a clean build...
  #include "sphere_cast_vs.h"
  #include "sphere_cast_fs.h"

  struct glyph_renderall_t {
    glProgram*        prog;      // shader program. 
    glVertexShader*   vshader;   // vertex shader.
    glFragmentShader* fshader;   // fragment shader.   
    glColorBuffer*    cbo;       // color buffer for glyph colors
    glVertexBuffer*   vbo;       // vertex buffer for glyph locations (points)
    float   					radius;    // radius (all same initially)
		// should have a camera probably, but do this for now
		float   					win_width; // 
		float   					near;    // 
		float   					far;    // 
    unsigned int      npoints;   // number of vertices stored in the vbo. 
  };

using namespace std;

// ----- __sc_init_glyph_renderall
//
glyph_renderall_t* __sc_init_glyph_renderall(dim_t nglyphs)
{
  glyph_renderall_t* info = new glyph_renderall_t;
  info->npoints = nglyphs;

  info->prog = new glProgram;
  
  info->vshader = new glVertexShader;
  info->vshader->setSource(sphere_cast_vs);
  if (info->vshader->compile() == false) {
    cerr << "scout: internal runtime error -- failed to compile glyph vertex shader!\n";
    abort();
    // TODO: At some point we can probably guard our renderall
    // constructs with a check for a valid renderall type and
    // simply skip rendering vs. aborting the entire program.
    // For now, we'll slap ourselves around for letting back
    // code sneak into the runtime...
  }
  info->prog->attachShader(info->vshader);

  
  info->fshader = new glFragmentShader;
  info->fshader->setSource(sphere_cast_fs);
  if (info->fshader->compile() == false) {
    cerr << "scout: internal runtime error -- failed to compile glyph fragment shader!\n";
    abort();
    // TODO: At some point we can probably guard our renderall
    // constructs with a check for a valid renderall type and
    // simply skip rendering vs. aborting the entire program.
    // For now, we'll slap ourselves around for letting back
    // code sneak into the runtime...    
  }

  info->prog->attachShader(info->fshader);

	info->radius = .25;
	info->near = 1.0;
	info->far = 20.0;
	info->win_width = 20.0;
	info->prog->bindUniformValue("radius", &info->radius);
	info->prog->bindUniformValue("windowWidth", &info->win_width);
	info->prog->bindUniformValue("near", &info->near);
	info->prog->bindUniformValue("far", &info->far);
	

  info->cbo = new glColorBuffer;
  info->cbo->bind();
  info->cbo->alloc(sizeof(float) * 4 * nglyphs, GL_STREAM_DRAW_ARB);
  info->cbo->release();

  info->vbo = new glVertexBuffer;
  info->vbo->bind();
  info->vbo->alloc(sizeof(float) * 3 * nglyphs, GL_STREAM_DRAW_ARB);
  info->vbo->release();

  OpenGLErrorCheck();

  return info;
}


// ----- __sc_destroy_glyph_renderall
//
void __sc_destroy_glyph_renderall(glyph_renderall_t* info)
{
  delete info->prog;
  delete info->fshader;
  delete info->prog;
  delete info->cbo;  
  delete info->vbo;
}


// ----- __sc_map_glyph_colors
//
float4* __sc_map_glyph_colors(glyph_renderall_t* info)
{
  if (info)
    return (float4*)info->cbo->mapForWrite();
  else
    return 0;
}


// ----- __sc_unmap_glyph_colors
//
void __sc_unmap_glyph_colors(glyph_renderall_t* info)
{
  if (info)
    info->cbo->unmap();
}


// TODO: need to map/unmap glyph positions...
// positions are a float3 on the scout language side.

// ----- __sc_map_glyph_positions
//
float* __sc_map_glyph_positions(glyph_renderall_t* info)
{ 
	if (info)
		return (float*)info->vbo->mapForWrite();
	else
		return 0;
}

// ----- __sc_unmap_glyph_positions
//
void __sc_unmap_glyph_positions(glyph_renderall_t* info)
{
  if (info)
    info->vbo->unmap();
}


// TODO: need to map/unmap glyph radius...
// radius is a single float on the scout language side.


// ----- __sc_exec_glyph_renderall
//
void __sc_exec_glyph_renderall(glyph_renderall_t* info)
{
  info->prog->enable();
  
  glEnable(GL_POINT_SPRITE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

  glEnableClientState(GL_VERTEX_ARRAY);
  info->vbo->bind();
  glVertexPointer(3, GL_FLOAT, 0, 0);

  glEnableClientState(GL_COLOR_ARRAY);
  info->cbo->bind();
  glColorPointer(4, GL_FLOAT, 0, 0);

  glDrawArrays(GL_POINT, 0, info->npoints);

  glDisableClientState(GL_VERTEX_ARRAY);
  info->cbo->release();
  info->vbo->release();

  info->prog->disable();
}

}
