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

#include "runtime/opengl/glyph_renderall.h"
#include "runtime/types.h"
#include "runtime/opengl/glProgram.h"
#include "runtime/opengl/glVertexShader.h"
#include "runtime/opengl/glFragmentShader.h"
#include "runtime/opengl/glAttributeBuffer.h"

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
    glAttributeBuffer*   abo;    // attribute buffer 

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

	if (info->prog->link() == false) {
		cerr << "scout: internal runtime error -- failed to link -- " << info->prog->linkLog() << endl;
	}


	info->near = 1.0;
	info->far = 3000.0;
	info->win_width = 1000.0;
	info->prog->bindUniformValue("windowWidth", &info->win_width);
	info->prog->bindUniformValue("near", &info->near);
	info->prog->bindUniformValue("far", &info->far);
	glBindAttribLocation(info->prog->id(), 1, "radius");

  info->abo = new glAttributeBuffer;
  info->abo->bind();
  info->abo->alloc(sizeof(glyph_vertex) * nglyphs, GL_STREAM_DRAW_ARB);
  info->abo->release();

  OpenGLErrorCheck();

  return info;
}


// ----- __sc_destroy_glyph_renderall
//
void __sc_destroy_glyph_renderall(glyph_renderall_t* info)
{
  delete info->prog;
  delete info->fshader;
  delete info->abo;
}


// ----- __sc_map_glyph_attributes
//
glyph_vertex* __sc_map_glyph_attributes(glyph_renderall_t* info)
{
  if (info)
    return (glyph_vertex*)info->abo->mapForWrite();
  else
    return 0;
}


// ----- __sc_unmap_glyph_attributes
//
void __sc_unmap_glyph_attributes(glyph_renderall_t* info)
{
  if (info)
    info->abo->unmap();
}


#define BUFFER_OFFSET(i) ((char *)NULL + (i))

// ----- __sc_exec_glyph_renderall
//
void __sc_exec_glyph_renderall(glyph_renderall_t* info)
{

  info->prog->enable();
 
  glEnable(GL_POINT_SPRITE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

  info->abo->bind();


	// vertices
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex), 
		BUFFER_OFFSET(0));

	// radiuses
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex), 
		BUFFER_OFFSET(sizeof(float) * 3));

	// colors
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof (glyph_vertex), 
		BUFFER_OFFSET(sizeof(float) * 4));

//	glPointSize(30.0); // for debugging
  glDrawArrays(GL_POINTS, 0, info->npoints);

  info->abo->release();
  info->prog->disable();
}

}
