/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 

#include <iostream>
#include "scout/Config/defs.h"
#include "scout/Runtime/Device.h"
#include "scout/Runtime/opengl/vectors.h"
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/renderall/RenderallGlyph.h"
#include "scout/Runtime/opengl/glyph_vertex.h"
#include "scout/Runtime/opengl/glSDL.h"
using namespace std;
using namespace scout;

static const size_t WINDOW_WIDTH = 1000;
static const size_t WINDOW_HEIGHT = 1000;

extern glyph_vertex* __scrt_renderall_glyph_vertex_data;
extern glSDL* __sc_glsdl;

void Loop()
{
  // loop with event queue processing
  for(int i = 0 ; i <100; i++)
  {
    __scrt_renderall_begin();
    __scrt_renderall_end();
  }
}

int main(int argc, char *argv[])
{
  glCamera camera;
  camera.near = 1.0;
  camera.far = 4000.0;
  camera.fov  = 35.0;
  camera.focal_length = 70.0f;
  const glfloat3 pos = glfloat3(500.0, 500.0, 3000.0);
  const glfloat3 lookat = glfloat3(500.0, 500.0, 500.0);
  const glfloat3 up = glfloat3(0.0, 1.0, 0.0);
  camera.setPosition(pos);
  camera.setLookAt(lookat);
  camera.setUp(up);
  camera.resize(WINDOW_WIDTH, WINDOW_HEIGHT);

  size_t dim = 1000;
  __scrt_renderall_glyph_init(WINDOW_WIDTH, WINDOW_HEIGHT, 0, dim, &camera);
  //glsdl.setBackgroundColor(.5, .55, .65, 0.0);

  __scrt_renderall_begin();

  glyph_vertex* vertex_attribs  = __scrt_renderall_glyph_vertex_data;

  // Create points in order of lowest z first.
  int p = 0;  
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 10; k++) {
        vertex_attribs[p].x = (i*100 + 50);
        vertex_attribs[p].y = (j*100 + 50);
        vertex_attribs[p].z = (k*100 + 50);
        vertex_attribs[p].radius = (j+1)*2; // radius
        vertex_attribs[p].r = i/10.0;
        vertex_attribs[p].g = j/10.0;
        vertex_attribs[p].b = k/10.0;
        vertex_attribs[p].a = 1.0;
        p++;
      }
    }
  }

  __scrt_renderall_end();

  Loop();
  // destroy
  __scrt_renderall_delete();
  return 0;
}
