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
#include <stdlib.h>

#include "scout/Runtime/types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/renderall/glyph_renderall.h"
#include "scout/Runtime/opengl/glGlyphRenderable.h"

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

namespace scout 
{

  using namespace std;

  glyph_renderall::glyph_renderall(size_t width, size_t height, size_t depth, 
      size_t npoints, glCamera* camera)
    : renderall_base_rt(width, height, depth), _camera(camera)
  {
      if(!__sc_glsdl){
        __sc_init_sdl(__sc_initial_width, __sc_initial_height, camera);
      }

    _renderable = new glGlyphRenderable(npoints);

    register_buffer();

    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
        cerr << "Warning: no camera so can't view anything!" << endl;
    }

    _renderable->initialize(camera);

    // show empty buffer
    __sc_glsdl->swapBuffers();
  }


  glyph_renderall::~glyph_renderall()
  {
    delete _renderable;
  }


  void glyph_renderall::begin()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    map_gpu_resources();
  }


  void glyph_renderall::end()
  {
    unmap_gpu_resources();
    exec();

    // show what we just drew
    __sc_glsdl->swapBuffers();

    bool done = __sc_glsdl->processEvent();

    // fix this
    if (done) exit(0);
  }

  void glyph_renderall::exec() 
  {
    __sc_glsdl->update();
    _renderable->draw(_camera);
  }
}

void __sc_init_glyph_renderall(size_t width, size_t height, size_t depth, 
    size_t npoints, glCamera* camera)
{
  if(!__sc_renderall){
    __sc_renderall = new glyph_renderall(width, height, depth, npoints, camera);
  }
}
