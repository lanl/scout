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

#include "scout/types.h"
#include "scout/Runtime/opengl/opengl.h"
#include "scout/Runtime/renderall/RenderallGlyph.h"
#include "scout/Runtime/opengl/glGlyphRenderable.h"

namespace scout 
{

  using namespace std;

  RenderallGlyph::RenderallGlyph(size_t width, size_t height, size_t depth,
      size_t npoints, RenderTarget* renderTarget, glCamera* camera)
    : RenderallBase(width, height, depth), renderTarget_(renderTarget), camera_(camera)
  {

    renderTarget_->makeContextCurrent();

    renderable_ = new glGlyphRenderable(npoints);

    registerBuffer();

    // we need a camera or nothing will happen! 
    if (camera ==  NULL) 
    {
        cerr << "Warning: no camera so can't view anything!" << endl;
    }

    renderable_->initialize(camera);

    // show empty buffer
    renderTarget_->swapBuffers();
  }


  RenderallGlyph::~RenderallGlyph()
  {
    delete renderable_;
  }


  void RenderallGlyph::begin()
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mapGpuResources();
  }


  void RenderallGlyph::end()
  {
    unmapGpuResources();
    exec();

    // show what we just drew
    renderTarget_->swapBuffers();
  }

  void RenderallGlyph::exec()
  {
    if (camera_) {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(camera_->fov, camera_->aspect, camera_->near, camera_->far);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      gluLookAt(camera_->position.x, camera_->position.y, camera_->position.z,
          camera_->look_at.x, camera_->look_at.y, camera_->look_at.z,
          camera_->up[0], camera_->up[1], camera_->up[2]);
    }

    renderable_->draw(camera_);
  }
}

extern "C" void __scrt_renderall_glyph_init(size_t width, size_t height, size_t depth,
    size_t npoints, void* renderTarget, glCamera* camera)
{
  if(!__scrt_renderall){
    __scrt_renderall = new RenderallGlyph(width, height, depth, npoints, (RenderTarget*)renderTarget, camera);
  }
}
