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
#include "scout/Runtime/renderall/renderall_uniform.h"
#include "scout/Runtime/renderall/renderall_uniform_.h"
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

// scout includes
#include "scout/Config/defs.h"

using namespace std;
using namespace scout;

// ------  LLVM - globals accessed by LLVM / CUDA driver

float4* __sc_renderall_uniform_colors;

// -------------

extern glSDL* __sc_glsdl;
extern size_t __sc_initial_width;
extern size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL);

namespace scout{

  renderall_uniform_rt_::renderall_uniform_rt_(renderall_uniform_rt* o)
  : o_(o){


    if(!__sc_glsdl){
      __sc_init_sdl(__sc_initial_width, __sc_initial_height);
    }

    init();
  }

  renderall_uniform_rt_::~renderall_uniform_rt_(){
    if (_renderable != NULL) delete _renderable;
  }

  void renderall_uniform_rt_::init(){
    _renderable = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0),
        glfloat3(o_->width(), o_->height(), 0.0));

    register_pbo(_renderable->get_buffer_object_id());

    _renderable->initialize(NULL);

    // show empty buffer
    __sc_glsdl->swapBuffers();
  }

  void renderall_uniform_rt_::begin(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    map_gpu_resources();
  }

  void renderall_uniform_rt_::end(){
    unmap_gpu_resources();

    exec();

    // show what we just drew
    __sc_glsdl->swapBuffers();

    bool done = __sc_glsdl->processEvent();

    if (done) exit(0);

  }
  void renderall_uniform_rt_::exec(){
    _renderable->draw(NULL);
  }

} // end namespace scout

renderall_uniform_rt::renderall_uniform_rt(size_t width,
    size_t height,
    size_t depth)
: renderall_base_rt(width, height, depth){

  x_ = new renderall_uniform_rt_(this);

}

renderall_uniform_rt::~renderall_uniform_rt(){
  delete x_;
}

void renderall_uniform_rt::begin(){
  x_->begin();
}

void renderall_uniform_rt::end(){
  x_->end();
}

void __sc_begin_uniform_renderall(size_t width,
    size_t height,
    size_t depth){
  if(!__sc_renderall){
    __sc_renderall = new renderall_uniform_rt(width, height, depth);
  }

  __sc_renderall->begin();

}
