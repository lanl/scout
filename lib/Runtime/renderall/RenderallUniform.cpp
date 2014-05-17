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
#include "scout/Runtime/renderall/RenderallUniform.h"
#include "scout/Runtime/renderall/RenderallUniformImpl.h"
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

// scout includes
#include "scout/Config/defs.h"

using namespace std;
using namespace scout;

namespace scout{

  RenderallUniformImpl::RenderallUniformImpl(RenderallUniform* rendUnif, RenderTarget* renderTarget)
  : rendUnif_(rendUnif), renderTarget_(renderTarget){

    init();
  }

  RenderallUniformImpl::~RenderallUniformImpl(){
    if (renderable_ != NULL) delete renderable_;
  }

  void RenderallUniformImpl::init(){

    renderTarget_->makeContextCurrent();

    renderable_ = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0),
        glfloat3(rendUnif_->width(), rendUnif_->height(), 0.0));

    registerPbo(renderable_->get_buffer_object_id());

    renderable_->initialize(NULL);

    // show empty buffer
    renderTarget_->swapBuffers();
  }

  void RenderallUniformImpl::begin(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mapGpuResources();
  }

  void RenderallUniformImpl::end(){
    unmapGpuResources();

    exec();

    // show what we just drew
    renderTarget_->swapBuffers();
    
  }
  void RenderallUniformImpl::exec(){
    renderable_->draw(NULL);
  }

} // end namespace scout

RenderallUniform::RenderallUniform(size_t width,
    size_t height,
    size_t depth,
    RenderTarget* renderTarget)
: RenderallBase(width, height, depth){

  x_ = new RenderallUniformImpl(this, renderTarget);

}

RenderallUniform::~RenderallUniform(){
  delete x_;
}

void RenderallUniform::begin(){
  x_->begin();
}

void RenderallUniform::end(){
  x_->end();
}

extern "C" void __scrt_renderall_uniform_begin(size_t width,
    size_t height,
    size_t depth,
    void* renderTarget){
  //std::cout << "scrt_renderall_uniform_begin: width " << width << " height " << height << " depth " << depth << "\n";
  if(!__scrt_renderall){
    __scrt_renderall = new RenderallUniform(width, height, depth, (RenderTarget*) renderTarget);
  }

  __scrt_renderall->begin();

}

