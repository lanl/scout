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
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/renderall/RenderallSurface.h"
#include "scout/Runtime/RenderTarget.h"

// scout includes
#include "scout/Config/defs.h"

#ifdef SC_ENABLE_CUDA
#include <cuda.h>
#include <cudaGL.h>
#include "scout/Runtime/cuda/CudaDevice.h"
#endif

#ifdef SC_ENABLE_OPENCL
#include "scout/Runtime/opencl/scout_opencl.h"
#endif

#ifdef SC_ENABLE_CUDA
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#endif

using namespace std;
using namespace scout;

// ------  LLVM - globals accessed by LLVM / CUDA driver


// CUDA and OPENCL parts not done yet, just borrowed code from RenderallUniform

#ifdef SC_ENABLE_OPENCL
cl_mem __scrt_renderall_surface_opencl_device;
#endif

// -------------


RenderallSurface::RenderallSurface(size_t width, size_t height, size_t depth,
    float* vertices, float* normals, float* colors, int numVertices, 
    RenderTarget* renderTarget, glCamera* camera) 
:RenderallBase(width, height, depth), vertices_(vertices),
  normals_(normals), colors_(colors), numVertices_(numVertices), 
  renderTarget_(renderTarget), camera_(camera)
{
  renderTarget_->makeContextCurrent();



  localCamera_ = false;

  // we need a camera or nothing will happen! 
  if (camera_ ==  NULL)
  {
    cerr << "Warning: creating default camera" << endl;

    camera_ = new glCamera();
    localCamera_ = true;

    camera_->near = 70.0;
    camera_->far = 500.0;
    camera_->fov  = 40.0;
    const glfloat3 pos = glfloat3(350.0, -100.0, 650.0);
    const glfloat3 lookat = glfloat3(350.0, 200.0, 25.0);
    const glfloat3 up = glfloat3(-1.0, 0.0, 0.0);

    camera_->setPosition(pos);
    camera_->setLookAt(lookat);
    camera_->setUp(up);
    camera_->resize(renderTarget_->width(), renderTarget_->height());

  }

  renderable_ = new glSurfaceRenderable(width, height, depth, vertices_, normals_,
      colors, numVertices_, camera_);

  // show empty buffer
  renderTarget_->swapBuffers();
}

RenderallSurface::~RenderallSurface(){
  if (renderable_ != NULL) delete renderable_;
  if (localCamera_ && (camera_ != NULL)) delete camera_;
}

void RenderallSurface::begin(){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void RenderallSurface::end(){

  exec();

  // show what we just drew
  renderTarget_->swapBuffers();

}

void RenderallSurface::exec(){
  renderable_->draw(camera_);
}


extern "C" void __scrt_renderall_surface_begin(size_t width, size_t height, size_t depth,
    float* vertices, float* normals, float* colors, size_t numVertices, 
    void* renderTarget, glCamera* camera){

  __scrt_renderall = new RenderallSurface(width, height, depth, vertices, normals,
      (float*)colors, numVertices, (RenderTarget*) renderTarget, camera);

}

