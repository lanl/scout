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
#include <cassert>
#include "scout/Runtime/base_types.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/opengl/glQuadRenderableVA.h"

#include <cuda.h>
#include <cudaGL.h>
#include "scout/Runtime/DeviceList.h"
#include "scout/Runtime/renderall/RenderallUniformImpl.h"

// global in lib/Runtime/CudaRuntime.cpp
extern CUgraphicsResource __sc_cuda_device_resource;

namespace scout{

  void RenderallUniformImpl::registerPbo(GLuint pbo) {
    DeviceList *devicelist = DeviceList::Instance();
    if(devicelist->hasCudaDevice()) {
      assert(cuGraphicsGLRegisterBuffer(&__sc_cuda_device_resource,
          pbo,
          CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD) == CUDA_SUCCESS);
    } else {
      //std::cout << "no cuda using SDL\n";
    }
  }

  void RenderallUniformImpl::mapGpuResources(void) {
    DeviceList *devicelist = DeviceList::Instance();
    if(devicelist->hasCudaDevice()) {
    // map one graphics resource for access by CUDA
      assert(cuGraphicsMapResources(1, &__sc_cuda_device_resource, 0) == CUDA_SUCCESS);

      size_t bytes;
             // return a pointer by which the mapped graphics resource may be accessed.
      assert(cuGraphicsResourceGetMappedPointer((CUdeviceptr *)&__scrt_renderall_uniform_cuda_device,
             &bytes, __sc_cuda_device_resource) == CUDA_SUCCESS);
    } else {
      __scrt_renderall_uniform_colors =renderable_->map_colors();
    }
  }

  void RenderallUniformImpl::unmapGpuResources(void) {
    DeviceList *devicelist = DeviceList::Instance();
    if(devicelist->hasCudaDevice()) {
      assert(cuGraphicsUnmapResources(1, &__sc_cuda_device_resource, 0) == CUDA_SUCCESS);
      renderable_->alloc_texture();
    } else {
      renderable_->unmap_colors();
    }
  }
}

