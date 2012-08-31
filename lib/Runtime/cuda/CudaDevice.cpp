/*
 * ###########################################################################
 * Copyrigh (c) 2010, Los Alamos National Security, LLC.
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

#include "scout/Config/defs.h"
#include "scout/Runtime/cuda/Cuda.h"
#include "scout/Runtime/cuda/CudaDevice.h"

using namespace scout;

// ----- CudaDevice
/// 
CudaDevice::CudaDevice(int device_id)
{
  deviceID = device_id;
  
  if ((deviceStatus = cuDeviceGet(&cuDevice, deviceID)) != CUDA_SUCCESS) {
    cuError(deviceStatus);
  }
  
  char device_name[256];
  if ((deviceStatus = cuDeviceGetName(device_name, 256, cuDevice)) != CUDA_SUCCESS) {
    cuError(deviceStatus);
  } else {
    deviceName = std::string(device_name);
  }

  // ToDo -- CUDA 5 introduces some new flags for how host CPU threads
  // interact relative to the GPU kernels they submit.  At some point
  // we should consider what impact they have on our overall
  // scheduling...
  unsigned int flags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST;
  if ((deviceStatus = cuCtxCreate(&cuContext, flags, deviceID)) != CUDA_SUCCESS) {
    cuError(deviceStatus);
  }
}


// ----- ~CudaDevice
///
CudaDevice::~CudaDevice()
{
  CUresult error_id;    
  if ((error_id = cuCtxDestroy(cuContext)) != CUDA_SUCCESS) {
    cuError(error_id);
  }
}
