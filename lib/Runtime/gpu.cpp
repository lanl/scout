/*
 *  
 *###########################################################################
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
 */ 

#include "scout/Runtime/gpu.h"

using namespace std;
using namespace scout;

#include <cassert>

//#define USE_OPENCL
//#define USE_CUDA

#ifdef USE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elifdef USE_CUDA
#include <cuda.h>
#endif // USE_OPENCL

static const size_t MAX_DEVICES = 8;

namespace scout{

class gpu_rt_{
public:
  gpu_rt_(gpu_rt* o)
  : o_(o){

#ifdef USE_OPENCL
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    
    cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    assert(errNum == CL_SUCCESS);

    
    cl_device_id devices[MAX_DEVICES];
    cl_uint numDevices;

    errNum = clGetDeviceIDs(firstPlatformId, 
          CL_DEVICE_TYPE_GPU,
          8,
          devices,
          &numDevices);

    assert(errNum == CL_SUCCESS);

    numDevices_ = numDevices;

    for(size_t i = 0; i < numDevices_; ++i){
      cl_uint maxClockFrequency;
      clGetDeviceInfo(devices[i],
          CL_DEVICE_MAX_CLOCK_FREQUENCY,
          sizeof(cl_uint),
          &maxClockFrequency,
          0);

      deviceInfo_[i].clockRate = maxClockFrequency;

      cl_ulong memory;
      clGetDeviceInfo(devices[i],
      CL_DEVICE_GLOBAL_MEM_SIZE,
      sizeof(cl_ulong),
      &memory,
      0);

      deviceInfo_[i].memory = memory;

      cl_uint numMultiProcessors;
      clGetDeviceInfo(devices[i],
          CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint),
          &numMultiProcessors,
          0);

      deviceInfo_[i].numMultiProcessors = numMultiProcessors;
      
      char vendor[1024];
      clGetDeviceInfo(devices[i],
          CL_DEVICE_VENDOR,
          sizeof(char)*1024,
          vendor,
          0);

      deviceInfo_[i].vendor = vendor;
    }

#elifdef USE_CUDA
    cuInit(0);
    
    int numDevices;
    assert(cuDeviceGetCount(&numDevices) == CUDA_SUCCESS);
    numDevices_ = numDevices;
    
    for(size_t i = 0; i < numDevices_; ++i){
      CUdevprop dp;
      assert(cuDeviceGetProperties(&dp, i) == CUDA_SUCCESS);

      deviceInfo_[i].clockRate = dp.clockRate / 1000;

      size_t memory;
      assert(cuDeviceTotalMem(&memory, i) == CUDA_SUCCESS);

      deviceInfo_[i].memory = memory;

      int numMultiProcessors;
      cuDeviceGetAttribute(&numMultiProcessors, 
         CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, i);

      deviceInfo_[i].numMultiProcessors = numMultiProcessors;

      deviceInfo_[i].vendor = "NVIDIA";
    }
#endif
  }

  ~gpu_rt_(){

  }

  size_t numDevices() const{
    return numDevices_;
  }

  size_t clockRate(size_t device) const{
    assert(device < numDevices_);
    
    return deviceInfo_[device].clockRate;
  }

  size_t memory(size_t device) const{
    assert(device < numDevices_);
    
    return deviceInfo_[device].memory;
  }

  size_t numMultiProcessors(size_t device) const{
    assert(device < numDevices_);
    
    return deviceInfo_[device].numMultiProcessors;
  }

  const string& vendor(size_t device) const{
    assert(device < numDevices_);
    
    return deviceInfo_[device].vendor;
  }

private:
  struct DeviceInfo_{
    size_t clockRate;
    size_t memory;
    size_t numMultiProcessors;
    string vendor;
  };

  gpu_rt* o_;
  size_t numDevices_;
  DeviceInfo_ deviceInfo_[MAX_DEVICES];
};

} // end namespace scout

gpu_rt::gpu_rt(){
  x_ = new gpu_rt_(this);
}

gpu_rt::~gpu_rt(){
  delete x_;
}

size_t gpu_rt::numDevices() const{
  return x_->numDevices();
}

size_t gpu_rt::clockRate(size_t device) const{
  return x_->clockRate(device);
}

size_t gpu_rt::memory(size_t device) const{
  return x_->memory(device);
}

size_t gpu_rt::numMultiProcessors(size_t device) const{
  return x_->numMultiProcessors(device);
}

const std::string& gpu_rt::vendor(size_t device) const{
  return x_->vendor(device);
}
