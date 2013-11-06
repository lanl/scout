/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
 */
#include <iostream>
#include <sstream>

using namespace std;

#include "scout/Config/defs.h"
#include "scout/Runtime/Device.h"
#include "scout/Runtime/DeviceList.h"
#include "scout/Runtime/opengl/opengl.h"
#include "scout/Runtime/opengl/glyph_vertex.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/types.h"
#include "scout/Runtime/gpu.h"
#include "scout/Runtime/cpu/CpuInitialization.h"
#include "scout/Runtime/init_mac.h"


#ifdef SC_ENABLE_CUDA
#include "scout/Runtime/cuda/CudaInitialization.h"
#include "scout/Runtime/cuda/CudaDevice.h"
#include "scout/Runtime/cuda/CudaUtilities.h"
#endif // SC_ENABLE_CUDA

#ifdef SC_ENABLE_OPENCL
#include "scout/Runtime/opencl/scout_opencl.h"
#endif // SC_ENABLE_OPENCL

using namespace scout;

//globals accessed by llvm/tools/clang/lib/CodeGen/CGStmt.cpp
scout::float4* __scrt_renderall_uniform_colors;
glyph_vertex* __scrt_renderall_glyph_vertex_data;
// -------------

// globals accessed by lib/Compiler/llvm/Transforms/Driver/CudaDriver.cpp
unsigned long long __scrt_renderall_glyph_cuda_device;
unsigned long long __scrt_renderall_surface_cuda_device;
unsigned long long __scrt_renderall_uniform_cuda_device;
unsigned long long __scrt_renderall_volume_cuda_device;
// -------------

#ifdef SC_ENABLE_MPI
#include <mpi.h>
MPI_Comm __volren_gcomm = MPI_COMM_WORLD;
#endif // SC_ENABLE_MPI

extern "C"
void __sc_debugger_dump_mesh_field(size_t width,
                                   size_t height,
                                   size_t depth,
                                   void* fp,
                                   size_t fieldType){

  if(height == 0){
    height = 1;
  }

  if(depth == 0){
    depth = 1;
  }

  size_t span = width * height * depth;

  switch(fieldType){
    case 0:
    {
      int8_t* f = static_cast<int8_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << int(f[i]);
      }

      break;
    }
    case 1:
    {
      uint8_t* f = static_cast<uint8_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << int(f[i]);
      }

      break;
    }
    case 2:
    {
      int16_t* f = static_cast<int16_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 3:
    {
      uint16_t* f = static_cast<uint16_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 4:
    {
      int32_t* f = static_cast<int32_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 5:
    {
      uint32_t* f = static_cast<uint32_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 6:
    {
      int64_t* f = static_cast<int64_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 7:
    {
      uint64_t* f = static_cast<uint64_t*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 8:
    {
      float* f = static_cast<float*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
    case 9:
    {
      double* f = static_cast<double*>(fp);

      for(size_t i = 0; i < span; ++i){
        if(i > 0){
          cerr << ", ";
        }
        cerr << i << ": " << f[i];
      }

      break;
    }
  }

  cerr << endl;
}

extern "C" void __scrt_init_cpu() {
  DeviceList *devicelist = DeviceList::Instance();
  cpu::scInitialize(*devicelist);
}

extern "C" void __scrt_init_cuda() {
#ifdef SC_ENABLE_CUDA
  DeviceList *devicelist = DeviceList::Instance();
  glSDL *glsdl = glSDL::Instance();
  cuda::scInitialize(*devicelist);
#else
  cerr << "Error: Attempt to use CUDA GPU mode when Scout was "
        "compiled without CUDA." << endl;
  exit(1);
#endif 
}

extern "C" void __scrt_init_opencl() {
#ifdef SC_ENABLE_OPENCL
  __sc_init_opencl();
#else
  cerr << "Error: Attempt to use OpenCL GPU mode when Scout was "
        "compiled without OpenCL." << endl;
  exit(1);
#endif // SC_ENABLE_OPENCL
}
#if 0
double cshift(double a, int dx, int axis){
  return 0.0;
}

float cshift(float a, int dx, int axis){
  return 0.0;
}

int cshift(int a, int dx, int axis){
  return 0.0;
}
#endif
