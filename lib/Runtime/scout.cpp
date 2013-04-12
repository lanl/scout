#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

#include "scout/Config/defs.h"
#include "scout/Runtime/Device.h"
#include "scout/Runtime/DeviceList.h"
#include "scout/Runtime/opengl/opengl.h"
#include "scout/Runtime/opengl/glyph_vertex.h"
#include "scout/Runtime/opengl/glSDL.h"
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

using namespace std;
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


void __scrt_init(ScoutDeviceType devType){
  DeviceList *devicelist = DeviceList::Instance();
  switch(devType){
    case ScoutGPUCUDA:
    {
#ifdef SC_ENABLE_CUDA
      glSDL *glsdl = glSDL::Instance();
      cuda::scInitialize(*devicelist);
#else
      cerr << "Error: Attempt to use CUDA GPU mode when Scout was "
        "compiled without CUDA." << endl;
      exit(1);
#endif // SC_ENABLE_CUDA
      break;
    }
    case ScoutGPUOpenCL:
    {
#ifdef SC_ENABLE_OPENCL
      __sc_init_opencl();
#else
      cerr << "Error: Attempt to use OpenCL GPU mode when Scout was "
        "compiled without OpenCL." << endl;
      exit(1);
#endif // SC_ENABLE_OPENCL
      break;
    }
    case ScoutGPUNone:
    {
        cpu::scInitialize(*devicelist);
    }
  }
}


void __scrt_end(){
}

double cshift(double a, int dx, int axis){
  return 0.0;
}

float cshift(float a, int dx, int axis){
  return 0.0;
}

int cshift(int a, int dx, int axis){
  return 0.0;
}
