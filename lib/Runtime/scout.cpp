#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

#include "scout/Config/defs.h"

#ifdef SC_ENABLE_OPENGL
#include "scout/Runtime/opengl/opengl.h"
#endif

#ifdef SC_ENABLE_CUDA
#include "scout/Runtime/cuda/CudaInitialization.h"
#include "scout/Runtime/cuda/CudaDevice.h"
#include "scout/Runtime/cuda/CudaUtilities.h"
#endif // SC_ENABLE_CUDA

#ifdef SC_ENABLE_OPENCL
#include "scout/Runtime/opencl/scout_opencl.h"
#endif // SC_ENABLE_OPENCL
#include "scout/Runtime/gpu.h"
#include "scout/Runtime/cpu/CpuInitialization.h"
#include "scout/Runtime/init_mac.h"
#include "scout/Runtime/opengl/glSDL.h"
#include "scout/Runtime/DeviceList.h"


using namespace std;
using namespace scout;

DeviceList DevList;
glSDL* __sc_glsdl = 0;
size_t __sc_initial_width = 768;
size_t __sc_initial_height = 768;

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

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL){

  if (__sc_glsdl) {
    __sc_glsdl->resize(width, height);
  } else {
    __sc_glsdl = new glSDL(width, height, camera);
  }
}

void __sc_init(int argc, char** argv, ScoutGPUType gpuType){
  switch(gpuType){
    case ScoutGPUCUDA:
    {
#ifdef SC_ENABLE_CUDA
      __sc_init_sdl(__sc_initial_width, __sc_initial_height);
      if (cuda::scInitialize(DevList)) __sc_cuda = true;
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
        cpu::scInitialize(DevList);
    }
  }
}

void __sc_init(ScoutGPUType gpuType){
  __sc_init(0, 0, gpuType);
}

void __sc_end(){
  // Destroy all devices.
  DeviceList::iterator it = DevList.begin();
  while(it != DevList.end()) {
    delete *it;
    ++it;
  }
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
