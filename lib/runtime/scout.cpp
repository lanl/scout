#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

#define SC_USE_PNG

#ifdef SC_ENABLE_CUDA
#include "runtime/cuda/scout_cuda.h"
#include "runtime/cuda/cuda.h"
#endif // SC_ENABLE_CUDA

#ifdef SC_ENABLE_OPENCL
#include "runtime/opencl/scout_opencl.h"
#endif // SC_ENABLE_OPENCL

#include "runtime/init_mac.h"
#include "runtime/opengl/glSDL.h"
#include "runtime/tbq.h"

using namespace std;
using namespace scout;

tbq_rt* __sc_tbq;
glSDL* __sc_glsdl = 0;

size_t __sc_initial_width = 768;
size_t __sc_initial_height = 768;

enum ScoutGPUType{
  ScoutGPUNone,
  ScoutGPUCUDA,
  ScoutGPUOpenCL
};

extern "C"
void __sc_queue_block(void* blockLiteral, int numDimensions, int numFields){
  __sc_tbq->run(blockLiteral, numDimensions, numFields);
}

extern "C"
void __sc_dump_mesh(void* mp){
  int32_t width = *(int32_t*)mp;
  int32_t height = *(int32_t*)((char*)mp + sizeof(int32_t));
  int32_t depth = *(int32_t*)((char*)mp + sizeof(int32_t)*2);
  
  // mesh starts at this i32*4 offset due to alignment
  float** mesh = (float**)((char*)mp + sizeof(int32_t)*4);

  cout << "-------- mesh dump" << endl;

  float* aStart = (float*)mesh[0];

  size_t len = width * height;

  for(size_t i = 0; i < len; ++i){
    if(i > 0){
      cout << ", ";
    }
    cout << i << ": " << aStart[i];
  }

  cout << endl << "-------- end mesh dump" << endl;
}

void __sc_init_sdl(size_t width, size_t height, glCamera* camera = NULL){

  if (__sc_glsdl) {
    __sc_glsdl->resize(width, height);
  } else {
    __sc_glsdl = new glSDL(width, height, camera);
  }
}

void __sc_init(int argc, char** argv, ScoutGPUType gpuType){
  __sc_tbq = new tbq_rt;

  switch(gpuType){
    case ScoutGPUCUDA:
    {
#ifdef SC_ENABLE_CUDA
      __sc_init_sdl(__sc_initial_width, __sc_initial_height);
      __sc_init_cuda();
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

    }
  }
}

void __sc_init(ScoutGPUType gpuType){
  __sc_init(0, 0, gpuType);
}

void __sc_end(){
  delete __sc_tbq;
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
