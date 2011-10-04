#include <iostream>
#include <sstream>

#define SC_USE_PNG

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/tbq.h"

using namespace std;
using namespace scout;

uniform_renderall_t* _uniform_renderall;
float4* _pixels;
tbq_rt* _tbq;

static const size_t RENDER_WIDTH = 1024;
static const size_t RENDER_HEIGHT = 1;

void scoutInit(int argc, char** argv){
  _uniform_renderall = __sc_init_uniform_renderall(RENDER_WIDTH);
  _tbq = new tbq_rt;
}

float4* scoutBeginRenderAll(){
  return __sc_map_uniform_colors(_uniform_renderall);
}

void scoutEndRenderAll(){
  __sc_unmap_uniform_colors(_uniform_renderall);
  __sc_destroy_uniform_renderall(_uniform_renderall);
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

float4 hsv(float h, float s, float v){
  float4 no_op;
  return no_op;
}
