#ifndef SCOUT_SCH_
#define SCOUT_SCH_

#include "scout/Runtime/renderall_uniform.h"
#include "scout/Runtime/window.h"
#include "scout/Runtime/image.h"
#include "scout/Runtime/tbq.h"
#include "scout/Runtime/opengl/glCamera.h"

#ifdef SC_ENABLE_MPI
#include "scout/Runtime/volume_renderall.h"
#endif

extern scout::tbq_rt* __sc_tbq;
extern scout::glCamera* __sc_camera;

extern const size_t __sc_initial_width;
extern const size_t __sc_initial_height;

enum ScoutGPUType{
  ScoutGPUNone,
  ScoutGPUCUDA,
  ScoutGPUOpenCL
};

void __sc_init_sdl(size_t width, size_t height, scout::glCamera* cam = NULL);

void __sc_init(int argc, char** argv, ScoutGPUType gpuType);

void __sc_init(ScoutGPUType gpuType);

void __sc_end();

double cshift(double a, int dx);
double cshift(double a, int dx, int dy);
double cshift(double a, int dx, int dy, int dz);

float cshift(float a, int dx);
float cshift(float a, int dx, int dy);
float cshift(float a, int dx, int dy, int dz);

int cshift(int a, int dx);
int cshift(int a, int dx, int dy);
int cshift(int a, int dx, int dy, int dz);

// ---- hsva
//
static float4 hsva(float h, float s, float v, float a)
{
  float4 r;

  r.w = a;

  int i;
  float f, p, q, t;
  if (s == 0.0f) {
    r.x = v;
    r.y = v;
    r.z = v;
    return r;
  }

  h /= 60.0f;
  i = int(h);
  f = h - float(i);
  p = v * (1.0 - s);
  q = v * (1.0 - s * f);
  t = v * (1.0 - s * (1.0 - f));

  switch(i) {

    case 0:
      r.x = v;
      r.y = t;
      r.z = p;
      break;
      
    case 1:
      r.x = q;
      r.y = v;
      r.z = p;
      break;
      
    case 2:
      r.x = p;
      r.y = v;
      r.z = t;
      break;
      
    case 3:
      r.x = p;
      r.y = q;
      r.z = v;
      break;
      
    case 4:
      r.x = t;
      r.y = p;
      r.z = v;
      break;
      
    default:
      r.x = v;
      r.y = p;
      r.z = q;
      break;
  }

  return r;
}

static float4 hsv(float h, float s, float v){
  return hsva(h, s, v, 1.0f);
}

#endif // SCOUT_SCH_
