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

#endif // SCOUT_SCH_
