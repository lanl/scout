#ifndef SCOUT_SCH
#define SCOUT_SCH

//#include "runtime/framebuffer.h"
#include "runtime/opengl/uniform_renderall.h"
#include "runtime/window.h"
#include "runtime/image.h"
#include "runtime/tbq.h"

extern scout::uniform_renderall_t* _uniform_renderall;
extern scout::tbq_rt* _tbq;
extern float4* _pixels;

double cshift(double a, int dx, int axis);
float cshift(float a, int dx, int axis);
int cshift(int a, int dx, int axis);
float4 hsv(float h, float s, float v);
void scoutInit(int argc, char** argv);
float4* scoutBeginRenderAll();
void scoutEndRenderAll();


#endif // SCOUT_SCH
