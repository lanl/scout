#ifndef SCOUT_SCH
#define SCOUT_SCH

#include "runtime/framebuffer.h"
#include "runtime/window.h"
#include "runtime/image.h"
#include "runtime/tbq.h"

extern scout::framebuffer_rt* _framebuffer;
extern scout::tbq_rt* _tbq;

double cshift(double a, int dx, int axis);
float cshift(float a, int dx, int axis);
int cshift(int a, int dx, int axis);
float4 hsv(float h, float s, float v);
void scoutInit(int argc, char** argv);
void scoutSwapBuffers();

#endif // SCOUT_SCH
