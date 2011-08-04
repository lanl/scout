#ifndef SCOUT_H
#define SCOUT_H

#define SCOUT_TOP_LEVEL

#include "runtime/framebuffer.h"

extern scout::framebuffer_rt* _framebuffer;

double cshift(double a, int dx, int axis);
float cshift(float a, int dx, int axis);
int cshift(int a, int dx, int axis);
float4 hsv(float h, float s, float v);
void scoutInit(int argc, char** argv);
void scoutSwapBuffers();

#undef SCOUT_TOP_LEVEL

#endif // SCOUT_H
