#ifndef SCOUT_SCH_
#define SCOUT_SCH_

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/window.h"
#include "runtime/image.h"
#include "runtime/tbq.h"

extern scout::uniform_renderall_t* _uniform_renderall;
extern scout::tbq_rt* _tbq;
extern float4* _pixels;

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
float4 hsva(float h, float s, float v, float a)
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

float4 hsv(float h, float s, float v)
{
  return hsva(h, s, v, 1.0f);
}

void scoutInit(int& argc, char** argv, bool gpu);

void scoutInit(bool gpu);

void scoutEnd();

void scoutBeginRenderAll(size_t dx, size_t dy, size_t dz);

void scoutEndRenderAll();

void scoutBeginRenderAllElements(size_t dx, size_t dy, size_t dz);

void scoutEndRenderAllElements();

#endif // SCOUT_SCH_
