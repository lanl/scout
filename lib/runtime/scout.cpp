#include <iostream>
#include <sstream>

#ifdef __APPLE__

#include <cmath>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include "runtime/init_mac.h"

#else

#include <GL/gl.h>
#include <GL/glu.h>

#endif

#include <SDL/SDL.h>

#define SC_USE_PNG

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/tbq.h"

using namespace std;
using namespace scout;

SDL_Surface* _sdl_surface;
uniform_renderall_t* _uniform_renderall = 0;
float4* _pixels;
tbq_rt* _tbq;

static const size_t WINDOW_WIDTH = 1024;
static const size_t WINDOW_HEIGHT = 1024;

void scoutInit(int& argc, char** argv){
  _tbq = new tbq_rt;
}

void scoutInit(){
  _tbq = new tbq_rt;
}

void scoutBeginRenderAll(size_t dx, size_t dy, size_t dz){
  // if this is our first render all, then initialize SDL and
  // the OpenGL runtime
  if(!_uniform_renderall){

#ifdef __APPLE__
    scoutInitMac();
#endif

    if(SDL_Init(SDL_INIT_VIDEO) < 0){
      cerr << "Error: failed to initialize SDL." << endl;
      exit(1);
    }

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
 
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
 
    SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE, 8);
 
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 2);

    _sdl_surface = SDL_SetVideoMode(WINDOW_WIDTH, WINDOW_HEIGHT, 32,
				    SDL_HWSURFACE | 
				    SDL_GL_DOUBLEBUFFER | 
				    SDL_OPENGL);

    if(!_sdl_surface){
      cerr << "Error: failed to initialize SDL surface." << endl;
      exit(1);
    }

    //glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glClearColor(0.5, 0.55, 0.65, 0.0);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
        
    if(dy == 0){
      gluOrtho2D(0, dx, 0, dx);
      _uniform_renderall = __sc_init_uniform_renderall(dx);
    }
    else{
      gluOrtho2D(0, dx, 0, dy);
      _uniform_renderall = __sc_init_uniform_renderall(dx, dy);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SDL_GL_SwapBuffers();
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  _pixels = __sc_map_uniform_colors(_uniform_renderall);
}

void scoutEndRenderAll(){
  __sc_unmap_uniform_colors(_uniform_renderall);
  __sc_exec_uniform_renderall(_uniform_renderall);
  SDL_GL_SwapBuffers();
}

void scoutEnd(){
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
  float4 r;

  /*
  r.components[0] = 0.0f;
  r.components[1] = 0.0f;
  r.components[2] = 1.0f;
  r.components[3] = 1.0f;
  
  return r;
  */

  r.components[3] = 1.0;

  int i;
  float f, p, q, t;
  if(s == 0){
    r.components[0] = r.components[1] = r.components[2] = v;
    return r;
  }

  h /= 60;
  i = floor(h);
  f = h - i;
  p = v * (1 - s);
  q = v * (1 - s * f);
  t = v * (1 - s * ( 1 - f ));

  switch(i) {
  case 0:
    r.components[0] = v;
    r.components[1] = t;
    r.components[2] = p;
    break;
  case 1:
    r.components[0] = q;
    r.components[1] = v;
    r.components[2] = p;
    break;
  case 2:
    r.components[0] = p;
    r.components[1] = v;
    r.components[2] = t;
    break;
  case 3:
    r.components[0] = p;
    r.components[1] = q;
    r.components[2] = v;
    break;
  case 4:
    r.components[0] = t;
    r.components[1] = p;
    r.components[2] = v;
    break;
  default:
    r.components[0] = v;
    r.components[1] = p;
    r.components[2] = q;
    break;
  }

  return r;
}
