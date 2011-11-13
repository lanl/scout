#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

#ifdef __APPLE__

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include "runtime/init_mac.h"

#else

#include <GL/gl.h>
#include <GL/glu.h>

#endif

#include <SDL/SDL.h>

#define SC_USE_PNG

#include "runtime/scout_gpu.h"
#include "runtime/cuda/cuda.h"

#include "runtime/opengl/uniform_renderall.h"
#include "runtime/tbq.h"

using namespace std;
using namespace scout;

SDL_Surface* _sdl_surface;
uniform_renderall_t* _uniform_renderall = 0;
scout::float4* _pixels;
tbq_rt* _tbq;

static bool _gpu = false;
static size_t _dx;
static size_t _dy;
static size_t _dz;

static const size_t WINDOW_WIDTH = 1024;
static const size_t WINDOW_HEIGHT = 1024;

void initSDLWindow() {
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
                                  SDL_RESIZABLE |
                                  SDL_GL_DOUBLEBUFFER |
                                  SDL_OPENGL);
}

void scoutInit(int& argc, char** argv, bool gpu){
  if(SDL_Init(SDL_INIT_VIDEO) < 0){
    cerr << "Error: failed to initialize SDL." << endl;
    exit(1);
  }

  _tbq = new tbq_rt;

  if(gpu){
    initSDLWindow(); // CUDA requires an active OpenGL context.
    __sc_init_cuda();
  }
}

void scoutInit(bool gpu){
  if(SDL_Init(SDL_INIT_VIDEO) != 0){
    cerr << "Error: failed to initialize SDL." << endl;
    exit(1);
  }

  _tbq = new tbq_rt;

  if(gpu){
    initSDLWindow(); // CUDA requires an active OpenGL context.
    __sc_init_cuda();
  }
}

static void _initViewport(){
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();

  static const float pad = 0.05;

  if(_dy == 0){
    float px = pad * _dx;
    gluOrtho2D(-px, _dx + px, -px, _dx + px);
    _uniform_renderall = __sc_init_uniform_renderall(_dx);
  }
  else{
    if(_dx >= _dy){
      float px = pad * _dx;
      float py = (1 - float(_dy)/_dx) * _dx * 0.50;
      gluOrtho2D(-px, _dx + px, -py - px, _dx - py + px);
    }
    else{
      float py = pad * _dy;
      float px = (1 - float(_dx)/_dy) * _dy * 0.50;
      gluOrtho2D(-px - py, _dx + px + py, -py, _dy + py);
    }

    _uniform_renderall = __sc_init_uniform_renderall(_dx, _dy);
  }

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glClearColor(0.5, 0.55, 0.65, 0.0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  SDL_GL_SwapBuffers();
}

void scoutBeginRenderAll(size_t dx, size_t dy, size_t dz){
  _dx = dx;
  _dy = dy;
  _dz = dz;

  // if this is our first render all, then initialize SDL and
  // the OpenGL runtime
  if(!_uniform_renderall){

#ifdef __APPLE__
    scoutInitMac();
#endif

    initSDLWindow();

    if(!_sdl_surface){
      cerr << "Error: failed to initialize SDL surface." << endl;
      exit(1);
    }

    _initViewport();
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(_scout_gpu)
    _map_gpu_resources();
  else
    _pixels = __sc_map_uniform_colors(_uniform_renderall);
}

void scoutEndRenderAll(){
  if(_scout_gpu)
    _unmap_gpu_resources(_uniform_renderall);
  else
    __sc_unmap_uniform_colors(_uniform_renderall);

  __sc_exec_uniform_renderall(_uniform_renderall);

  SDL_GL_SwapBuffers();

  SDL_Event evt;
  if(!SDL_PollEvent(&evt)){
    return;
  }

  switch(evt.type){
    case SDL_QUIT:
    {
      exit(0);
    }
    case SDL_KEYDOWN:
    {
      switch(evt.key.keysym.sym){
        case SDLK_ESCAPE:
	{
	  exit(0);
	}
        default:
	{
	  break;
	}
      }
      break;
    }
    case SDL_VIDEORESIZE:
    {
      size_t width = evt.resize.w;
      size_t height = evt.resize.h;

      SDL_FreeSurface(_sdl_surface);


      _sdl_surface = SDL_SetVideoMode(width, height, 32,
				      SDL_HWSURFACE |
				      SDL_RESIZABLE |
				      SDL_GL_DOUBLEBUFFER |
				      SDL_OPENGL);

      _initViewport();

      break;
    }
  }
}

void scoutBeginRenderAllElements(size_t dx, size_t dy, size_t dz){
  _dx = dx;
  _dy = dy;
  _dz = dz;
}

void scoutEndRenderAllElements(){

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
