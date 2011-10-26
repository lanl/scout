#include <cmath>
#include <iostream>
#include <sstream>

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
				    /*SDL_RESIZABLE |*/
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

    static const float pad = 0.05;

    if(dy == 0){
      float px = pad * dx;
      gluOrtho2D(-px, dx + px, -px, dx + px);
      _uniform_renderall = __sc_init_uniform_renderall(dx);
    }
    else{
      if(dx >= dy){
	float px = pad * dx;
	float py = (1 - float(dy)/dx) * dx * 0.50; 
	gluOrtho2D(-px, dx + px, -py - px, dx - py + px);
      }
      else{
	float py = pad * dy;
	float px = (1 - float(dx)/dy) * dy * 0.50;
	gluOrtho2D(-px - py, dx + px + py, -py, dy + py);
      }

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

  SDL_Event evt;
  SDL_PollEvent(&evt);
  switch(evt.type){
    case SDL_QUIT:
    {
      exit(0);
      break;
    }
    case SDL_KEYDOWN:
    {
      switch(evt.key.keysym.sym){
        case SDLK_ESCAPE:
	{
	  exit(0);
	  break;
	}
        default:
	{
	  break;
	}
      }
    } 
    /*
    case SDL_VIDEORESIZE:
    {
      _sdl_surface = SDL_SetVideoMode(evt.resize.w, evt.resize.h, 32,
				      SDL_HWSURFACE |
				      SDL_RESIZABLE |
				      SDL_GL_DOUBLEBUFFER |
				      SDL_OPENGL);
      break;
    }
    */
  }
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
