#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

#include <SDL/SDL.h>

#define SC_USE_PNG

#include "runtime/scout_gpu.h"
#include "runtime/cuda/cuda.h"
#include "runtime/init_mac.h"

#include "runtime/tbq.h"

using namespace std;
using namespace scout;

tbq_rt* __sc_tbq;
SDL_Surface* __sc_sdl_surface = 0;

size_t __sc_initial_width = 768;
size_t __sc_initial_height = 768;

extern "C"
void __sc_queue_block(void* blockLiteral, int numDimensions, int numFields){
  __sc_tbq->run(blockLiteral, numDimensions, numFields);
}

void __sc_init_sdl(size_t width, size_t height){

  if(__sc_sdl_surface){

    SDL_FreeSurface(__sc_sdl_surface);
    
    __sc_sdl_surface = SDL_SetVideoMode(width, height, 32,
					SDL_HWSURFACE |
					SDL_RESIZABLE |
					SDL_GL_DOUBLEBUFFER |
					SDL_OPENGL);

    return;
  }
  
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

  __sc_sdl_surface = SDL_SetVideoMode(width, height, 32,
				      SDL_HWSURFACE |
				      SDL_RESIZABLE |
				      SDL_GL_DOUBLEBUFFER |
				      SDL_OPENGL);
}

void __sc_init(int argc, char** argv, bool gpu){
  __sc_tbq = new tbq_rt;

  if(gpu){
    __sc_init_sdl(__sc_initial_width, __sc_initial_height);
    __sc_init_cuda();
  }
}

void __sc_init(bool gpu){
  __sc_init(0, 0, gpu);
}

void __sc_end(){

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
