#include <iostream>
#include <sstream>

#include "runtime/framebuffer.h"

using namespace std;
using namespace scout;

framebuffer_rt* _framebuffer;
size_t _frame = 0;

void scoutInit(int argc, char** argv){
  _framebuffer = new framebuffer_rt(1024, 1);
}

void scoutSwapBuffers(){
  stringstream sstr;
  sstr << "scout-" << _frame << ".png";
  
  save_framebuffer_as_png(_framebuffer, sstr.str().c_str());
  _framebuffer->clear();
  ++_frame;
}

double cshift(double a, int dx, int axis){

}

float cshift(float a, int dx, int axis){

}

int cshift(int a, int dx, int axis){

}

float4 hsv(float h, float s, float v){

}
