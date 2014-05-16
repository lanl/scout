/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * simplerenderable.cpp just makes a renderallUniform displays an image and closes
 * 
 */
#include <iostream>
#include "scout/Runtime/opengl/glfw/glfwDevice.h"
#include <unistd.h>

using namespace std;
using namespace scout;

extern "C" float4* __scrt_renderall_uniform_colors;
extern "C" void __scrt_renderall_uniform_begin(size_t width, size_t height, size_t depth,
                   RenderTarget* renderTarget);
extern "C" void __scrt_renderall_end();
extern "C" void __scrt_renderall_delete();

int main(int argv, char** argc) {

  printf("Device instance\n");
  glDevice* glDevice = glfwDevice::Instance();


  if (glDevice) {


    printf("make glWindow\n");
    glWindow* glWindow = glDevice->createWindow(512, 512);

    printf("begin renderall\n");
    __scrt_renderall_uniform_begin(512, 512, 0, glWindow);

    float* color;

    printf("write to colors\n");
    // write into colors buffer
    for (float x = 0.0f; x < 512.0f; x++) {
      for (float y = 0.0f; y < 512.0f; y++) {
        color = (float*)(__scrt_renderall_uniform_colors + ((int)x*512 + (int)y));
        *color = (float)(x/512.0); color++;
        *color = (float)((x+y)/(512.0+512.0)); color++;
        *color = (float)((x*y)/(512.0*512.0)); color++;
        *color = 1.0f;
      }
    }

    printf("end renderall\n");
    __scrt_renderall_end();

    sleep(1);

    printf("delete renderall\n");
    __scrt_renderall_delete();

    printf("Done: \n");
  }

  return 0;
}  
