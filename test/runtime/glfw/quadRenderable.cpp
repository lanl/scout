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
#include "scout/Runtime/opengl/glQuadRenderableVA.h"
#include <unistd.h>

using namespace std;
using namespace scout;

extern "C" float4* __scrt_renderall_uniform_colors;

int main(int argv, char** argc) {

  printf("Device instance\n");
  glDevice* glDevice = glfwDevice::Instance();


  if (glDevice) {


    printf("make glWindow\n");
    glWindow* glWindow = glDevice->createWindow(512, 512);

    printf("make context current\n");
    glWindow->makeContextCurrent();

    // need to have window made and a context before can make a renderable
    printf("quad renderable\n");
    glQuadRenderableVA* renderable = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0),
      glfloat3(512.0, 512.0, 0.0));

    printf("initialize renderable\n");
    renderable->initialize(NULL);
    printf("swap buffers\n");
    // show empty buffer
    glWindow->swapBuffers();
    printf("clear\n");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    printf("map_colors\n");
    __scrt_renderall_uniform_colors = renderable->map_colors();  
    printf("write colors into buffer\n");

    float* color;

    // write into colors buffer
    for (float x = 0.0f; x < 512.0f; x++) {
      for (float y = 0.0f; y < 512.0f; y++) {
        color = (float*)(__scrt_renderall_uniform_colors + ((int)y*512 + (int)x));
        *color = (float)(x/512.0); color++;
        *color = (float)((x+y)/(512.0+512.0)); color++;
        *color = (float)((x*y)/(512.0*512.0)); color++;
        *color = 1.0f;
      }
    }

    printf("unmap colors\n");
    renderable->unmap_colors();

    printf("draw\n");
    renderable->draw(NULL);

    printf("swap buffers\n");

    // show what we have drawn
    glWindow->swapBuffers();
    printf("sleep\n");

    sleep(1);
    printf("delete window\n");

    glDevice->deleteWindow(glWindow);

    printf("Done: \n");
  }

  return 0;
}  
