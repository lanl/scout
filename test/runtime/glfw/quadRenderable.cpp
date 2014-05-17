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

  glDevice* glDevice = glfwDevice::Instance();


  if (glDevice) {


    glWindow* glWindow = glDevice->createWindow(512, 512);

    glWindow->makeContextCurrent();

    // need to have window made and a context before can make a renderable
    glQuadRenderableVA* renderable = new glQuadRenderableVA( glfloat3(0.0, 0.0, 0.0),
      glfloat3(512.0, 512.0, 0.0));

    renderable->initialize(NULL);
    // show empty buffer
    glWindow->swapBuffers();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    __scrt_renderall_uniform_colors = renderable->map_colors();  

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

    renderable->unmap_colors();

    renderable->draw(NULL);


    // show what we have drawn
    glWindow->swapBuffers();

    sleep(1);

    glDevice->deleteWindow(glWindow);

  }

  return 0;
}  
