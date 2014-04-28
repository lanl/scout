/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 * simple tests that we can make a glWindow and draw to it.
 * 
 */

#include <iostream>
#include "scout/Runtime/opengl/glfw/glfwDevice.h"
#include <unistd.h>

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024

using namespace std;
using namespace scout;

int main(int argv, char** argc) {

  glDevice* glDevice_ = glfwDevice::Instance();

  int width = WINDOW_WIDTH;
  int height = WINDOW_HEIGHT;

  if (glDevice_) {
    glWindow* glWindow_ = glDevice_->createWindow(width, height);

    glWindow_->makeContextCurrent();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // show empty buffer
    glWindow_->swapBuffers();

    bool done = false;

    for (int i = 0; i < 100; i++) {
      float ratio = width / (float) height;

      glViewport(0, 0, width, height);
      glClear(GL_COLOR_BUFFER_BIT);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
      glMatrixMode(GL_MODELVIEW);

      glLoadIdentity();
      glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);

      glBegin(GL_TRIANGLES);
      glColor3f(1.f, 0.f, 0.f);
      glVertex3f(-0.6f, -0.4f, 0.f);
      glColor3f(0.f, 1.f, 0.f);
      glVertex3f(0.6f, -0.4f, 0.f);
      glColor3f(0.f, 0.f, 1.f);
      glVertex3f(0.f, 0.6f, 0.f);
      glEnd();

      glWindow_->swapBuffers();
    }

    glDevice_->deleteWindow(glWindow_);

  }

  return 0;

}

