/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ###########################################################################
 *
 * Notes
 *
 * #####
 */
//
#include <cstdlib>
#include "scout/Runtime/opengl/glfw/glfwWindow.h"

using namespace scout;

// ---- constructor
//
glfwWindow::glfwWindow(ScreenCoord width, ScreenCoord height, glCamera* camera)
:glWindow(width, height, camera)
{
  // need to check here?
  assert(glfwInit());
  
  size_t accumRedBits = 8;
  size_t accumGreenBits = 8;
  size_t accumBlueBits = 8;
  size_t samples = 1;
  
  glfwWindowHint(GLFW_ACCUM_RED_BITS, accumRedBits);
  glfwWindowHint(GLFW_ACCUM_GREEN_BITS, accumGreenBits);
  glfwWindowHint(GLFW_ACCUM_BLUE_BITS, accumBlueBits);
  glfwWindowHint(GLFW_SAMPLES, samples);
  
  /*
   glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
   glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
   */
  
  size_t redBits = 8;
  size_t greenBits = 8;
  size_t blueBits = 8;
  size_t alphaBits = 8;
  size_t depthBits = 16;
  size_t stencilBits = 0;
  
  // 0 values mean use defaults
  
  
  
  _window = glfwCreateWindow(_width, _height, "title", NULL, NULL);  
  if (!_window)
  {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  
  // make glfw window point back to this object, so we can refer to this object in callbacks
  glfwSetWindowUserPointer(_window, (void*)this);
  
  glfwSetInputMode(_window, GLFW_STICKY_KEYS, GL_TRUE);
  
  // set standard callbacks
  glfwSetKeyCallback(_window, keyCallback);
#ifdef CMA
  glfwSetFramebufferSizeCallback(_window, framebufferSizeCallback);
  glfwSetWindowSizeCallback(_window, windowSizeCallback);
  glfwSetErrorCallback(_window, errorCallback);
  glfwSetWindowRefreshCallback(_window, windowRefreshCallback);
  glfwSetCursorPosCallback(_window, cursorPosCallback);
  glfwSetMouseButtonCallback(_window, mouseButtonCallback);
  glfwSetScrollCallback(_window, scrollCallback);
#endif
  
}

// ---- destructor
//
glfwWindow::~glfwWindow()
{
  glfwDestroyWindow(_window);
}

void glfwWindow::keyfun(int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(_window, GL_TRUE);
}

#ifdef CMA

static void keyPress(GLFWwindow* win, int a, int b)
{
  if (_ignore_events)
    return;
}

static void mousePress(GLFWwindow* win, int x, int y)
{
  if (_ignore_events)
    return;
  
  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseReleaseLeft(x, y);
    update();
  }
}

static void mouseMove(int x, int y)
{
  if (_ignore_events)
    return;
  
  if (_manipulator && _manipulator->isActive()) {
    _manipulator->mouseMoveEvent(x, y);
    update();
  }
}
#endif

void glfwWindow::eventLoop()
{
  bool done = false;
  while (!done)
  {
    paint();
    glfwSwapBuffers(_window);
    done = processEvent();
  }
  
  glfwDestroyWindow(_window);
  
}

bool glfwWindow::processEvent()
{
  bool done = false;
  glfwWaitEvents();  // better than glfwPollEvents() for our purposes
  if (glfwWindowShouldClose(_window)) {
    done = true;
  }
  return done;
}

