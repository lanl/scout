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

#ifndef __SC_GLFW_WINDOW_H__
#define __SC_GLFW_WINDOW_H__



#include <GLFW/glfw3.h>

#include "scout/Runtime/opengl/glWindow.h"
#include "scout/Runtime/opengl/glCamera.h"

namespace scout
{
  
  class glfwWindow : public glWindow {
    
  public:
    glfwWindow(ScreenCoord width, ScreenCoord height, glCamera* camera = NULL);
    ~glfwWindow();
    
  private:
    
    // putting these here since it makes them available to subclasses
    static void windowPosCallback(GLFWwindow* GLFWwin, int x, int y) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowPos(x, y);
    }
    static void windowSizeCallback(GLFWwindow* GLFWwin, int w, int h) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowSize(w, h);
    }
    static void windowCloseCallback(GLFWwindow* GLFWwin) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowClose();
    }
    static void windowRefreshCallback(GLFWwindow* GLFWwin) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowRefresh();
    }
    static void windowIconifyCallback(GLFWwindow* GLFWwin, int iconified) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowIconify(iconified);
    }
    static void windowFocusCallback(GLFWwindow* GLFWwin, int focused) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->windowFocus(focused);
    }
    static void framebufferSizeCallback(GLFWwindow* GLFWwin, int w, int h) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->framebufferSize(w, h);
    }
    static void errorCallback(GLFWwindow* GLFWwin, int errcode, const char* desc) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->errorfun(errcode, desc);
    }
    static void charCallback(GLFWwindow* GLFWwin, unsigned int codepoint) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->charfun(codepoint);
    }
    static void cursorEnterCallback(GLFWwindow* GLFWwin, int entered) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->cursorEnter(entered);
    }
    static void cursorPosCallback(GLFWwindow* GLFWwin, double xpos, double ypos) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->cursorPos(xpos, ypos);
    }
    static void keyCallback(GLFWwindow* GLFWwin, int key, int scancode, int action, int mods) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->keyfun(key, scancode, action, mods);
    }
    static void mouseButtonCallback(GLFWwindow* GLFWwin, int button, int action, int mods) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->mouseButton(button, action, mods);
    }
    static void scrollCallback(GLFWwindow* GLFWwin, double xoffset, double yoffset) {
      glfwWindow* glfwWin = static_cast<glfwWindow*>(glfwGetWindowUserPointer(GLFWwin));
      glfwWin->scroll(xoffset, yoffset);
    }
    
    // implementation of callbacks
  public:
    void windowPos(int w, int h){}
    void windowSize(int w, int h){}
    void windowClose(){}
    void windowRefresh(){}
    void windowIconify(int iconified){}
    void windowFocus(int focused){}
    void framebufferSize(int w, int h){}
    void errorfun(int errcode, const char* desc){}
    void charfun(unsigned int codepoint){}
    void cursorEnter(int entered){}
    void cursorPos(double xpos, double ypos){}
    void keyfun(int key, int scancode, int action, int mods);
    void mouseButton(int button, int action, int mods){}
    void scroll(double xoffset, double yoffset){}
    
  public:    
    void eventLoop();
    bool processEvent();
    void swapBuffers() { glfwSwapBuffers(_window); }
    void makeContextCurrent() { glfwMakeContextCurrent(_window); }
    void makeContextNotCurrent() { glfwMakeContextCurrent(0); }

   
#ifdef CMA
    void keyPress(int a, int b);
    void mousePress(int x, int y);
    void mouseMove(int x, int y);
#endif
    
    void update() {}
    
    
  private:
    GLFWwindow* _window;  // actual window handle for GLFW
    
  };
  
}

#endif

