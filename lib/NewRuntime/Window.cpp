/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
 */

#include <cassert>
#include <limits.h>
#include <GLFW/glfw3.h>
#include "scout/new-runtime/opengl.h"
#include "scout/new-runtime/graphics.h"
#include "scout/new-runtime/Window.h"
using namespace scout;

Window::Window(unsigned width, unsigned height)
    : RenderTarget(RTK_window, width, height) {
  Handle = glfwCreateWindow(Width, Height, "scout", NULL, NULL);
  assert(Handle && "fatal error - could not create window");
  ColorBuffer = 0;
  glfwSetWindowUserPointer(Handle, (void*)this); // attach ourselves to the window. 
}

Window::~Window() {
  if (Handle) {
    release();
    glfwDestroyWindow(Handle);
  }
}

void Window::bind() {
  if (Handle) {
    RenderTarget::setActiveTarget(this);
    glfwMakeContextCurrent(Handle);
  }
}

void Window::release() {
  RenderTarget *RT = RenderTarget::getActiveTarget();
  if (RT == this) { // Don't accidently trash another active RT. 
    RenderTarget::setActiveTarget(0);
    glfwMakeContextCurrent(0);
  }
}

void Window::clear() {
  if (Handle) {
    assert(RenderTarget::getActiveTarget() == this && "target must be bound before clear()");
    glClearColor(Background.x, Background.y, Background.z, Background.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
}
  
void Window::swapBuffers() {
  if (Handle) {
    assert(RenderTarget::getActiveTarget() == this && "target must be bound before swapBuffers()");  
    glfwSwapBuffers(Handle);
  }
}

const float4 *Window::readColorBuffer() {
  assert(RenderTarget::getActiveTarget() == this && "target must be bound before reading buffer");
  if (ColorBuffer == 0) {
    ColorBuffer = new float4[Width * Height];
  }
  glReadPixels(0, 0, Width, Height, GL_RGBA, GL_FLOAT, (GLvoid*)ColorBuffer);
  return ColorBuffer;
}

bool Window::savePNG(const char *filename) {
  assert(RenderTarget::getActiveTarget() == this && "target must be bound before reading buffer");
  const float4* buf = readColorBuffer();
  unsigned NPixels = Width * Height;
  unsigned char *buf8 = new unsigned char[NPixels * 3];
  for(unsigned npix = 0, upix = 0; npix < NPixels; ++npix, upix += 3) {
    buf8[upix]   = (unsigned char)(ColorBuffer[npix].x * UCHAR_MAX);
    buf8[upix+1] = (unsigned char)(ColorBuffer[npix].y * UCHAR_MAX);
    buf8[upix+2] = (unsigned char)(ColorBuffer[npix].z * UCHAR_MAX);
  }
  bool retval = __scout_write_png(buf8, Width, Height, filename);
  delete []buf8;
  return retval;
}
