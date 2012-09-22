/*
 * ###########################################################################
 * Copyrigh (c) 2010, Los Alamos National Security, LLC.
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

#include "scout/Runtime/opengl/glx/glxDevice.h"

using namespace scout;


/** ----- glxDevice
 *
 *
 */
glxDevice::glxDevice(Display *dpy) {
  display = dpy;
  if (display != 0)
    enabled = true;
  else
    enabled = false;
}


/** ----- ~glxDevice
 *
 *
 */
glxDevice::~glxDevice()
{
  // We currently manage/close the display within the top-level
  // initialization code (see glxInitialization.cpp). 
}


/** ----- createWindow
 *
 *
 */
glWindow *glxDevice::createWindow(unsigned short xpos,
                                 unsigned short ypos,
                                 unsigned short width,
                                 unsigned short height)
    : glWindow(WindowRect(xpos, ypos, width, height)) {

  assert(display != 0);

  Bool err;  
  int glx_major, glx_minor;

  err = glXQueryVersion(display, &glx_major, &glx_minor);
  if (err) {
    // TODO: Need error message here -- can't query GLX. 
    return 0;
  }

  if ((glx_major == 1 && glx_minor < 3) || glx_major < 1) {
    // TODO: Need error message here -- unsupported version.
    return 0;
  }

  static int visual_attribs[] = {
    GLX_X_RENDERABLE,         True,
    GLX_DRAWABLE_TYPE,        GLX_WINDOW_BIT,
    GLX_RENDER_TYPE,          GLX_RGBA_BIT,
    GLX_RED_SIZE,             8,
    GLX_GREEN_SIZE,           8,
    GLX_BLUE_SIZE,            8,
    GLX_ALPHA_SIZE,           8,
    GLX_DEPTH_SIZE,           24,
    GLX_DOUBLEBUFFER,         True,
    None
  };

  int matching_count = 0;
  GLXFBConfig *fbufferConfig = glXChooseFBConfig(display, DefaultScreen(display),
                                                 visual_attribs, &matching_count);
  
  // not sure matching_count test below is valid...  
  if (! fbufferConfig || matching_count == 0) {
    // TODO: Need error message here -- no macthing frame buffer configuration.
    return 0;
  }

  int best_config = 0, best_samples = -1;
  
  for(int i = 0; i < matching_count; ++i) {
    XVisualInfo *vinfo = glXGetVisualFromFBConfig(display, fbufferConfigs[i]);
    if (vinfo) {
      int has_samp_buf, num_samples;
      glXGetFBConfigAttrib(display, fbufferConfig[i], GLX_SAMPLE_BUFFERS, &has_samp_buf);
      glXGetFBConfigAttrib(display, fbufferConfig[i], GLX_SAMPLES, &num_samples);
      if (has_samp_buf) {
        if (num_samples > best_samples) {
          best_samples = num_samples;
          best_config  = i;
        }
      }
      XFree(vi);
    }
  }

  GLXFBConfig bestConfig = fbufferConfig[best_config];
  XFree(fbufferConfig);

  glxWindow *win = new glXWindow(xpos, ypos, width, height, bestConfig);
  if (win) {
    winList.push_back(win);
  }
  XFree(fbufferConfig);
  return win;
}


/** ----- processXEvent
 * Process only a single X event.  
 */
bool glxDevice::processXEvent() {

  bool windowClosed = false;

  XEvent x_event;
  XNextEvent(display, &x_event);

  // We have to first map from the X11 window that received
  // the event to the higher level (glWindow) that we should
  // dispatch the event to.  For now we do this in a brute
  // force way -- assuming we will never have a large enough
  // list of windows to make a linked list walk a performance
  // bottleneck.
  glWindowList::iterator it = windows.begin(), end = windows.end();
  glEvent sc_event;
  bool dispatched = false;
  
  while(it != end && !dispatched) {
    glxWindow *win = static_cast<glxWindow*>(*it);
    if (event.window == win->xWindow()) {
      translateXEvent(&x_event, &sc_event);
      win->dispatchEvent(&sc_event);
      dispatched = true;
      break;
    }
    ++it;
  }
}
