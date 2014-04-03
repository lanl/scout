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

#include "scout/Runtime/opengl/glx/glxWindow.h"

using namespace scout;

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>

/** ----- glxWindow
 *
 *
 */
glxWindow::glxWindow(unsigned short  xpos,
                     unsigned short  ypos,
                     unsigned short  width,
                     unsigned short  height,
                     Display        *dpy, 
                     GLXFBConfig     fbConfig) {
  : glWindow(WindowRect(xpos, ypos, width, height)) {

    display = dpy;
    XVisualInfo *vinfo = glXGetVisualFromFBConfig(display, fbConfig);
    
    XSetWindowAttributes swa;
    Color cmap;
    swa.colormap = cmap XCreateColormap(display, RootWindow(display, vinfo->screen),
                                        vinfo->visual, AllocNone);
    swa.background_pixmap = None;
    swa.border_pixel      = 0;
    swa.event_mask        = StructureNotifyMask;

    xWin = XCreateWindow(display, RootWindow(display, vinfo->screen),
                        xpos, ypos, width, height, vinfo->depth,
                        InputOutput, vinfo->visual,
                        CWBorderPixel | CWColormap | CWEventMask, &swa);

    XFree(vinfo);
    if (! xWin) {
      valid = false;
      XFreeColormap(display, cmap);      
    } else {

      XStoreName(display, xWin, "Scout");

      const char *glxExts = glXQueryExtensionsString(display, DefaultScreen(display));
      glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
      glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
        glxGetProcAddressARB((const GLubyte *)"glXCreateContextAttribsARB");
      
      context = 0;

      int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        None
      };
 
      context = glXCreateContextAttribsARB(display, fbConfig, 0,
                                        True, context_attribs);

      // Sync to ensure errors are generated/processed.      
      XSync(display, False); 

      if (! glXIsDirect(display, context)) {
        valid = false;
        glXDestroyContext(display, ctx);
        XDestroyWindow(display, win);
        XFreeColormap(display, cmap);
      } else {
        XMapWindow(display, xWin);
        glXMakeCurrent(display, xWin, context);
        valid = true;
      }
    }
  }
}


/** ----- ~glxWindow
 *
 *
 */
glxWindow::~glxWindow() {
  if (valid) {
    glXMakeCurrent(display, 0, 0);
    glXDestroyContext(display, ctx);
    XDestroyWindow(display, win);
    XFreeColormap(display, cmap);
  }
}
  
  
/** ----- setTitle
 *
 *
 */
void glxWindow::setTitle(const char *title) {
  XStoreName(display, xWin, title);
  XSetIconName(display, xWin, title );
}


/** ----- minimize
 *
 *
 */
void glxWindow::minimize() {
  XIconifyWindow(display, xWin, DefaultScreen(display, xWin));
}


/** ----- restore
 *
 *
 */
void glxWindow::restore() {
  XMapWindow(display, xWin);
}


/** ----- refresh
 *
 *
 */
void glxWindow::refresh() {
  glXMakeCurrent(display, xWin, context);
   glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // TODO: do scout-centric drawing here... 
  glXSwapBuffers( _glfwLibrary.display, _glfwWin.window );
}


