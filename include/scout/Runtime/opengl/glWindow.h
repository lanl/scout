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

#ifndef __SC_GL_WINDOW_H__
#define __SC_GL_WINDOW_H__

#include <string>
#include <list>

#include "scout/Runtime/opengl/opengl.h"

namespace scout {

  /**
   * For future flexibility we represent both screen coordinates and
   * dimensions using this type -- both measured in terms of pixels.
   */   
  typedef unsigned short ScreenCoord;

  
  /** ----- ScreenPoint
   * A location on the screen measured in pixels. 
   */
  struct ScreenPoint {

    ScreenPoint(ScreenCoord px,
                ScreenCoord py) 
        : x(px),
          y(py) {
    }

    ScreenPoint(const ScreenPoint &other)
        : x(other.x),
          y(other.y) {
    }
    
    ScreenCoord x;
    ScreenCoord y;
  };

  
  /** ----- ScreenSize
   * The width and height of an on-screen element measured in pixels.
   */
  struct ScreenSize {
    
    ScreenSize(ScreenCoord w,
               ScreenCoord h)
        : width(w),
          height(h) {
    }

    ScreenSize(const ScreenSize &other)
        : width(other.width),
          height(other.height) {
    }
    
    ScreenCoord   width;
    ScreenCoord   height;
  };


  /** ----- WindowRect
   * A window rectangle captures both the position and the dimensions
   * of an on-screen region.  As the name implies, it is primarily
   * used to describe a window.
   */
  struct WindowRect {

    WindowRect(ScreenCoord w, ScreenCoord h)
        : origin(0, 0),
          size(w, h) {
    }

    WindowRect(ScreenCoord x, ScreenCoord y,
               ScreenCoord w, ScreenCoord h)
        : origin(x, y),
          size(w, h) {
    }

    WindowRect(const ScreenPoint &P,
               const ScreenSize  &dims)
        : origin(P),
          size(dims) {
    }
    
    ScreenPoint  origin;
    ScreenSize   size;
  };

  
  /** ----- glWindow
   * The base class for an OpenGL-enabled window.  The window is
   * described by a given rectangle region, or an given set of width
   * and height dimensions.  In addition, it is possible to assign an
   * on-screen title and background color for the rendering area of
   * the window.  The overall, platform-centric, details of the window
   * must be provied by subclasses. 
   */
  class glWindow {
    
   public:
    
    /// Create a window of the given width and height.  The window's
    /// position is undetermined and the background color will default
    /// to black.
    glWindow(ScreenCoord width, ScreenCoord height)
        : frame(width, height) {
      bgColor.red   = 0.0f;
      bgColor.green = 0.0f;
      bgColor.blue  = 0.0f;
      bgColor.alpha = 1.0f;      
    }
    
    /// Create a window with the given location and size (as described
    /// by the given WindowRect).  The background color will default
    /// to black.
    glWindow(const WindowRect &rect)
        : frame(rect) {
      bgColor.red   = 0.0f;
      bgColor.green = 0.0f;
      bgColor.blue  = 0.0f;
      bgColor.alpha = 1.0f;      
    }
    
    /// Create a window with the given location, size (as described by
    /// the given WindowRect) and background color.
    glWindow(const WindowRect &rect, const oglColor &color) 
        : frame(rect) {
      bgColor.red   = color.red;
      bgColor.green = color.green;
      bgColor.blue  = color.blue;
      bgColor.alpha = color.alpha;      
    }

    virtual ~glWindow() {
      // no-op
    };

    /// Set the window's on screen title.  
    virtual void setTitle(const char *title) = 0;

    /// Minimize the window (aka iconify). 
    virtual void minimize() = 0;

    /// Restore the window from a minimized/iconic state. 
    virtual void restore() = 0;

    /// Refresh/update the window's display. 
    virtual void refresh() = 0;

    /// Set the window's background color.
    void setBackgroundColor(float red,
                            float green,
                            float blue) {
      bgColor.red   = red;
      bgColor.green = green;
      bgColor.blue  = blue;
      bgColor.alpha = 1.0;
    }

    /// Set the window's background color.
    void setBackgroundColor(const oglColor &rgba) {
      bgColor.red   = rgba.red;
      bgColor.green = rgba.green;
      bgColor.blue  = rgba.blue;
      bgColor.alpha = rgba.alpha;
    }

   protected:
    WindowRect  frame;     /// The position and dimensions of window. 
    oglColor    bgColor;   /// Window's background color. 
  };
  
  /**
   * A convenient shorthand that reduces template syntax when managing
   * a collection of windows. 
   */
  typedef std::list<glWindow*> glWindowList;
}

#endif
