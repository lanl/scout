/*
 * ###########################################################################
 * Copyright(c) 2010, Los Alamos National Security, LLC.
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

#include "scout/Runtime/opengl/glWindow.h"

using namespace scout;

/// Create a window of the given width and height.  The window's
/// position is undetermined and the background color will default
/// to black.
glWindow::glWindow(ScreenCoord width, ScreenCoord height)
: _frame(width, height) {
  _bgColor.red   = 0.0f;
  _bgColor.green = 0.0f;
  _bgColor.blue  = 0.0f;
  _bgColor.alpha = 1.0f;
}

/// Create a window with the given location and size (as described
/// by the given WindowRect).  The background color will default
/// to black.
glWindow::glWindow(const WindowRect &rect)
: _frame(rect) {
  _bgColor.red   = 0.0f;
  _bgColor.green = 0.0f;
  _bgColor.blue  = 0.0f;
  _bgColor.alpha = 1.0f;
}

/// Create a window with the given location, size (as described by
/// the given WindowRect) and background color.
glWindow::glWindow(const WindowRect &rect, const oglColor &color)
: _frame(rect) {
  _bgColor.red   = color.red;
  _bgColor.green = color.green;
  _bgColor.blue  = color.blue;
  _bgColor.alpha = color.alpha;
}

/// Set the window's background color.
void glWindow::setBackgroundColor(float red,
                        float green,
                        float blue) {
  _bgColor.red   = red;
  _bgColor.green = green;
  _bgColor.blue  = blue;
  _bgColor.alpha = 1.0;
}


/// Set the window's background color.
void glWindow::setBackgroundColor(const oglColor &rgba) {
  _bgColor.red   = rgba.red;
  _bgColor.green = rgba.green;
  _bgColor.blue  = rgba.blue;
  _bgColor.alpha = rgba.alpha;
}
