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

#ifndef __SC_NSGL_WINDOW_H__
#define __SC_NSGL_WINDOW_H__

#include "scout/Runtime/opengl/glWindow.h"
#include "scout/Runtime/opengl/macosx/nsglView.h"

namespace scout {

  /** ----- oglWindow
   * This class serves primarily as the interface between C++
   * and the supporting Cocoa/Objective C code. 
   */
  class nsglWindow : public glWindow {

   public:

    /**
     * Create a window at the given (x, y) position and width
     * and height.
     */
    nsglWindow(unsigned short xpos,
               unsigned short ypos,
               unsigned short width,
               unsigned short height);

    /**
     * Destroy the window -- this will terminate an entire 
     * Scout application.
     */
    ~nsglWindow();
    
    void swapBuffers(){}
    
    bool processEvent(){}
    
    void eventLoop(){}

    /**
     * Set the title of the window.
     */
    void setTitle(const char *title);

    /**
     * Close the window.
     */
    void close();

    /**
     * Minimize (iconify) the window.
     */
    void minimize();

    /**
     * Restore the window from an iconified state.
     */
    void restore();

    /**
     * Force the window to refresh/redraw its contents.
     */
    void refresh();

    /**
     * Is the window in a valid/usable state?
     */
    bool isValid() const {
       return valid;
    }
    
   private:
    bool        valid;
    NSWindow   *window;
    nsglView   *view;
  };

}

#endif

