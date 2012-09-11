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

#include "scout/Runtime/opengl/macosx/oglContext.h"
#include "scout/Runtime/opengl/macosx/oglView.h"

using namespace scout;

@implementation oglView

/** ----- initWithFrame 
 *
 */
- (id)initWithFrame:(NSRect)frameRect openglContext:(scout::oglContext*)ctx {

  self = [super initWithFrame:frameRect];
  if (self != nil) {
    context = ctx;
    [[NSNotificationCenter defaultCenter] addObserver:self
                                           selector:@selector(_surfaceNeedsUpdate:)
                                           name:NSViewGlobalFrameDidChangeNotification
                                           object:self];
  }
  
  return self;
}


/** ----- _surfaceNeeedsUpdate
 *
 */
- (void) _surfaceNeedsUpdate:(NSNotification*)notification {
  [self update];
}


/** ----- isOpaque
 *
 */
- (BOOL)isOpaque {
  return YES;
}


/** ----- canBecomeKeyView
 *
 */
- (BOOL)canBecomeKeyView {
  return YES;
}


/** ----- acceptsFirstResponder
 *
 */
- (BOOL)acceptsFirstResponder {
  return YES;
}


/** ----- mouseDown
 *
 */
- (void)mouseDown:(NSEvent*)event {

}


/** ----- mouseDragged
 *
 */
- (void)mouseDragged:(NSEvent*)event {

}


/** ----- mouseUp
 *
 */
- (void)mouseUp:(NSEvent*)event {

}


/** ----- mouseMoved
 *
 */
- (void)mouseMoved:(NSEvent*)event {

}


/** ----- rightMouseDown
 *
 */
- (void)rightMouseDown:(NSEvent*)event {

}


/** ----- rightMouseDragged
 *
 */
- (void)rightMouseDragged:(NSEvent*)event {

}

/** ----- rightMouseUp
 *
 */
- (void)rightMouseUp:(NSEvent*)event {

}


/** ----- otherMouseDown
 *
 */
- (void)otherMouseDown:(NSEvent*)event {

}


/** ----- otherMouseDragged
 *
 */
- (void)otherMouseDragged:(NSEvent*)event {

}


/** ----- otherMouseUp
 *
 */
- (void)otherMouseUp:(NSEvent*)event {

}


/** ----- keyDown
 *
 */
- (void)keyDown:(NSEvent*)event {

}


/** ----- flagsChanged
 *
 */
- (void)flagsChanged:(NSEvent*)event {

}


/** ----- keyUp
 *
 */
- (void)keyUp:(NSEvent*)event {

}


/** ----- scrollWheel
 *
 */
- (void)scrollWheel:(NSEvent*)event {

}


/** ----- update
 *
 */
- (void)update {

}
 

@end 
