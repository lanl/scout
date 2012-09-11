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

#include <Cocoa/Cocoa.h>
#include <CoreGraphics/CGDirectDisplay.h>

#include "scout/Runtime/opengl/macosx/oglApplication.h"
#include "scout/Runtime/opengl/macosx/oglDevice.h"

static id scAutoreleasePool;


/** ----- getActiveGPUCount
 * Return the number of GPUs (video devices) that have a display
 * attached... 
 */
static int getActiveGPUCount()
{
  CGDisplayCount count;
  CGError err = CGGetActiveDisplayList(0, NULL, &count);
  return (int)count;
}

/** ----- createAppMenu
 *
 */
static void createAppMenu() {

  NSMenu *mainMenu = [[[NSMenu alloc] initWithTitle:@"Scout"] autorelease];
  NSMenuItem *item;
  NSMenu *submenu;

  item = [mainMenu addItemWithTitle:@"Apple" action:NULL keyEquivalent:@""];
  submenu = [[[NSMenu alloc] initWithTitle:@"Apple"] autorelease];
  [NSApp performSelector:@selector(setAppleMenu:) withObject:submenu];
  // Need to populate App menu here...
  [mainMenu setSubmenu:submenu forItem:item];

  item = [mainMenu addItemWithTitle:@"File" action:NULL keyEquivalent:@""];
  submenu = [[[NSMenu alloc] initWithTitle:NSLocalizedString(@"File", @"The File menu")] autorelease];
  // Need to populate File menu here... 
  [mainMenu setSubmenu:submenu forItem:item];

  item = [mainMenu addItemWithTitle:@"Edit" action:NULL keyEquivalent:@""];
  submenu = [[[NSMenu alloc] initWithTitle:NSLocalizedString(@"Edit", @"The Edit menu")] autorelease];
  // Need to populate File menu here... 
  [mainMenu setSubmenu:submenu forItem:item];

  item = [mainMenu addItemWithTitle:@"Window" action:NULL keyEquivalent:@""];
  submenu = [[[NSMenu alloc] initWithTitle:NSLocalizedString(@"Window", @"The Window menu")] autorelease];
  // Need to populate Window menu here... 
  [mainMenu setSubmenu:submenu forItem:item];
  [NSApp setWindowsMenu:submenu];

  item = [mainMenu addItemWithTitle:@"Help" action:NULL keyEquivalent:@""];
  submenu = [[[NSMenu alloc] initWithTitle:NSLocalizedString(@"Help", @"The Help menu")] autorelease];
  // Need to populate Window menu here...   
  [mainMenu setSubmenu:submenu forItem:item];

  [NSApp setMainMenu:mainMenu];
}


/** ----- scRunEventLoop
 *
 *
 */
void scRunEventLoop()
{
  NSEvent *event;
  do {

    event = [NSApp nextEventMatchingMask:NSAnyEventMask
                               untilDate:[NSDate distantPast]
                                  inMode:NSDefaultRunLoopMode
                                 dequeue:YES];
    if (event) {
      [NSApp sendEvent:event];
    }
  } while (event);

  [scAutoreleasePool drain];
  scAutoreleasePool = [[NSAutoreleasePool alloc] init];  
}



/** ----- initializeCocoa
 *
 */
static void initializeCocoa() {

  scAutoreleasePool = [[NSAutoreleasePool alloc] init];
  
  if (NSApp == nil) {
    // Initialize the display environment -- this also has the side
    // effect of giving us a global handle to the application instance
    // NSApp...
    [oglApplication sharedApplication];

    // Make sure Scout application show up in the dock and have a
    // menu bar interface for the user... 
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Make sure we're the active application... 
    [NSApp activateIgnoringOtherApps:YES];


    // Create the main menu for the app.
    createAppMenu();
    
    // We intentionally want to avoid the event loop loop here -- we
    // need to take some special steps to deal with that to provide
    // not only the runtime but also end-user applications with
    // flexibility...  The following call runs the final
    // initialization steps and leaves in a state where we are ready
    // to start processing events.
    [NSApp finishLaunching];
  }
}


/** ----- finalizeCocoa
 *
 *
 */
static void finalizeCocoa() {
  [scAutoreleasePool release];
  scAutoreleasePool = nil;
}


namespace scout {

  namespace opengl {
    
    /** ----- scInitialize
     * Initialize the OpenGL runtime.  For MacOS X this basically
     * requires us to create a Cocoa enviornment that provides not
     * only access to a suitable OpenGL context but also the full
     * application infrastructure for event handling, menu bar and
     * dock icon.
     */
    int scInitialize(DeviceList &devList) {

      // Get the Cocoa enviornment up and going... 
      initializeCocoa();
      
      // See how many displays we have connected to the system.
      // We have this here to get a rough count of the number of
      // GPUs but note that this may be inaccurate...  It has been
      // difficult to sort out the guts of the core libraries of
      // Mac OS X...
      //
      // TODO: This entire code based needs some serious help to
      // seamlessly support multiple GPUS and displays.
      int numDevices = getActiveGPUCount();

      // Now create a MacOS-centric glDevice and add it to the
      // runtime's device list.  At present we only return a single
      // device -- i.e. we don't handle the case where there are
      // multiple GPUs in a system.
      for(int i = 0; i < numDevices; ++i) {
        oglDevice *device = new oglDevice();
        if (device->isEnabled()) {
          devList.push_back(device);
        }
      }
      return 0;
    }

    
    /** ----- scFinalize
     * Clean up the OpenGL runtime. 
     *
     */
    void scFinalize() {
      finalizeCocoa();
    }
    
  }
}
