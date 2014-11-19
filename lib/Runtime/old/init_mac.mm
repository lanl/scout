#include <stdlib.h>
#include "scout/Runtime/init_mac.h"

#include <Cocoa/Cocoa.h>

namespace scout{

void scoutStopMac() {
  NSLog(@"terminate");
  [NSApp terminate: nil];
}

void scoutInitMac(){
  NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  [NSApp activateIgnoringOtherApps:YES];
  //start one thread with NSThread to make this a multithreaded app
  //NSLog(@"start dummy thread");
  [[NSThread new] start]; 
}

} // end namespace scout
