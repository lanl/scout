#include "scout/Runtime/init_mac.h"

#include <Cocoa/Cocoa.h>

namespace scout{

void scoutInitMac(){
  NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  [NSApp activateIgnoringOtherApps:YES];
}

} // end namespace scout
