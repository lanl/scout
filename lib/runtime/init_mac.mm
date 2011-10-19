#include "runtime/init_mac.h"

#include <Cocoa/Cocoa.h>

namespace scout{

void scoutInitMac(){
  NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
  [NSApplication sharedApplication];
}

} // end namespace scout
