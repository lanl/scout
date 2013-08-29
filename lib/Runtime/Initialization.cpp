/*
 * ###########################################################################
 * Copyrigh (c) 2013, Los Alamos National Security, LLC.
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

#include "scout/Runtime/Initialization.h"

namespace scout {

  /** ----- scInitializeRuntime
   * Initialize the runtime and return 'true' on success and
   * false.
   */
  bool __scInitializeRuntime(...) {

    // Step 1. Initialize Legion.

    // Step 2. Determine if we have any local graphics capabilities
    // (e.g. OpenGL) or if we should fall back to software rendering.
    // Note we treat framebuffers (in hardware or software) as
    // exclusive devices.  In other words, a set of independent tasks
    // that all need graphics access will be serialized via a lock.
  }

  
  /** ----- scInitializeRuntime
   * Search for and build a list of supported devices. 
   */
  int __scInitializeRuntime(DeviceList &deviceList) {
    //(void)opengl::scInitialize(deviceList);    
    //(void)cuda::scInitialize(deviceList);
    //(void)opencl::scInitialize(deviceList);
    //(void)numa::scInitialize(deviceList);
    return 0; // Should we return the total number of devices initialized???
  }


  /** ----- scFinalizeRuntime
   * Shutdown and destroy all devices. 
   */
  int __scFinalizeRuntime(DeviceList &deviceList) {

    // Destroy all devices.
    /*
    Devices::iterator it = deviceList.devices.begin();
    while(it != deviceList.devices.end()) {
      delete *it;
      ++it;
    }
    */
    // Clean up each of the invididual platform runtimes. 
    // (void)cuda::scFinalize(deviceList);
    // (void)opengl::scFinalize(deviceList);
    return 0;
  }
}


  
