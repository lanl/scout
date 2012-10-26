/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
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

#include <cstdlib>
#include <iostream>
#include "scout/Runtime/cpu/Settings.h"

using namespace std;

namespace scout {
  namespace cpu {

   int getenvBool(const char *name) {
      char *env;
      env = getenv(name);
      if (env == NULL) return -1;
      if (atoi(env) == 1) return 1;
      return 0;
    }

    int getenvUint(const char *name) {
      char *env;
      int val;
      env = getenv(name);
      if (env == NULL) return 0;
      val = atoi(env);
      if (val > 0) return val;
      return 0;
    }

    Settings::Settings() {
      enableHt_ = getenvBool("SC_RUNTIME_HT");
      enableNuma_ = getenvBool("SC_RUNTIME_NUMA");
      nThreads_ = getenvUint("SC_RUNTIME_NTHREADS");
      blocksPerThread_ = getenvUint("SC_RUNTIME_BPT");
      debug_ = getenvBool("SC_RUNTIME_DEBUG");
      if (debug_ == -1) debug_ = 0;           // debug off by default
      if (enableHt_ == -1) enableHt_ = 1;     // Hyperthreading on by default;
      if (enableNuma_ == -1) enableNuma_ = 0; // Numa off by default

      // numa specific settings in NumaSettings.cpp/CpuSettings.cpp
      // depending on if hwloc is available or not.
      numaSettings();

      if (debug_) {
        cerr << "HT " << enableHt_ << endl;
        cerr << "NUMA " << enableNuma_ << endl;
        cerr << "NTHREADS " << nThreads_ << endl;
        cerr << "BPT " << blocksPerThread_ << endl;
        cerr << "NDOMAINS " << nDomains_ << endl;
        cerr << "THREADBIND " << threadBind_ << endl;
        cerr << "WORKSTEALING " << workStealing_ << endl;
      }
    }

  }
}

