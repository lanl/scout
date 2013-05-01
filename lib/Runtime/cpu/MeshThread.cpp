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

#include <stdlib.h>
#include "scout/Runtime/Settings.h"
#include "scout/Runtime/cpu/MeshThread.h"
#include "scout/Runtime/cpu/Queue.h"
#include "scout/Runtime/cpu/CpuUtilities.h"
#include "scout/Runtime/cpu/CpuRuntime.h"
using namespace scout;

namespace scout {
  namespace cpu {

    // each thread runs this which gets an block from the queue and invokes it.
    void MeshThread::run() {
      Block *block;
      BlockLiteral* bl;
      size_t qCurrent;
      size_t size;
      bool done;
      Settings *settings = Settings::Instance();

      if (settings->threadBind() == 1) system_->bindThreadInside();
      for (;;) {
        beginSem_.acquire();

        for (;;) {
          if (settings->workStealing() == 2) { //steal from all
            qCurrent = qIndex_;
            size = queueVec_.size();
            done = false;
            for(size_t i = 0; i < size; i++) {
              block = queueVec_[qCurrent]->get();
              if (block) break;
              qCurrent = (qCurrent+1) % queueVec_.size();
              if (i == size - 1) done = true;
            }
            if (done) break;
          } else if (settings->workStealing() == 1) { //steal from neighbors only
            qCurrent = qIndex_;
            block = queueVec_[qCurrent]->get();
            if (!block) {
              qCurrent = (qCurrent+1) % queueVec_.size();
              block = queueVec_[qCurrent]->get();
            }
            if (!block) {
              qCurrent = (qCurrent-1) % queueVec_.size();
              block = queueVec_[qCurrent]->get();
            }
            if (!block) break;
          } else { // default case, no stealing
            block = queueVec_[qIndex_]->get();
            if (!block) break;
          }

          bl = block->getBlockLiteral();

          bl->invoke(bl);

          delete(block);
        }
        finishSem_.release();
      }
    }
  } // end namespace cpu
} // end namespace scout
