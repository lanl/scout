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
#include "scout/Runtime/Settings.h"
#include "scout/Runtime/cpu/Block.h"
#include "scout/Runtime/cpu/CpuRuntime.h"
#include "scout/Runtime/cpu/CpuUtilities.h"
#include "scout/Runtime/cpu/Queue.h"
#include "scout/Runtime/cpu/MeshThread.h"
#include <iostream>
#include <cstring>
#include <cassert>

using namespace std;

namespace scout {

  // hook used in llvm/tools/clang/lib/CodeGen/CGBlocks.cpp
  extern "C"
  void __sc_queue_block(void *blockLiteral, int numDimensions,
      int numFields) {

    CpuRuntime *cpuRuntime = CpuRuntime::Instance();
    cpuRuntime->run(blockLiteral, numDimensions, numFields);
  }

  namespace cpu {

    CpuRuntime* CpuRuntime::instance_=0;

    CpuRuntime* CpuRuntime::Instance() {
      if (instance_ == 0) {
        instance_ = new CpuRuntime();
      }
      return instance_;
    }

    CpuRuntime::CpuRuntime() {
      int val;
      Settings *settings = Settings::Instance();
      system_ = new System();
      nThreads_ = system_->nThreads();
      nDomains_ = system_->nDomains();

      val = settings->blocksPerThread();
      if (val) blocksPerThread_ = val;
      else blocksPerThread_ = 4;
      if (settings->debug()) cerr << "blocksPerThread " << blocksPerThread_ << endl;

      // setup queues
      for(size_t i = 0; i < nDomains_; i++) {
        Queue* queue = new Queue;
        queueVec_.push_back(queue);
      }

      //start threads
      for (size_t i = 0; i < nThreads_; i++) {
        MeshThread* ti = new MeshThread(system_, queueVec_);
        ti->start();
        if (settings->threadBind() == 2) system_->bindThreadOutside(ti->thread());
        threadVec_.push_back(ti);
      }
      delete system_;
    }

    CpuRuntime::~CpuRuntime() {

      for (size_t i = 0; i < nThreads_; i++) {
        threadVec_[i]->stop();
      }
      for (size_t i = 0; i < nThreads_; i++) {
        threadVec_[i]->await();
        delete threadVec_[i];
      }
      for(size_t i = 0; i < nDomains_; i++) {
        delete queueVec_[i];
      }
    }

    void CpuRuntime::queueBlocks(void* blockLiteral, int numDimensions, int numFields) {
      BlockLiteral* bl = (BlockLiteral*) blockLiteral;
      size_t count, extent, chunk, end;

      extent = findExtent(bl, numDimensions); // find product of dimensions
      chunk = extent / (threadVec_.size() * blocksPerThread_);
      nChunk_ = extent / chunk;
      if (extent % nChunk_) {
        nChunk_++;
      }
      count = 0;
      for (size_t i = 0; i < extent; i += chunk) {
        end = i + chunk;

        if (end > extent) {
          end = extent;
        }

        Block* block = new Block(bl, numDimensions, numFields, i, end);

        // One queue for each numa domain
        queueVec_[count++ * nDomains_ / nChunk_]->add(block);
      }
    }

    void CpuRuntime::run(void* blockLiteral, int numDimensions, int numFields) {
      queueBlocks(blockLiteral, numDimensions, numFields);
      size_t n = threadVec_.size();

      for (size_t i = 0; i < n; i++) {  // release each beginSem
        threadVec_[i]->begin(i*nDomains_/nThreads_);
      }
      //MeshThread::run() in each thread till queue empty
      for (size_t i = 0; i < n; i++) { //acquire each finishSem
        threadVec_[i]->finish();
      }
      for (size_t i = 0; i < queueVec_.size(); i++) {
        queueVec_[i]->reset();
      }
    }
  } // end namespace cpu
} // end namespace scout
