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

#ifndef SCOUT_CPU_RUNTIME_H_
#define SCOUT_CPU_RUNTIME_H_

#include "scout/Runtime/cpu/Queue.h"
#include "scout/Runtime/cpu/MeshThread.h"
#include <map>
using namespace std;
using namespace scout::cpu;

namespace scout {
  namespace cpu {

    void *createSubBlock(BlockLiteral * bl, size_t numDimensions,
                         size_t numFields);
    void deleteSubBlock(BlockLiteral * bl, size_t numDimensions);
    Item *createItem(BlockLiteral * bl, int numDimensions, int numFields,
                     size_t start, size_t end);
    void deleteItem(Item *item);
    size_t findExtent(BlockLiteral * bl, int numDimensions);

    class CpuRuntime {
    public:
      static CpuRuntime* Instance();

      ~CpuRuntime();

      void run(void *blockLiteral, int numDimensions, int numFields);

    protected:
      CpuRuntime();
      CpuRuntime(const CpuRuntime&);
      CpuRuntime& operator= (const CpuRuntime&);
      void queueBlocks(void *blockLiteral, int numDimensions,
                       int numFields);
      int nThreads();
      int nDomains();
      int blocksPerThread();
    private:
      static CpuRuntime* instance_;
      System *system_;
      QueueVec queueVec_;
      ThreadVec threadVec_;
      size_t nThreads_, nDomains_, nChunk_, blocksPerThread_;
    };
  }
}

#endif
