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
 * Notes: See the various mesh types in AST/Types.h for some
 * -----  more details on Scout's mesh types.  It is important
 *        to keep a connection between the various Decls in this
 *        file and those types.
 *
 * #####
 */

#include "clang/Analysis/CFG.h"

#include <unordered_set>

namespace clang {

  class LastRenderall{
  public:
    typedef std::unordered_set<CFGBlock*> BlockSet;
    
    static void Run(Sema& S, FunctionDecl* fd){
      CFG* cfg =
      CFG::buildCFG(fd, fd->getBody(), &S.Context, CFG::BuildOptions());

      BlockSet vs;
      Visit(cfg, vs, cfg->getEntry());
      
      delete cfg;
    }
    
    static void Visit(CFG* cfg,
                      BlockSet& vs,
                      CFGBlock& block,
                      const RenderallMeshStmt* last=0){
      
      vs.insert(&block);
      
      for(auto itr = block.begin(), itrEnd = block.end();
          itr != itrEnd; ++itr){
        
        CFGStmt s = itr->castAs<CFGStmt>();
        const Stmt* stmt = s.getStmt();
        
        const RenderallMeshStmt* rs = dyn_cast<RenderallMeshStmt>(stmt);
        if(rs){
          last = rs;
        }
      }
      
      if(block.succ_empty()){
        if(last){
          last->setLast(true);
        }
        
        return;
      }
      
      for(auto itr = block.succ_begin(), itrEnd = block.succ_end();
          itr != itrEnd; ++itr){
        CFGBlock::AdjacentBlock b = *itr;
        
        CFGBlock* block = b.getReachableBlock();
        
        if(block && vs.find(block) == vs.end()){
          Visit(cfg, vs, *block, last);
        }
      }
    }
  };
  
} // end namespace clang
