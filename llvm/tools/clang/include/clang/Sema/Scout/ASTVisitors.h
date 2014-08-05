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

#ifndef CLANG_SEMA_SCOUT_ASTVISTORS_H
#define CLANG_SEMA_SCOUT_ASTVISTORS_H

#include "clang/Sema/Sema.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;

// make sure task functions are pure
class TaskStmtVisitor : public StmtVisitor<TaskStmtVisitor> {
public:

  TaskStmtVisitor(Sema& sema, Stmt* S)
  : S_(S), sema_(sema) {
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E) {
    VisitChildren(E);
  }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(child)) {
          if(VarDecl *VD = dyn_cast<VarDecl>(dr->getDecl())) {
            if(VD->hasGlobalStorage()) {
              sema_.Diag(S->getLocStart(), diag::err_nonpure_task_fuction);
            }
          }
        }
        Visit(child);
      }
    }
  }

  void VisitDeclStmt(DeclStmt* S) {
    VisitChildren(S);
  }


private:
  Stmt *S_;
  Sema& sema_;
};


class TaskVisitor : public DeclVisitor<TaskVisitor> {
public:

  TaskVisitor(Sema& sema, FunctionDecl *FD)
  : FD_(FD), sema_(sema) {
  }

  void Visit(Stmt* S) {
    if(FD_->isTaskSpecified()) {
      for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
        if (Stmt* child = *I) {
          TaskStmtVisitor v(sema_, child);
          v.Visit(child);
        }
      }
    }
  }

private:
  FunctionDecl *FD_;
  Sema& sema_;

};

#endif
