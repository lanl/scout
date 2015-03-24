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
#include "clang/Basic/Builtins.h"
#include "clang/Sema/SemaDiagnostic.h"
#include <map>

using namespace clang;
using namespace clang::sema;

namespace clang {
namespace sema {

enum ShiftKind {
  CShift,
  EOShift
};

bool isPosition(unsigned id);

bool isCShift(unsigned id);

bool isEOShift(unsigned id);

bool CheckShift(unsigned id, CallExpr *E, Sema &S);

// look for foralls outside of tasks
class NonTaskForallVisitor : public StmtVisitor<NonTaskForallVisitor> {
public:
  NonTaskForallVisitor(Sema& sema) :sema_(sema) {
  }

  void VisitStmt(Stmt* S) {
     VisitChildren(S);
   }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }

  void VisitForallMeshStmt(ForallMeshStmt *S) {
    sema_.Diag(S->getLocStart(), diag::err_forall_non_task);
    VisitChildren(S);
  }

private:
  Sema& sema_;
};


// ForAllVisitor class to check that LHS mesh field assignment
// operators do not appear as subsequent RHS values, and various other
// semantic checks
class ForallVisitor : public StmtVisitor<ForallVisitor> {
public:

  enum NodeType{
    NodeNone,
    NodeLHS,
    NodeRHS
  };

  ForallVisitor(Sema& sema, ForallMeshStmt* fs, bool isTask = false)
  : sema_(sema),
    fs_(fs),
    meshAccess_(false),
    error_(false),
    isTask_(isTask),
    nodeType_(NodeNone) {
    (void)fs_; //suppress warning
  }

  void VisitBinaryOperator(BinaryOperator* S);

  void VisitCallExpr(CallExpr* E);

  void VisitDeclStmt(DeclStmt* S);

  void VisitMemberExpr(MemberExpr* E);

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }

  bool getMeshAccess() {
    return meshAccess_;
  }

  bool error(){
    return error_;
  }

private:
  Sema& sema_;
  ForallMeshStmt *fs_;
  typedef std::map<std::string, bool> RefMap_;
  RefMap_ refMap_;
  RefMap_ localMap_;
  bool meshAccess_;
  bool error_;
  bool isTask_;
  NodeType nodeType_;
};


class RenderallVisitor : public StmtVisitor<RenderallVisitor> {
public:

  enum NodeType{
    NodeNone,
    NodeLHS,
    NodeRHS
  };

  RenderallVisitor(Sema& sema, RenderallMeshStmt* fs)
  : sema_(sema),
    fs_(fs),
    error_(false),
    nodeType_(NodeNone),
    foundColorAssign_(false) {
    for(size_t i = 0; i < 4; ++i) {
      foundComponentAssign_[i] = false;
    }

  }

  void VisitBinaryOperator(BinaryOperator* S);

  void VisitCallExpr(CallExpr* E);

  void VisitDeclStmt(DeclStmt* S);

  void VisitIfStmt(IfStmt* S);

  void VisitMemberExpr(MemberExpr* E);

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }

  bool isColorExpr(Expr *E) {
    if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E)) {
      if(isa<ImplicitColorParamDecl>(DR->getDecl())) {
        return true;
      }
    }
    return false;
  }

  bool foundColorAssign() {
    if(foundColorAssign_) {
      return true;
    }

    for(size_t i = 0; i < 4; ++i) {
      if(!foundComponentAssign_[i]) {
        return false;
      }
    }
    return true;
  }


  bool error(){
    return error_;
  }

private:
  Sema& sema_;
  RenderallMeshStmt *fs_;
  typedef std::map<std::string, bool> RefMap_;
  RefMap_ refMap_;
  RefMap_ localMap_;
  bool error_;
  NodeType nodeType_;
  bool foundColorAssign_;
  bool foundComponentAssign_[4];
};


// make sure task functions are pure
class TaskStmtVisitor : public StmtVisitor<TaskStmtVisitor> {
public:

  TaskStmtVisitor(Sema& sema, Stmt* S)
: S_(S), sema_(sema), meshAccess_(false) {
    (void)S_;
  }

  void VisitCallExpr(CallExpr* E);

  void VisitChildren(Stmt* S);

  void VisitDeclRefExpr(DeclRefExpr *E);

  void VisitForallMeshStmt(ForallMeshStmt *S);

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitDeclStmt(DeclStmt* S) {
    VisitChildren(S);
  }

  bool getMeshAccess() {
    return meshAccess_;
  }

private:
  Stmt *S_;
  Sema& sema_;
  bool meshAccess_;
};


class TaskDeclVisitor : public DeclVisitor<TaskDeclVisitor> {
public:

  TaskDeclVisitor(Sema& sema, FunctionDecl *FD)
: FD_(FD), sema_(sema), meshAccess_(false) {
  }

  void VisitStmt(Stmt* S);

private:
  FunctionDecl *FD_;
  Sema& sema_;
  bool meshAccess_;

};
  
} // end namespace sema
} // end namespace clang

#endif
