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
#include <map>

using namespace clang;

namespace {

enum ShiftKind {
  CShift,
  EOShift
};

//check if builtin id is a cshift
bool isCShift(unsigned id) {
  if (id == Builtin::BIcshift || id == Builtin::BIcshifti
      || id == Builtin::BIcshiftf || id == Builtin::BIcshiftd ) return true;
  return false;
}

//check if builtin id is an eoshift
bool isEOShift(unsigned id) {
  if (id == Builtin::BIeoshift || id == Builtin::BIeoshifti
      || id == Builtin::BIeoshiftf || id == Builtin::BIeoshiftd ) return true;
  return false;
}

bool CheckShift(unsigned id, CallExpr *E, Sema &S) {
  bool error = false;
  unsigned kind = 0;
  if (isCShift(id)) kind = ShiftKind::CShift;
  if (isEOShift(id)) kind = ShiftKind::EOShift;

  unsigned args = E->getNumArgs();

  // max number of args is 4 for cshift and 5 for eoshift
  if (args > kind + 4) {
    S.Diag(E->getRParenLoc(), diag::err_shift_args) << kind;
    error = true;
  } else {
    Expr* fe = E->getArg(0);

    if (ImplicitCastExpr* ce = dyn_cast<ImplicitCastExpr>(fe)) {
      fe = ce->getSubExpr();
    }

    if (MemberExpr* me = dyn_cast<MemberExpr>(fe)) {
      if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(me->getBase())) {
        ValueDecl* bd = dr->getDecl();
        const Type *T= bd->getType().getCanonicalType().getTypePtr();
        if (!isa<MeshType>(T)) {
          S.Diag(fe->getExprLoc(), diag::err_shift_field) << kind;
          error = true;
        }
        if(isa<StructuredMeshType>(T)) {
          S.Diag(fe->getExprLoc(), diag::err_shift_not_allowed) << kind << 0;
          error = true;
        }
        if(isa<UnstructuredMeshType>(T)) {
          S.Diag(fe->getExprLoc(), diag::err_shift_not_allowed) << kind << 1;
          error = true;
        }

      }
    } else {
      S.Diag(fe->getExprLoc(), diag::err_shift_field) << kind;
      error = true;
    }
    //disable this check for now as Jamal needs this functionality
#if 0
    // only allow integers for shift values
    for (unsigned i = kind+1; i < args; i++) {
      Expr *arg = E->getArg(i);
      // remove unary operator if it exists
      if(UnaryOperator *UO = dyn_cast<UnaryOperator>(arg)) {
        arg = UO->getSubExpr();
      }
      if(!isa<IntegerLiteral>(arg)) {
        S.Diag(arg->getExprLoc(), diag::err_shift_nonint) << kind;
        error = true;
      }
    }
#endif

  }
  return error;
}


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

  ForallVisitor(Sema& sema, ForallMeshStmt* fs)
  : sema_(sema),
    fs_(fs),
    error_(false),
    nodeType_(NodeNone) {
    (void)fs_; //suppress warning
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E) {

    FunctionDecl* fd = E->getDirectCallee();

    if (fd) {
      std::string name = fd->getName();
      unsigned id = fd->getBuiltinID();
      if (name == "printf" || name == "fprintf") {
        // SC_TODO -- for now we'll warn that you're calling a print
        // function inside a parallel construct -- in the long run
        // we can either (1) force the loop to run sequentially or
        // (2) replace print function with a "special" version...
        sema_.Diag(E->getExprLoc(), diag::warn_forall_calling_io_func);
      } else if (isCShift(id) || isEOShift(id)) {
        error_ = CheckShift(id, E, sema_);
      }
    }

    VisitChildren(E);
  }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }

  void VisitMemberExpr(MemberExpr* E) {

    if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(E->getBase())) {
      ValueDecl* bd = dr->getDecl();

      if (isa<MeshType>(bd->getType().getCanonicalType().getTypePtr())){

        ValueDecl* md = E->getMemberDecl();

        std::string ref = bd->getName().str() + "." + md->getName().str();

        if (nodeType_ == NodeLHS) {
          refMap_.insert(make_pair(ref, true));
        } else if (nodeType_ == NodeRHS) {
          RefMap_::iterator itr = refMap_.find(ref);
          if (itr != refMap_.end()) {
            sema_.Diag(E->getMemberLoc(), diag::err_rhs_after_lhs_forall);
            error_ = true;
          }
        }
      }
    }
  }


  void VisitDeclStmt(DeclStmt* S) {

    DeclGroupRef declGroup = S->getDeclGroup();

    for(DeclGroupRef::iterator itr = declGroup.begin(),
        itrEnd = declGroup.end(); itr != itrEnd; ++itr){
      Decl* decl = *itr;

      if (NamedDecl* nd = dyn_cast<NamedDecl>(decl)) {
        localMap_.insert(make_pair(nd->getName().str(), true));
      }
    }

    VisitChildren(S);
  }

  void VisitBinaryOperator(BinaryOperator* S){

    switch(S->getOpcode()){
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      if(DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S->getLHS())){
        RefMap_::iterator itr = localMap_.find(DR->getDecl()->getName().str());
        if(itr == localMap_.end()){

          sema_.Diag(DR->getLocation(),
              diag::warn_lhs_outside_forall) << DR->getDecl()->getName();
        }
      }

      nodeType_ = NodeLHS;
      break;
    default:
      break;
    }

    Visit(S->getLHS());
    nodeType_ = NodeRHS;
    Visit(S->getRHS());
    nodeType_ = NodeNone;
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
  bool error_;
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

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E) {

    FunctionDecl* fd = E->getDirectCallee();

    if (fd) {
      std::string name = fd->getName();
      unsigned id = fd->getBuiltinID();
      if (name == "printf" || name == "fprintf") {
        // SC_TODO -- for now we'll warn that you're calling a print
        // function inside a parallel construct -- in the long run
        // we can either (1) force the loop to run sequentially or
        // (2) replace print function with a "special" version...
        sema_.Diag(E->getExprLoc(), diag::warn_renderall_calling_io_func);
      } else if (isCShift(id) || isEOShift(id)) {
        error_ = CheckShift(id, E, sema_);
      }
    }

    VisitChildren(E);
  }

  void VisitChildren(Stmt* S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }

  void VisitMemberExpr(MemberExpr* E) {

    if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(E->getBase())) {
      ValueDecl* bd = dr->getDecl();

      if (isa<MeshType>(bd->getType().getCanonicalType().getTypePtr())){

        ValueDecl* md = E->getMemberDecl();

        std::string ref = bd->getName().str() + "." + md->getName().str();

        if (nodeType_ == NodeLHS) {
          refMap_.insert(make_pair(ref, true));
        } else if (nodeType_ == NodeRHS) {
          RefMap_::iterator itr = refMap_.find(ref);
          if (itr != refMap_.end()) {
            sema_.Diag(E->getMemberLoc(), diag::err_rhs_after_lhs_forall);
            error_ = true;
          }
        }
      }
    }
  }


  void VisitDeclStmt(DeclStmt* S) {

    DeclGroupRef declGroup = S->getDeclGroup();

    for(DeclGroupRef::iterator itr = declGroup.begin(),
        itrEnd = declGroup.end(); itr != itrEnd; ++itr){
      Decl* decl = *itr;

      if (NamedDecl* nd = dyn_cast<NamedDecl>(decl)) {
        localMap_.insert(make_pair(nd->getName().str(), true));
      }
    }

    VisitChildren(S);
  }

  bool isColorExpr(Expr *E) {
    if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E)) {
      if(isa<ImplicitColorParamDecl>(DR->getDecl())) {
        return true;
      }
    }
    return false;
  }

  void VisitBinaryOperator(BinaryOperator* S){
    switch(S->getOpcode()){
    case BO_Assign:
      if(S->getOpcode() == BO_Assign){

        // "color = " case
        if(isColorExpr(S->getLHS())) {
          foundColorAssign_ = true;
          // "color.{rgba} = " case
        } else if(ExtVectorElementExpr* VE = dyn_cast<ExtVectorElementExpr>(S->getLHS())) {
          if(isColorExpr(VE->getBase())) { // make sure Base is a color expr

            //only allow .r .g .b .a not combos like .rg
            if (VE->getNumElements() == 1) {
              const char *namestart = VE->getAccessor().getNameStart();
              // find the index for this accesssor name {r,g,b,a} -> {0,1,2,3}
              int idx = ExtVectorType::getAccessorIdx(*namestart);
              foundComponentAssign_[idx] = true;
            }
          }
        }
      } else {
        VisitChildren(S);
      }
      break;
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      if(DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S->getLHS())){
        RefMap_::iterator itr = localMap_.find(DR->getDecl()->getName().str());
        if(itr == localMap_.end()){

          sema_.Diag(DR->getLocation(),
              diag::warn_lhs_outside_forall) << DR->getDecl()->getName();
        }
      }

      nodeType_ = NodeLHS;
      break;
    default:
      break;
    }

    Visit(S->getLHS());
    nodeType_ = NodeRHS;
    Visit(S->getRHS());
    nodeType_ = NodeNone;
  }

  void VisitIfStmt(IfStmt* S) {
    size_t ic = 0;
    for(Stmt::child_iterator I = S->child_begin(),
        E = S->child_end(); I != E; ++I){

      if(Stmt* child = *I) {
        if(isa<CompoundStmt>(child) || isa<IfStmt>(child)) {
          RenderallVisitor v(sema_,fs_);
          v.Visit(child);
          if(v.foundColorAssign()) {
            foundColorAssign_ = true;
          } else {
            foundColorAssign_ = false;
            break;
          }
        } else {
          Visit(child);
        }
        ++ic;
      }
    }
    if(ic == 2) {
      foundColorAssign_ = false;
    }
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
: S_(S), sema_(sema) {
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }


  void VisitCallExpr(CallExpr* E) {

    FunctionDecl* fd = E->getDirectCallee();

    if (fd) {
      std::string name = fd->getName();
      if (name == "printf" || name == "fprintf") {
        // SC_TODO -- for now we'll warn that you're calling a print
        // function inside a task
        sema_.Diag(E->getExprLoc(), diag::warn_task_calling_io_func);
      }
    }

    VisitChildren(E);
  }



  void VisitChildren(Stmt* S) {
    if(S) {
      for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
        if (Stmt* child = *I) {
          if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(child)) {
            if(VarDecl *VD = dyn_cast<VarDecl>(dr->getDecl())) {
              if(VD->hasGlobalStorage() && !VD->getType().isConstQualified()) {
                sema_.Diag(S->getLocStart(), diag::err_nonpure_task_fuction);
              }
            }
          }
          Visit(child);
        }
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
      if(S) {
        for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
          if (Stmt* child = *I) {
            TaskStmtVisitor v(sema_, child);
            v.Visit(child);
          }
        }
      }
    }
  }

private:
  FunctionDecl *FD_;
  Sema& sema_;

};

} // end namespace
#endif
