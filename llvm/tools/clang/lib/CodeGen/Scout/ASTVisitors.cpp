/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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

#include "ASTVisitors.h"


void ForallVisitor::VisitBinaryOperator(BinaryOperator* S) {

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

void ForallVisitor::VisitChildren(Stmt* S) {
  if(S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }
}

void ForallVisitor::VisitMemberExpr(MemberExpr* E) {
  if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(E->getBase())) {
    if(ImplicitMeshParamDecl *bd = dyn_cast<ImplicitMeshParamDecl>(dr->getDecl())) {
      if (isa<MeshType>(bd->getType().getCanonicalType().getTypePtr())) {
        ValueDecl* md = E->getMemberDecl();

        std::string ref = bd->getMeshVarDecl()->getName().str() + "." + md->getName().str();
        //llvm::errs() << "ref " << ref << "\n";
        if (nodeType_ == NodeLHS) {
          LHS_.insert(make_pair(ref, true));
        } else if (nodeType_ == NodeRHS) {
          RHS_.insert(make_pair(ref, true));
        }
      }
    }
  }
}


void TaskDeclVisitor::VisitStmt(Stmt* S) {
  if(S) {
    TaskStmtVisitor v(S);
    v.Visit(S);
    MeshFieldMap lhs = v.getLHSmap();
    LHS_.insert(lhs.begin(), lhs.end());
    MeshFieldMap rhs = v.getRHSmap();
    RHS_.insert(rhs.begin(), rhs.end());
    MeshNameMap mnm = v.getMeshNamemap();
    MNM_.insert(mnm.begin(), mnm.end());
  }
}


void TaskStmtVisitor::VisitChildren(Stmt* S) {
  if(S) {
    for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
      if (Stmt* child = *I) {
        Visit(child);
      }
    }
  }
}

void TaskStmtVisitor::VisitForallMeshStmt(ForallMeshStmt *S) {
  std::string MeshName = S->getMeshVarDecl()->getName().str();
  std::string MeshTypeName =  S->getMeshType()->getName().str();
  MNM_.insert(make_pair(MeshName, MeshTypeName));

  ForallVisitor v(S);
  v.Visit(S);

  MeshFieldMap lhs = v.getLHSmap();
  LHS_.insert(lhs.begin(), lhs.end());
  MeshFieldMap rhs = v.getRHSmap();
  RHS_.insert(rhs.begin(), rhs.end());

  VisitChildren(S);
}

// look for function calls in task function
void TaskStmtVisitor::VisitCallExpr(CallExpr* E) {

  TaskDeclVisitor v(E->getDirectCallee());
  v.VisitStmt(E->getDirectCallee()->getBody());
  MeshFieldMap subLHS = v.getLHSmap();
  LHS_.insert(subLHS.begin(), subLHS.end());
  MeshFieldMap subRHS = v.getRHSmap();
  RHS_.insert(subRHS.begin(), subRHS.end());

  VisitChildren(E->getDirectCallee()->getBody());
  VisitChildren(E);

}


