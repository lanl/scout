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

#ifndef CLANG_CODEGEN_SCOUT_ASTVISTORS_H
#define CLANG_CODEGEN_SCOUT_ASTVISTORS_H

#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace clang::CodeGen;

// find what mesh fields are used in a forall
class ForallVisitor : public StmtVisitor<ForallVisitor> {
public:

  typedef std::map<std::string, bool> FieldMap;
  enum NodeType {
    NodeNone,
    NodeLHS,
    NodeRHS
  };

  ForallVisitor(ForallMeshStmt* fs)
  : fs_(fs),
    nodeType_(NodeNone) {
    (void)fs_; //suppress warning
  }

  const std::map<std::string, bool>& getLHSmap() const {
    return LHS_;
  }

  const std::map<std::string, bool>& getRHSmap() const {
    return RHS_;
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E) {
    VisitChildren(E);
  }

  void VisitChildren(Stmt* S) {
    if(S) {
      for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
        if (Stmt* child = *I) {
          Visit(child);
        }
      }
    }
  }

  void VisitMemberExpr(MemberExpr* E) {

    if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(E->getBase())) {

      if(ImplicitMeshParamDecl *bd = dyn_cast<ImplicitMeshParamDecl>(dr->getDecl())) {

        if (isa<MeshType>(bd->getType().getCanonicalType().getTypePtr())) {

          ValueDecl* md = E->getMemberDecl();

          std::string ref = bd->getMeshVarDecl()->getName().str() + "." + md->getName().str();

          if (nodeType_ == NodeLHS) {
            LHS_.insert(make_pair(ref, true));
            //llvm::errs() << "LHS " << ref << "\n";
          } else if (nodeType_ == NodeRHS) {
            RHS_.insert(make_pair(ref, true));
            //llvm::errs() << "RHS " << ref << "\n";
          }
        }
      }
    }
  }


  void VisitDeclStmt(DeclStmt* S) {
    VisitChildren(S);
  }

  void VisitBinaryOperator(BinaryOperator* S) {

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


private:

  ForallMeshStmt *fs_;
  FieldMap LHS_;
  FieldMap RHS_;
  NodeType nodeType_;
};


// find what mesh fields are used in a task function.
class TaskVisitor : public DeclVisitor<TaskVisitor> {
public:

  typedef std::map<std::string, bool> FieldMap;
  typedef std::map<std::string, std::string> MeshNameMap;

  TaskVisitor(const FunctionDecl *FD) { fd_ = FD; }

  const std::map<std::string, bool>& getLHSmap() const {
    return LHS_;
  }

  const std::map<std::string, bool>& getRHSmap() const {
    return RHS_;
  }

  const std::map<std::string, std::string>& getMeshNamemap() const {
    return MNM_;
  }

  void VisitChildren(Stmt* S) {
    if(S) {
      for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
        if (Stmt* child = *I) {
          if(ForallMeshStmt *FAMS = dyn_cast<ForallMeshStmt>(child)) {
            std::string MeshName = FAMS->getMeshVarDecl()->getName().str();
            std::string MeshTypeName =  FAMS->getMeshType()->getName().str();
            MNM_.insert(make_pair(MeshName, MeshTypeName));

            ForallVisitor v(FAMS);
            v.Visit(FAMS);

            FieldMap lhs = v.getLHSmap();
            LHS_.insert(lhs.begin(), lhs.end());
            FieldMap rhs = v.getRHSmap();
            RHS_.insert(rhs.begin(), rhs.end());
          }
          // look for function calls in task function
          if(CallExpr *CE = dyn_cast<CallExpr>(child)) {
            TaskVisitor v(CE->getDirectCallee());
            v.VisitStmt(CE->getDirectCallee()->getBody());
            FieldMap subLHS = v.getLHSmap();
            LHS_.insert(subLHS.begin(), subLHS.end());
            FieldMap subRHS = v.getRHSmap();
            RHS_.insert(subRHS.begin(), subRHS.end());
          }
        }
      }
    }
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }


private:

  const FunctionDecl *fd_;
  FieldMap LHS_;
  FieldMap RHS_;
  MeshNameMap MNM_;

};

#endif
