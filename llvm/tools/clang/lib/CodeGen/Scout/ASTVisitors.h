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

#ifndef CLANG_CODEGEN_SCOUT_ASTVISTORS_H
#define CLANG_CODEGEN_SCOUT_ASTVISTORS_H

#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace clang::CodeGen;

namespace clang {
namespace CodeGen {

enum ShiftKind {
  CShift,
  EOShift
};

typedef std::map<std::string, bool> MeshFieldMap;
typedef std::map<std::string, std::string> MeshNameMap;
typedef std::map<std::string, std::vector<int>> MeshShiftMap;

// find what mesh fields are used in a forall
class ForallVisitor : public StmtVisitor<ForallVisitor> {
public:

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

  void VisitBinaryOperator(BinaryOperator* S);

  void VisitChildren(Stmt* S);

  void VisitMemberExpr(MemberExpr* E);

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E);

  void VisitDeclStmt(DeclStmt* S) {
     VisitChildren(S);
   }

private:
  ForallMeshStmt *fs_;
  MeshFieldMap LHS_;
  MeshFieldMap RHS_;
  NodeType nodeType_;
  MeshShiftMap mins_;
  MeshShiftMap maxs_;
};

class TaskStmtVisitor : public StmtVisitor<TaskStmtVisitor> {
public:

  TaskStmtVisitor(Stmt* S)
: S_(S) {
    (void)S_;
  }


  const std::map<std::string, bool>& getLHSmap() const {
    return LHS_;
  }

  const std::map<std::string, bool>& getRHSmap() const {
    return RHS_;
  }

  const std::map<std::string, std::string>& getMeshNamemap() const {
    return MNM_;
  }

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitCallExpr(CallExpr* E);

  void VisitDeclStmt(DeclStmt* S) {
    VisitChildren(S);
  }

  void VisitChildren(Stmt* S);

  void VisitForallMeshStmt(ForallMeshStmt *S);
private:
  Stmt *S_;
  MeshFieldMap LHS_;
  MeshFieldMap RHS_;
  MeshNameMap MNM_;
};

// find what mesh fields are used in a task function.
class TaskDeclVisitor : public DeclVisitor<TaskDeclVisitor> {
public:


  TaskDeclVisitor(const FunctionDecl *FD) { fd_ = FD; }

  const std::map<std::string, bool>& getLHSmap() const {
    return LHS_;
  }

  const std::map<std::string, bool>& getRHSmap() const {
    return RHS_;
  }

  const std::map<std::string, std::string>& getMeshNamemap() const {
    return MNM_;
  }

  void VisitStmt(Stmt* S);

private:
  const FunctionDecl *fd_;
  MeshFieldMap LHS_;
  MeshFieldMap RHS_;
  MeshNameMap MNM_;

};


//look in function argument for a mesh access
class FunctionArgVisitor : public StmtVisitor<FunctionArgVisitor> {
public:
  FunctionArgVisitor(Stmt* S)
  : S_(S) {
      (void)S_;
    }

  const std::map<std::string, bool>& getLHSmap() const {
    return LHS_;
  }

  const std::map<std::string, bool>& getRHSmap() const {
    return RHS_;
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

  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }

  void VisitMemberExpr(MemberExpr* E);

private:
  Stmt *S_;
  MeshFieldMap LHS_;
  MeshFieldMap RHS_;
};

class PlotExprVisitor : public StmtVisitor<PlotExprVisitor> {
public:
  
  PlotExprVisitor(const PlotStmt& S)
  : S_(S), isConstant_(true){}
  
  void VisitDeclRefExpr(DeclRefExpr* E);
  
  bool isConstant() const{
    return isConstant_;
  }
  
  void VisitChildren(Stmt* S) {
    if(S){
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){
        if(Stmt* child = *I){
          Visit(child);
        }
      }
    }
  }
  
  void VisitStmt(Stmt* S) {
    VisitChildren(S);
  }
  
private:
  const PlotStmt& S_;
  bool isConstant_;
};
  
class PlotVarsVisitor : public StmtVisitor<PlotVarsVisitor> {
public:
  typedef std::set<VarDecl*> VarSet;
  
  typedef std::set<CallExpr*> CallSet;
  
  PlotVarsVisitor(const PlotStmt& S)
  : S_(S){}
  
  void VisitDeclRefExpr(DeclRefExpr* E);
  
  void VisitCallExpr(CallExpr* E);
  
  void VisitScoutExpr(ScoutExpr* S);
  
  void VisitChildren(Stmt* S){
    if(S){
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){
        if(Stmt* child = *I){
          Visit(child);
        }
      }
    }
  }
  
  void VisitStmt(Stmt* S){
    VisitChildren(S);
  }
  
  const VarSet& getVarSet() const{
    return varSet_;
  }
  
  const CallSet& getCallSet() const{
    return callSet_;
  }
  
private:
  VarSet varSet_;
  CallSet callSet_;
  const PlotStmt& S_;
};
  
} // end namespace CodeGen
} // end namespace clang

#endif
