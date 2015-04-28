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
#include "clang/Basic/Builtins.h"

namespace clang {
namespace CodeGen {

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

        if (nodeType_ == NodeLHS) {
          LHS_.insert(make_pair(ref, true));
        } else if (nodeType_ == NodeRHS) {
          RHS_.insert(make_pair(ref, true));
        }
      }
    }
  }
}

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


void ForallVisitor::VisitCallExpr(CallExpr* E) {
  FunctionDecl* fd = E->getDirectCallee();

  // look for stencil call and then check inside it for shifts
  if (fd->isStencilSpecified()) {
    //llvm::errs() << "stencil call\n";
    VisitStmt(fd->getBody());
  }

  // look for cshift/eoshift
  if (fd) {
    unsigned id = fd->getBuiltinID();
    bool iscshift = isCShift(id);
    bool iseoshift = isEOShift(id);
    if (iscshift || iseoshift) {
      //llvm::errs() << "found shift\n";

      std::string name;
      unsigned args = E->getNumArgs();
      unsigned kind = 0;
      if (iscshift) kind = ShiftKind::CShift;
      if (iseoshift) kind = ShiftKind::EOShift;

      // first arg
      Expr* fe = E->getArg(0);
      if (ImplicitCastExpr* ce = dyn_cast<ImplicitCastExpr>(fe)) {
        fe = ce->getSubExpr();
      }

      if (MemberExpr* me = dyn_cast<MemberExpr>(fe)) {
        name = me->getMemberDecl()->getName();
        //llvm::errs() << "member " << name << "\n";
        if (mins_.find(name) == mins_.end()) {
          mins_[name] = {0,0,0};
          maxs_[name] = {0,0,0};
        }

      }

      // shift args
      std::vector<int> min = mins_[name];
      std::vector<int> max = maxs_[name];

      for (unsigned i = kind+1; i < args; i++) {
        unsigned j = i-kind-1;
        Expr *arg = E->getArg(i);
        // remove unary operator if it exists
        bool neg = false;
        if(UnaryOperator *UO = dyn_cast<UnaryOperator>(arg)) {
          if(UO->getOpcode() == UO_Minus) neg = true;
          arg = UO->getSubExpr();
        }
        if(IntegerLiteral *il = dyn_cast<IntegerLiteral>(arg)) {
          int val = il->getValue().getLimitedValue();
          if(neg) val = -val;

          if (val < min[j]) {
            //llvm::errs() << "min " << name << " " << val << "\n";
            min[j] = val;
            mins_[name] = min;
          }
          if (val > max[j]) {
            //llvm::errs() << "max " << name << " " << val << "\n";
            max[j] = val;
            maxs_[name] = max;
          }
          //llvm::errs() << "ghost size " << name << "[" << j << "] " << max[j]-min[j] << "\n";
        }
      }
    }
  }
  VisitChildren(E);
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


void TaskStmtVisitor::VisitCallExpr(CallExpr* E) {
  FunctionDecl* fd = E->getDirectCallee();

  // look for function calls in task function
  TaskDeclVisitor v(fd);
  if(fd->getBody()) {
    //look inside function
    v.VisitStmt(fd->getBody());
    MeshFieldMap subLHS = v.getLHSmap();
    LHS_.insert(subLHS.begin(), subLHS.end());
    MeshFieldMap subRHS = v.getRHSmap();
    RHS_.insert(subRHS.begin(), subRHS.end());

    VisitChildren(fd->getBody());
  } else {
    // we can't find the body (e.g library functions, builtins)
    // so look in the arguments for meshes
    for(unsigned i = 0; i < E->getNumArgs(); i++) {
      FunctionArgVisitor av(E->getArg(i));
      av.VisitStmt(E->getArg(i));
      MeshFieldMap subLHS = av.getLHSmap();
      LHS_.insert(subLHS.begin(), subLHS.end());
      MeshFieldMap subRHS = av.getRHSmap();
      RHS_.insert(subRHS.begin(), subRHS.end());
    }
  }
  VisitChildren(E);

}

void FunctionArgVisitor::VisitMemberExpr(MemberExpr *E) {
  if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(E->getBase())) {
    if(ImplicitMeshParamDecl *bd = dyn_cast<ImplicitMeshParamDecl>(DRE->getDecl())) {
      const Type *T = bd->getType().getCanonicalType().getTypePtr();
      if (isa<MeshType>(T)) {
        ValueDecl* md = E->getMemberDecl();
        std::string ref = bd->getMeshVarDecl()->getName().str() + "." + md->getName().str();
        //don't know what type of access this is just that it is an access.
        //llvm::errs() << "adding " << ref << "\n";
        LHS_.insert(make_pair(ref, true));
        RHS_.insert(make_pair(ref, true));
      }
    }
  }
}

void PlotExprVisitor::VisitDeclRefExpr(DeclRefExpr* E){
  const FrameDecl* FD = S_.getFrameDecl();
  VarDecl* VD = dyn_cast<VarDecl>(E->getDecl());
  if(VD && (FD->hasVar(VD) || S_.getVarId(VD) != 0)){
    isConstant_ = false;
  }
}

void PlotVarsVisitor::VisitDeclRefExpr(DeclRefExpr* E){
  VarDecl* VD = dyn_cast<VarDecl>(E->getDecl());
  if(VD && FD_->hasVar(VD)){
    varSet_.insert(VD);
  }
}
  
void PlotVarsVisitor::VisitScoutExpr(ScoutExpr* S){
  switch(S->kind()){
    case ScoutExpr::SpecObject:{
      SpecObjectExpr* o = static_cast<SpecObjectExpr*>(S);
      
      auto m = o->memberMap();
      for(auto& itr : m){
        Visit(itr.second.second);
      }
      
      break;
    }
    case ScoutExpr::SpecValue:{
      SpecValueExpr* o = static_cast<SpecValueExpr*>(S);
      Visit(o->getExpression());
      break;
    }
    case ScoutExpr::SpecArray:{
      SpecArrayExpr* o = static_cast<SpecArrayExpr*>(S);
      
      auto v = o->elements();
      
      for(size_t i = 0; i < v.size(); ++i){
        Visit(v[i]);
      }
      
      break;
    }
    default:
      assert(false && "unimplemented");
  }
}
  
} // end namespace CodeGen
} // end namespace clang
