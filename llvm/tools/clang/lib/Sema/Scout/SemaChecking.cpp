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
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/Scout/BuiltinsScout.h"
#include "scout/Config/defs.h"

using namespace clang;
using namespace sema;

bool Sema::CheckMeshParameterCall(unsigned BuiltinID, CallExpr *TheCall) {
  int diagarg = 0;
  switch (BuiltinID) {
  case Builtin::BIwidth:
    diagarg = MeshParameterOffset::WidthOffset;
    break;
  case Builtin::BIheight:
    diagarg = MeshParameterOffset::HeightOffset;
    break;
  case Builtin::BIdepth:
    diagarg = MeshParameterOffset::DepthOffset;
    break;
  case Builtin::BIrank:
    diagarg = MeshParameterOffset::RankOffset;
    break;
  default:
    break;
  }

  // check number of args
  if(TheCall->getNumArgs() > 1) {
    Diag(TheCall->getExprLoc(), diag::err_mesh_builtin_nargs) << diagarg;
    return false;
  }

  bool star = false;
  // if we have an argument check it
  if(TheCall->getNumArgs() == 1) {

    ArrayRef<Stmt*> children = TheCall->getRawSubExprs();
    assert(children.size() == 2 && "bad mesh builtin CallExpr");
    Expr *E = dyn_cast<Expr>(children[1]);

    // Get the expr we are after, might have leading casts and star
    Expr *EE = E;
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(EE)) {
      EE = ICE->getSubExpr();
      if(const UnaryOperator *UO = dyn_cast<UnaryOperator>(EE)) {
        star = true;
        EE = UO->getSubExpr();
        if (ImplicitCastExpr *ICE2 = dyn_cast<ImplicitCastExpr>(EE)) {
          EE = ICE2->getSubExpr();
        }
      }
    }

    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(EE);
    if(!DRE) {
      Diag(TheCall->getExprLoc(), diag::err_mesh_builtin_arg) << diagarg;
      return false;
    }
    QualType QT = DRE->getType().getTypePtr()->getPointeeType();

    // warn about missing star if called from function w/ mesh passed a ptr
    if(!star && !QT.isNull() && isa<MeshType>(QT)) {
      Diag(E->getExprLoc(), diag::warn_mesh_builtin_ptr) << diagarg;
    }

    if(!isa<MeshType>(DRE->getType()) && !QT.isNull() && !isa<MeshType>(QT)) {
      Diag(TheCall->getExprLoc(), diag::err_mesh_builtin_arg) << diagarg;
      return false;  // not mesh or mesh ptr
    }
    if(!isa<VarDecl>(DRE->getDecl())) {
      Diag(TheCall->getExprLoc(), diag::err_mesh_builtin_arg) << diagarg;
      return false; //not vardecl
    }
  }
  return true;
}

bool Sema::CheckPlotCall(unsigned BuiltinID, CallExpr *TheCall) {
#ifndef SCOUT_ENABLE_PLOT
  Diag(TheCall->getExprLoc(), diag::err_plot_disabled);
  return false;
#endif
  
  if(TheCall->getNumArgs() != 2) {
    Diag(TheCall->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected 2 args";
    return false;
  }
  
  auto argsBegin = TheCall->arg_begin();
  
  const MemberExpr* memberExpr = dyn_cast<MemberExpr>(*argsBegin);

  if(!memberExpr){
    Diag((*argsBegin)->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a member expression";
    return false;
  }
  
  const DeclRefExpr* base = dyn_cast<DeclRefExpr>(memberExpr->getBase());
  if(!base){
    Diag(memberExpr->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a mesh in member expression";
    return false;
  }
  
  const ValueDecl* vd = dyn_cast<ValueDecl>(base->getDecl());
  if(!vd){
    Diag(memberExpr->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a uniform mesh";
    return false;
  }
  
  const UniformMeshType* mt = dyn_cast<UniformMeshType>(vd->getType().getTypePtr());
  if(!mt){
    Diag(memberExpr->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a uniform mesh";
    return false;
  }
  
  ++argsBegin;
  
  const StringLiteral* plotTypeLiteral = dyn_cast<StringLiteral>(*argsBegin);
  if(!plotTypeLiteral){
    Diag((*argsBegin)->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a string plot type specifier, one of: 'bars'";
    return false;
  }
  
  std::string plotType = plotTypeLiteral->getString();
  
  if(plotType != "bars"){
    Diag(plotTypeLiteral->getExprLoc(), diag::err_invalid_plot_call) <<
    "expected a string plot type specifier, one of: 'bars'";
    return false;
  }
  
  return true;
}
