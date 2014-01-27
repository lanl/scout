/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
 */

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;


// ----- RenderallStmt
//
// Constructor for a Renderall statement w/out a predicate expression.
//
RenderallStmt::RenderallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation RenderallLocation,
                       Stmt* Body)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    RenderallKWLoc(RenderallLocation) {

  SubExprs[PREDICATE] = 0;
  SubExprs[BODY]      = Body;
}


// ----- RenderallStmt::RenderallStmt
//
// Constructor for a Renderall statement w/ a predicate expression.
//
RenderallStmt::RenderallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation RenderallLocation,
                       Stmt* Body,
                       Expr* Predicate,
                       SourceLocation LeftParenLoc, SourceLocation RightParenLoc)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    RenderallKWLoc(RenderallLocation),
    LParenLoc(LeftParenLoc), RParenLoc(RightParenLoc) {

  SubExprs[PREDICATE] = Predicate;
  SubExprs[BODY]      = Body;
}






// ----- RenderallMeshStmt
//
// Constructor for a renderall mesh statement w/out a predicate expression.
//
RenderallMeshStmt::RenderallMeshStmt(MeshElementType RefElement,
                                     IdentifierInfo* RefVarInfo,
                                     IdentifierInfo* MeshInfo,
                                     VarDecl* MeshVarDecl,
                                     const MeshType* MT,
                                     SourceLocation ForallLocation,
                                     Stmt *Body)
  : RenderallStmt(ForallMeshStmtClass,
                  RefVarInfo,
                  MeshInfo, MeshVarDecl,
                  ForallLocation, Body) {

    MeshElementRef = RefElement;
    MeshRefType    = MT;
  }


// ----- RenderallMeshStmt
//
// Constructor for a renderall mesh statement w/ a predicate expression.
//
RenderallMeshStmt::RenderallMeshStmt(MeshElementType RefElement,
                                     IdentifierInfo* RefVarInfo,
                                     IdentifierInfo* MeshInfo,
                                     VarDecl* MeshVarDecl,
                                     const MeshType* MT,
                                     SourceLocation RenderallLocation,
                                     Stmt *Body,
                                     Expr* Predicate,
                                     SourceLocation LeftParenLoc,
                                     SourceLocation RightParenLoc)
  : RenderallStmt(RenderallMeshStmtClass,
                  RefVarInfo,
                  MeshInfo, MeshVarDecl,
                  RenderallLocation, Body,
                  Predicate, LeftParenLoc, RightParenLoc) {

    MeshElementRef = RefElement;
    MeshRefType    = MT;
  }

bool RenderallMeshStmt::isUniformMesh() const {
  return MeshRefType->getTypeClass() == Type::UniformMesh;
}

bool RenderallMeshStmt::isRectilinearMesh() const {
  return MeshRefType->getTypeClass() == Type::RectilinearMesh;
}

bool RenderallMeshStmt::isStructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::StructuredMesh;
}

bool RenderallMeshStmt::isUnstructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::UnstructuredMesh;
}

