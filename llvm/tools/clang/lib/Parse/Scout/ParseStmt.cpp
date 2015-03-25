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
#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/AST/Scout/MeshDecls.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/SmallString.h"

#include "clang/Sema/Lookup.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

// ----- LookupScoutVarDecl
//
VarDecl* Parser::LookupScoutVarDecl(IdentifierInfo *Info,
                                    SourceLocation Loc) {

  LookupResult Lookup(Actions, Info, Loc, Sema::LookupOrdinaryName);
  Actions.LookupName(Lookup, getCurScope());

  if (Lookup.getResultKind() != LookupResult::Found) {
    Diag(Loc, diag::err_unknown_mesh_or_query_variable) << Info;
    return 0;
  }

  NamedDecl* ND = Lookup.getFoundDecl();
  if (!isa<VarDecl>(ND)) {
    Diag(Loc, diag::err_unknown_mesh_or_query_variable) << Info;
    SkipUntil(tok::semi);
    return 0;
  }

  return cast<VarDecl>(ND);
}


// ----- LookupMeshType
//
const MeshType* Parser::LookupMeshType(IdentifierInfo *MeshInfo,
                                       SourceLocation MeshLoc) {

  VarDecl* VD = LookupScoutVarDecl(MeshInfo, MeshLoc);
  if (VD == 0)
    return 0;
  else {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();
    if (!T->isMeshType()) {
      T = VD->getType().getCanonicalType().getNonReferenceType().getTypePtr();
      if(!T->isMeshType()) {
        return 0;
      }
    }

    return const_cast<MeshType *>(cast<MeshType>(T));
  }
}
#

// ---- LookupMeshType
//
const MeshType* Parser::LookupMeshType(VarDecl *VD,
                                       IdentifierInfo *MeshInfo) {
  assert(VD != 0 && "null var decl passed for mesh type lookup");
  if (VD) {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();

    while(T->isPointerType() || T->isReferenceType()) {
      T = T->getPointeeType().getTypePtr();
    }
    
    if (!T->isMeshType()) {
      return 0;
    } else {
      return const_cast<MeshType *>(cast<MeshType>(T));
    }
  }

  return 0;
}

// ---- LookupQueryType
//
const QueryType* Parser::LookupQueryType(VarDecl *VD,
                                         IdentifierInfo *QueryInfo) {
  assert(VD != 0 && "null var decl passed for query type lookup");
  if (VD) {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();
    
    while(T->isPointerType() || T->isReferenceType()) {
      T = T->getPointeeType().getTypePtr();
    }
    
    if (!T->isScoutQueryType()) {
      return 0;
    } else {
      return const_cast<QueryType *>(cast<QueryType>(T));
    }
  }
  
  return 0;
}

static ForallMeshStmt::MeshElementType setMeshElementType(tok::TokenKind tkind) {

  switch (tkind) {

    case tok::kw_cells:
      return ForallMeshStmt::Cells;

    case tok::kw_vertices:
      return ForallMeshStmt::Vertices;

    case tok::kw_edges:
      return ForallMeshStmt::Edges;

    case tok::kw_faces:
      return ForallMeshStmt::Faces;
      break;

    default:
      return ForallMeshStmt::Undefined;
  }
}

// SC_TODO: Probably want to rename warning values
// e.g. 'diag::warn_mesh_has_no_cell_fields'.s..
void Parser::MeshElementTypeDiag(int MeshElementType,
    const MeshType *RefMeshType, SourceLocation MeshIdentLoc) {
  switch(MeshElementType) {

  case ForallMeshStmt::Cells:
    if (! RefMeshType->hasCellData())
      Diag(MeshIdentLoc, diag::warn_mesh_has_no_cell_fields);
    break;

  case ForallMeshStmt::Vertices:
    if (! RefMeshType->hasVertexData())
      Diag(MeshIdentLoc, diag::warn_mesh_has_no_vertex_fields);
    break;

  case ForallMeshStmt::Edges:
    if (! RefMeshType->hasEdgeData())
      Diag(MeshIdentLoc, diag::warn_mesh_has_no_edge_fields);
    break;

  case ForallMeshStmt::Faces:
    if (! RefMeshType->hasFaceData())
      Diag(MeshIdentLoc, diag::warn_mesh_has_no_face_fields);
    break;

  default:
    llvm_unreachable("unhandled/unrecognized mesh element type in case");
    break;
  }
}

// +---- Parse a forall statement operating on a mesh ------------------------+
//
//  forall [cells|edges|vertices|faces] element-id in mesh-instance {
//         ^
//          'Tok' should point here upon entering.
//    ...
//  }
//
//  where the identifier 'element-id' represents each instance of the
//  cell|edge|vertex|face in the mesh 'mesh-instance'.
//
// **Note - 'element-id' can become implicit within the loop body in
// terms of accessing mesh fields (stored at the given element
// location).
//
// Upon entering the current token should be on the mesh element kind
// (cells, vertices, edges, faces).
//
StmtResult Parser::ParseForallMeshStatement(ParsedAttributes &attrs) {

  // Upon entry we expect the input token to be on the 'forall'
  // keyword -- we'll throw an assertion in just to make sure
  // we help maintain consistency from the caller(s).
  assert(Tok.getKind() == tok::kw_forall && "expected input token to be 'forall'");

  // Swallow the forall token...
  SourceLocation ForallKWLoc = ConsumeToken();

  // At this point we should be sitting at the mesh element keyword
  // that identifies the locations on the mesh that are to be computed
  // over.  Keep track of the element token and its location (for later
  // use).  Also set the mesh element type we're processing so we can
  // refer to it later w/out having to query/translate token types...
  tok::TokenKind ElementToken = Tok.getKind();
  ConsumeToken();

  ForallMeshStmt::MeshElementType MeshElementType = setMeshElementType(ElementToken);
  if (MeshElementType == ForallMeshStmt::Undefined) {
    Diag(Tok, diag::err_forall_expected_mesh_element_kw);
    SkipUntil(tok::semi);
    return StmtError();
  }

  unsigned ScopeFlags = Scope::BreakScope    |
                        Scope::ContinueScope |
                        Scope::DeclScope     |
                        Scope::ControlScope;

  ParseScope ForallScope(this, ScopeFlags);

  // We consumed the element token above and should now be
  // at the element identifier portion of the forall; make
  // sure we have a valid identifier and bail if not...
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo *ElementIdentInfo = Tok.getIdentifierInfo();
  SourceLocation  ElementIdentLoc  = Tok.getLocation();
  ConsumeToken();

  // Next we should encounter the 'in' keyword...
  if (Tok.isNot(tok::kw_in)) {
    Diag(Tok, diag::err_forall_expected_kw_in);
    SkipUntil(tok::semi);
    return StmtError();
  }
  ConsumeToken();

  //if we are in scc-mode and in a function where the mesh was
  // passed as a parameter we will have a star here.
  bool meshptr = false;
  if(getLangOpts().ScoutC) {
    if(Tok.is(tok::star)) {
      ConsumeToken();
      meshptr = true;
    }
  }

  // Finally, we are at the identifier that specifies the mesh
  // that we are computing over.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo  *IdentInfo = Tok.getIdentifierInfo();
  SourceLocation   IdentLoc  = Tok.getLocation();

  VarDecl *VD = LookupScoutVarDecl(IdentInfo, IdentLoc);

  if (VD == 0)
    return StmtError();
  
  DeclStmt* Init = NULL; //declstmt for forall implicit variable
  
  VarDecl* QD = 0;
  
  const MeshType *RefMeshType = LookupMeshType(VD, IdentInfo);
  if(RefMeshType){
    // If we are in scc-mode and inside a function then make sure
    // we have a *
    if(getLangOpts().ScoutC && isa<ParmVarDecl>(VD) && meshptr == false) {
      Diag(Tok,diag::err_expected_star_mesh);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    bool success = Actions.ActOnForallMeshRefVariable(getCurScope(),
                                                      IdentInfo, IdentLoc,
                                                      MeshElementType,
                                                      ElementIdentInfo,
                                                      ElementIdentLoc,
                                                      RefMeshType,
                                                      VD, &Init);
    if (! success)
      return StmtError();
    
    MeshElementTypeDiag(MeshElementType, RefMeshType, IdentLoc);
  }
  else{
    const QueryType *RefQueryType = LookupQueryType(VD, IdentInfo);
    if(RefQueryType){
      QD = VD;
      
      const QueryExpr* QE = dyn_cast<QueryExpr>(QD->getInit());
      assert(QE && "expected a query expression");

      const MemberExpr* memberExpr = QE->getField();
      const DeclRefExpr* base = dyn_cast<DeclRefExpr>(memberExpr->getBase());
      assert(base && "expected a DeclRefExpr");
      
      const ImplicitMeshParamDecl* imp =
      dyn_cast<ImplicitMeshParamDecl>(base->getDecl());
      
      assert(base && "expected an ImplicitMeshParamDecl");
      
      VD = imp->getMeshVarDecl();
      
      QualType qt = VD->getType();
      RefMeshType = dyn_cast<MeshType>(qt.getTypePtr());
      assert(RefMeshType && "expected a mesh type");
    }
    else{
      Diag(IdentLoc, diag::err_expected_a_mesh_or_query_type);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }
  
  ConsumeToken();

  // Now check to see if we have a predicate/conditional expression...
  //
  // SC_TODO - do we need to validate the predicate is really a
  //           conditional expression?
  Expr *Predicate = 0;
  SourceLocation LParenLoc, RParenLoc;

  if (Tok.is(tok::kw_where)) {
    ConsumeToken();

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_lparen);
      // Multi-line skip, don't consume brace
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    LParenLoc = ConsumeParen();

    ExprResult PredicateResult = ParseExpression();
    if (PredicateResult.isInvalid()) {
      Diag(Tok, diag::err_forall_invalid_op);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    Predicate = PredicateResult.get();

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_rparen);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }
    RParenLoc = ConsumeParen();
  }

  //SourceLocation BodyLoc = Tok.getLocation();
  StmtResult BodyResult(ParseStatement());

  if (BodyResult.isInvalid()) {
    // SC_TODO -- is this a useful diagnostic?
    Diag(Tok, diag::err_invalid_forall_body);
    SkipUntil(tok::semi);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();
  StmtResult ForallResult = Actions.ActOnForallMeshStmt(ForallKWLoc,
                                                        MeshElementType,
                                                        RefMeshType, VD,
                                                        ElementIdentInfo,
                                                        IdentInfo,
                                                        LParenLoc,
                                                        Predicate,
                                                        RParenLoc,
                                                        Init, QD, Body);
  if (! ForallResult.isUsable())
    return StmtError();

  return ForallResult;
}

static RenderallMeshStmt::MeshElementType
setRenderallMeshElementType(tok::TokenKind tkind) {

  switch (tkind) {

    case tok::kw_cells:
      return RenderallMeshStmt::Cells;

    case tok::kw_vertices:
      return RenderallMeshStmt::Vertices;

    case tok::kw_edges:
      return RenderallMeshStmt::Edges;

    case tok::kw_faces:
      return RenderallMeshStmt::Faces;
      break;

    default:
      return RenderallMeshStmt::Undefined;
  }
}


// ----- LookupRenderTargetVarDecl
//
VarDecl* Parser::LookupRenderTargetVarDecl(IdentifierInfo *TargetInfo,
                                           SourceLocation TargetLoc) {

  LookupResult TargetLookup(Actions, TargetInfo, TargetLoc, Sema::LookupOrdinaryName);
  Actions.LookupName(TargetLookup, getCurScope());

  if (TargetLookup.getResultKind() != LookupResult::Found) {
    Diag(TargetLoc, diag::err_unknown_render_target_variable) << TargetInfo;
    return 0;
  }

  NamedDecl* ND = TargetLookup.getFoundDecl();
  if (!isa<VarDecl>(ND)) {
    Diag(TargetLoc, diag::err_expected_a_target_type) << TargetInfo;
    return 0;
  }

  return cast<VarDecl>(ND);
}

const RenderTargetType* Parser::LookupRenderTargetType(IdentifierInfo *TargetInfo,
                                                       SourceLocation TargetLoc) {
  
  VarDecl* VD = LookupRenderTargetVarDecl(TargetInfo, TargetLoc);
  if (VD == 0)
    return 0;
  else {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();

    if (!isa<RenderTargetType>(T)) {
      T = VD->getType().getCanonicalType().getNonReferenceType().getTypePtr();
      if(!isa<RenderTargetType>(T)) {
        return 0;
      }
    }
    
    return const_cast<RenderTargetType *>(cast<RenderTargetType>(T));
  }
}


// +---- Parse a renderall statement operating on a mesh ---------------------+
//
//  renderall [cells|edges|vertices|faces] element-id in mesh-instance {
//            ^
//             'Tok' should point here upon entering.
//    ...
//  }
//
//  where the identifier 'element-id' represents each instance of the
//  cell|edge|vertex|face in the mesh 'mesh-instance'.
//
// **Note - 'element-id' can become implicit within the loop body in
// terms of accessing mesh fields (stored at the given element
// location).
//
// Upon entering the current token should be on the mesh element kind
// (cells, vertices, edges, faces).
//
StmtResult Parser::ParseRenderallMeshStatement(ParsedAttributes &attrs) {

  // Upon entry we expect the input token to be on the 'renderall'
  // keyword -- we'll throw an assertion in just to make sure
  // we help maintain consistency from the caller(s).
  assert(Tok.getKind() == tok::kw_renderall && "expected 'renderall' token");

  // Swallow the renderall token...
  SourceLocation RenderallKWLoc = ConsumeToken();

  // At this point we should be sitting at the mesh element keyword
  // that identifies the locations on the mesh that are to be computed
  // over.  Keep track of the element token and its location (for later
  // use).  Also set the mesh element type we're processing so we can
  // refer to it later w/out having to query/translate token types...
  tok::TokenKind ElementToken = Tok.getKind();
  ConsumeToken();

  RenderallMeshStmt::MeshElementType MeshElementType;
  MeshElementType = setRenderallMeshElementType(ElementToken);
  if (MeshElementType == RenderallMeshStmt::Undefined) {
    Diag(Tok, diag::err_renderall_expected_mesh_element_kw);
    SkipUntil(tok::semi);
    return StmtError();
  }

  unsigned ScopeFlags = Scope::BreakScope    |
                        Scope::ContinueScope |
                        Scope::DeclScope     |
                        Scope::ControlScope;

  ParseScope RenderallScope(this, ScopeFlags);

  // We consumed the element token above and should now be at the
  // element identifier portion of the renderall; make sure we have a
  // valid identifier and bail if not...
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo *ElementIdentInfo = Tok.getIdentifierInfo();
  //SourceLocation  ElementIdentLoc  = Tok.getLocation();
  ConsumeToken();

  // Next we should encounter the 'in' keyword...
  if (Tok.isNot(tok::kw_in)) {
    Diag(Tok, diag::err_forall_expected_kw_in);
    SkipUntil(tok::semi);
    return StmtError();
  }
  ConsumeToken();

  //if we are in scc-mode and in a function where the mesh was
  // passed as a parameter we will have a star here.
  bool meshptr = false;
  if(getLangOpts().ScoutC) {
    if(Tok.is(tok::star)) {
      ConsumeToken();
      meshptr = true;
    }
  }

  // Next, we are at the identifier that specifies the mesh
  // that we are computing over.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  IdentifierInfo  *MeshIdentInfo = Tok.getIdentifierInfo();
  SourceLocation   MeshIdentLoc  = Tok.getLocation();

  VarDecl *VD = LookupScoutVarDecl(MeshIdentInfo, MeshIdentLoc);
  if (VD == 0)
    return StmtError();

  // If we are in scc-mode and inside a function then make sure
  // we have a *
  if(getLangOpts().ScoutC && isa<ParmVarDecl>(VD) && meshptr == false) {
    Diag(Tok,diag::err_expected_star_mesh);
    SkipUntil(tok::semi);
    return StmtError();
  }

  const MeshType *RefMeshType = LookupMeshType(VD, MeshIdentInfo);
  if (RefMeshType == 0){
    Diag(MeshIdentLoc, diag::err_expected_a_mesh_type);
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool success = Actions.ActOnRenderallMeshRefVariable(getCurScope(),
                                                       MeshElementType,
                                                       MeshIdentInfo,
                                                       MeshIdentLoc,
                                                       RefMeshType,
                                                       VD);
  if (! success)
    return StmtError();

  ConsumeToken();

  MeshElementTypeDiag(MeshElementType, RefMeshType, MeshIdentLoc);

  // Next we should find the 'to' keyword that is then followed by the 
  // the identifier for a render target. 
  if (Tok.isNot(tok::kw_to)) {
    Diag(Tok, diag::err_renderall_expected_kw_to);
    SkipUntil(tok::semi);
    return StmtError();
  }
  ConsumeToken();

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo *RenderTargetInfo = Tok.getIdentifierInfo();
  SourceLocation  RenderTargetLoc  = Tok.getLocation();

  VarDecl *RTVD = LookupRenderTargetVarDecl(RenderTargetInfo, RenderTargetLoc);
  if (RTVD == 0) {
    SkipUntil(tok::semi);    
    return StmtError();
  }

  const WindowType* wt = dyn_cast<WindowType>(RTVD->getType().getTypePtr());
  if(wt){
    if(wt->getUsage() == WindowType::Plot){
      Diag(Tok, diag::err_window_renderall_and_plot);
      return StmtError();
    }
    else{
      wt->setUsage(WindowType::Renderall);
    }
  }
  
  // this does not work from within LLDB because normally the render target type is
  // window - but from within LLDB it is a: struct __scout_win_t *
  // since we are not doing anything wih the RenderTargetType, this is commented
  // out for now
  
  /*
  const RenderTargetType *RefRenderTargetType =
  LookupRenderTargetType(RenderTargetInfo, RenderTargetLoc);
  if (RefRenderTargetType == 0) {
    return StmtError();
  }
  */

  ConsumeToken();
  
  // Now check to see if we have a predicate expression...
  //
  // SC_TODO - we need to validate/specialize the predicate...
  Expr *Predicate = 0;
  SourceLocation LParenLoc, RParenLoc;

  if (Tok.is(tok::kw_where)) {
    ConsumeToken();

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_lparen);
      // Multi-line skip, don't consume brace
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    LParenLoc = ConsumeParen();

    ExprResult PredicateResult = ParseExpression();
    if (PredicateResult.isInvalid()) {
      Diag(Tok, diag::err_forall_invalid_op);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    Predicate = PredicateResult.get();

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_rparen);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }
    RParenLoc = ConsumeParen();
  }

  //SourceLocation BodyLoc = Tok.getLocation();
  StmtResult BodyResult(ParseStatement());

  if (BodyResult.isInvalid()) {
    Diag(Tok, diag::err_invalid_forall_body);  // SC_TODO -- is this a useful diagnostic?
    SkipUntil(tok::semi);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();

  StmtResult RenderallResult = Actions.ActOnRenderallMeshStmt(RenderallKWLoc,
                                                              MeshElementType,
                                                              RefMeshType, VD,
                                                              RTVD, ElementIdentInfo,
                                                              MeshIdentInfo,
                                                              RenderTargetInfo,
                                                              LParenLoc,
                                                              Predicate,
                                                              RParenLoc,
                                                              Body);
  if (! RenderallResult.isUsable())
    return StmtError();

  return RenderallResult;
}


// +---- Parse a forall statement operating on an array --------------------+
//
// one of the following forms depending on array dimensions
// forall  i in [xstart:xend:xstride]
// forall  i,j in [xstart:xend:xstride,ystart:yend:ystride]
// forall  i,j,k in [xstart:xend:xstride,ystart:yend:ystride,zstart:zend:zstride]
// start and stride are optional and default to 0 and 1 respectfully
//
StmtResult Parser::ParseForallArrayStatement(ParsedAttributes &attrs) {
  assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");

  SourceLocation ForallLoc = ConsumeToken();

  IdentifierInfo* InductionVarInfo[3] = {0,0,0};
  SourceLocation InductionVarLoc[3];
  size_t dims;

  // parse up to 3 induction variables
  for(size_t i = 0; i < 3; ++i) {
    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_brace, StopBeforeMatch); //multiline skip, don't consume brace
      return StmtError();
    }

    InductionVarInfo[i] = Tok.getIdentifierInfo();
    InductionVarLoc[i] = ConsumeToken();

    dims = i + 1;

    if(Tok.is(tok::kw_in)){
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_forall_expected_comma_or_kw_in);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }
    ConsumeToken();
  } //end for i (induction vars)

  if(Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_forall_expected_kw_in);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }

  ConsumeToken();

  if(Tok.isNot(tok::l_square)){
    Diag(Tok, diag::err_expected_lsquare);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }

  ConsumeBracket();

  Expr* Start[3] = {0,0,0};
  Expr* End[3] = {0,0,0};
  Expr* Stride[3] = {0,0,0};

  // parse up to 3 (start:end:stride) ranges
  for(size_t i = 0; i < 3; ++i) {

    // don't allow :: as default end does not make sense
    // unless we look at what arrays are in the body.
    if(Tok.is(tok::coloncolon)) {
      Diag(Tok, diag::err_forall_array_invalid_end);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    // parse start
    if(Tok.is(tok::colon)) {
      Start[i] = IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 0),
                  Actions.Context.IntTy, ForallLoc);
    } else {
      ExprResult StartResult = ParseAssignmentExpression();
      if(StartResult.isInvalid()){
        Diag(Tok, diag::err_forall_array_invalid_start);
        SkipUntil(tok::r_brace, StopBeforeMatch);
        return StmtError();
      }
      Start[i] = StartResult.get();
    } // end if is :
    if (Tok.is(tok::colon) || isTokenStringLiteral() || isTokenParen() || isTokenBracket()) {
      ConsumeToken();
    } else if (Tok.is(tok::coloncolon)) {
      Diag(Tok, diag::err_forall_array_invalid_end);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    } else {
      Diag(Tok, diag:: err_forall_array_misformat);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    // parse end
    if(Tok.is(tok::colon)) {
      Diag(Tok, diag::err_forall_array_invalid_end);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    } else {
      ExprResult EndResult = ParseAssignmentExpression();
      if(EndResult.isInvalid()){
        Diag(Tok, diag::err_forall_array_invalid_end);
        SkipUntil(tok::r_brace, StopBeforeMatch);
        return StmtError();
      }
      End[i] = EndResult.get();
    }
    if (Tok.is(tok::colon) || isTokenStringLiteral() || isTokenParen() || isTokenBracket()) {
      ConsumeToken();
    } else {
      Diag(Tok, diag:: err_forall_array_misformat);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }

    // parse stride
    if(Tok.is(tok::comma) || Tok.is(tok::r_square)){
      Stride[i] =
      IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 1),
                             Actions.Context.IntTy, ForallLoc);
    } else {
      ExprResult StrideResult = ParseAssignmentExpression();
      if(StrideResult.isInvalid()){
        Diag(Tok, diag::err_forall_array_invalid_stride);
        SkipUntil(tok::r_brace, StopBeforeMatch);
        return StmtError();
      }
      Stride[i] = StrideResult.get();
    }

    if(Tok.isNot(tok::comma)){
      if(i != dims - 1){
        Diag(Tok, diag::err_forall_array_mismatch);
        SkipUntil(tok::r_brace, StopBeforeMatch);
        return StmtError();
      }
      break;
    }
    ConsumeToken();

  } // end for i (ranges)

  if(Tok.isNot(tok::r_square)){
    Diag(Tok, diag::err_expected_rsquare);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }

  ConsumeBracket();

  unsigned ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
  Scope::DeclScope | Scope::ControlScope;

  ParseScope ForAllScope(this, ScopeFlags);

  //DeclStmts to Init the Induction Vars.
  DeclStmt* Init[3] = {0,0,0};
  VarDecl* InductionVarDecl[3] = {0,0,0};
  for(size_t i = 0; i < dims; ++i){
    if(!InductionVarInfo[i]){
      break;
    }

    // returns InductionVarDecl[i] and Init[i]
    if(!Actions.ActOnForallArrayInductionVariable(getCurScope(),
        InductionVarInfo[i], InductionVarLoc[i],
        &InductionVarDecl[i], &Init[i])) {
      return StmtError();
    }
  }

  StmtResult BodyResult(ParseStatement());
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_forall_body);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();

  StmtResult ForallArrayResult =
  Actions.ActOnForallArrayStmt(InductionVarInfo, InductionVarDecl,
      Start, End, Stride, dims,
      ForallLoc, Init, Body);

  return ForallArrayResult;
}

StmtResult Parser::ParseFrameCaptureStatement(ParsedAttributes &Attr){
  assert(Tok.is(tok::kw_into) && "expected keyword into");
  
  SourceLocation IntoLoc = ConsumeToken();
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  IdentifierInfo  *IdentInfo = Tok.getIdentifierInfo();
  SourceLocation   IdentLoc  = Tok.getLocation();
  
  VarDecl *VD = LookupScoutVarDecl(IdentInfo, IdentLoc);
  
  if(!VD){
    Diag(Tok, diag::err_expected_frame);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  const FrameType* FT = dyn_cast<FrameType>(VD->getType().getTypePtr());
  if(!FT){
    Diag(Tok, diag::err_expected_frame);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  const FrameDecl* FD = FT->getDecl();
  
  SourceLocation FrameLoc = ConsumeToken();
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_frame_expected_capture);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  IdentInfo = Tok.getIdentifierInfo();
  if(IdentInfo->getName().str() != "capture"){
    Diag(Tok, diag::err_frame_expected_capture);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  SourceLocation CaptureLoc = ConsumeToken();
  
  if(Tok.isNot(tok::l_brace)){
    Diag(Tok.getLocation(), diag::err_frame_expected_specifier);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  ExprResult result = ParseSpecObjectExpression();
  if(result.isInvalid()){
    return StmtError();
  }
  
  ScoutExpr* expr = cast<ScoutExpr>(result.get());
  SpecObjectExpr* Spec = static_cast<SpecObjectExpr*>(expr);
  return Actions.ActOnFrameCaptureStmt(VD, Spec);
}

StmtResult Parser::ParsePlotStatement(ParsedAttributes &Attr){
  assert(Tok.is(tok::kw_with) && "expected keyword with");
  
  SourceLocation WithLoc = ConsumeToken();
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  IdentifierInfo  *IdentInfo = Tok.getIdentifierInfo();
  SourceLocation   IdentLoc  = Tok.getLocation();
  
  VarDecl *VD = LookupScoutVarDecl(IdentInfo, IdentLoc);
  
  if(!VD){
    Diag(Tok, diag::err_expected_frame);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  FrameDecl* FD;
  
  const FrameType* FT = dyn_cast<FrameType>(VD->getType().getTypePtr());
  
  if(FT){
    FD = FT->getDecl();
  }
  else{
    const UniformMeshType* MT =
    dyn_cast<UniformMeshType>(VD->getType().getTypePtr());
    
    MT->dump();
    
    if(!MT){
      Diag(Tok, diag::err_expected_frame_or_mesh);
      SkipUntil(tok::r_brace, StopBeforeMatch);
      return StmtError();
    }
  }
  
  SourceLocation FrameLoc = ConsumeToken();
  
  if(Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_plot_expected_kw_in);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  SourceLocation InLoc = ConsumeToken();
  
  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  IdentifierInfo* RenderTargetInfo = Tok.getIdentifierInfo();
  SourceLocation  RenderTargetLoc  = Tok.getLocation();
  
  VarDecl* RTVD = LookupRenderTargetVarDecl(RenderTargetInfo, RenderTargetLoc);
  if(RTVD == 0){
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  const WindowType* wt = dyn_cast<WindowType>(RTVD->getType().getTypePtr());
  if(wt){
    if(wt->getUsage() == WindowType::Renderall){
      Diag(Tok, diag::err_window_renderall_and_plot);
      return StmtError();
    }
    else{
      wt->setUsage(WindowType::Plot);
    }
  }
  
  SourceLocation RTLoc = ConsumeToken();
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_frame_expected_capture);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  IdentInfo = Tok.getIdentifierInfo();
  if(IdentInfo->getName().str() != "plot"){
    Diag(Tok, diag::err_frame_expected_capture);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }
  
  SourceLocation PlotLoc = ConsumeToken();
  
  if(Tok.isNot(tok::l_brace)){
    Diag(Tok.getLocation(), diag::err_plot_expected_specifier);
    SkipUntil(tok::r_brace, StopBeforeMatch);
    return StmtError();
  }

  ParseScope FrameScope(this, Scope::ControlScope|Scope::DeclScope);
  
  auto& M = FD->getVarMap();
  
  for(auto& itr : M){
    Actions.PushOnScopeChains(itr.second.varDecl, getCurScope(), false);
  }
  
  ExprResult result = ParseSpecObjectExpression();
  
  FrameScope.Exit();
  
  if(result.isInvalid()){
    return StmtError();
  }

  ScoutExpr* SE = cast<ScoutExpr>(result.get());
  SpecObjectExpr* Spec = static_cast<SpecObjectExpr*>(SE);
  
  return Actions.ActOnPlotStmt(WithLoc, FrameLoc, RTVD, VD, Spec);
}
