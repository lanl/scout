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

// ----- LookupMeshVarDecl
//
VarDecl* Parser::LookupMeshVarDecl(IdentifierInfo *MeshInfo,
                                   SourceLocation MeshLoc) {

  LookupResult MeshLookup(Actions, MeshInfo, MeshLoc, Sema::LookupOrdinaryName);
  Actions.LookupName(MeshLookup, getCurScope());

  if (MeshLookup.getResultKind() != LookupResult::Found) {
    Diag(MeshLoc, diag::err_unknown_mesh_variable) << MeshInfo;
    return 0;
  }

  NamedDecl* ND = MeshLookup.getFoundDecl();
  if (!isa<VarDecl>(ND)) {
    Diag(MeshLoc, diag::err_expected_a_mesh_type) << MeshInfo;
    return 0;
  }

  return cast<VarDecl>(ND);
}


// ----- LookupMeshType
//
const MeshType* Parser::LookupMeshType(IdentifierInfo *MeshInfo,
                                       SourceLocation MeshLoc) {

  VarDecl* VD = LookupMeshVarDecl(MeshInfo, MeshLoc);
  if (VD == 0)
    return 0;
  else {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();
    if (!isa<MeshType>(T)) {
      T = VD->getType().getCanonicalType().getNonReferenceType().getTypePtr();
      if(!isa<MeshType>(T)) {
        // Should this diag go in sema instead?
        //Diag(MeshLoc, diag::err_forall_not_a_mesh_type) << MeshInfo;
        return 0;
      }
    }

    return const_cast<MeshType *>(cast<MeshType>(T));
  }
}


// ---- LookupMeshType
//
const MeshType* Parser::LookupMeshType(VarDecl *VD,
                                       IdentifierInfo *MeshInfo,
                                       SourceLocation MeshLoc) {
  assert(VD != 0 && "null var decl passed for mesh type lookup");
  if (VD) {
    const Type* T = VD->getType().getCanonicalType().getTypePtr();
    if (! isa<MeshType>(T)) {
      //Should this diag go in sema instead?
      //Diag(MeshLoc, diag::err_forall_not_a_mesh_type) << MeshInfo;
      return 0;
    } else {
      return const_cast<MeshType *>(cast<MeshType>(T));
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
  assert(Tok.getKind() == tok::kw_forall && "epxected input token to be 'forall'");

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

  // Finally, we are at the identifier that specifies the mesh
  // that we are computing over.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo  *MeshIdentInfo = Tok.getIdentifierInfo();
  SourceLocation   MeshIdentLoc  = Tok.getLocation();

  VarDecl *VD = LookupMeshVarDecl(MeshIdentInfo, MeshIdentLoc);
  if (VD == 0)
    return StmtError();

  const MeshType *RefMeshType = LookupMeshType(VD, MeshIdentInfo, MeshIdentLoc);
  if (RefMeshType == 0)
    return StmtError();

  bool success = Actions.ActOnForallMeshRefVariable(getCurScope(),
                                                    ElementIdentInfo,
                                                    ElementIdentLoc,
                                                    RefMeshType,
                                                    VD);
  if (! success)
    return StmtError();

  ConsumeToken();


  // SC_TODO - we might want to lift this block of code out into a
  // function where we can reuse it. 
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
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }

    LParenLoc = ConsumeParen();

    ExprResult PredicateResult = ParseExpression();
    if (PredicateResult.isInvalid()) {
      Diag(Tok, diag::err_forall_invalid_op);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }

    Predicate = PredicateResult.get();

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_rparen);
      SkipUntil(tok::r_brace, false, true);
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
                                                        MeshIdentInfo,
                                                        LParenLoc,
                                                        Predicate,
                                                        RParenLoc,
                                                        Body);
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
  assert(Tok.getKind() == tok::kw_renderall &&
         "epxected input token to be 'renderall'");

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

  // Finally, we are at the identifier that specifies the mesh
  // that we are computing over.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo  *MeshIdentInfo = Tok.getIdentifierInfo();
  SourceLocation   MeshIdentLoc  = Tok.getLocation();

  VarDecl *VD = LookupMeshVarDecl(MeshIdentInfo, MeshIdentLoc);
  if (VD == 0)
    return StmtError();

  const MeshType *RefMeshType = LookupMeshType(VD, MeshIdentInfo, MeshIdentLoc);
  if (RefMeshType == 0)
    return StmtError();

  bool success = Actions.ActOnForallMeshRefVariable(getCurScope(),
                                                    ElementIdentInfo,
                                                    ElementIdentLoc,
                                                    RefMeshType,
                                                    VD);
  if (! success)
    return StmtError();

  ConsumeToken();


  // SC_TODO - we might want to lift this block of code out into a
  // function where we can reuse it.  Probably want to rename warning
  // values as well -- e.g. 'diag::warn_mesh_has_no_cell_fields'...
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
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }

    LParenLoc = ConsumeParen();

    ExprResult PredicateResult = ParseExpression();
    if (PredicateResult.isInvalid()) {
      Diag(Tok, diag::err_forall_invalid_op);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }

    Predicate = PredicateResult.get();

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_forall_predicate_missing_rparen);
      SkipUntil(tok::r_brace, false, true);
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
  StmtResult RenderallResult = Actions.ActOnRenderallMeshStmt(RenderallKWLoc,
                                                              MeshElementType,
                                                              RefMeshType, VD,
                                                              ElementIdentInfo,
                                                              MeshIdentInfo,
                                                              LParenLoc,
                                                              Predicate,
                                                              RParenLoc,
                                                              Body);
  if (! RenderallResult.isUsable())
    return StmtError();

  return RenderallResult;
}


bool Parser::ParseMeshStatement(StmtVector &Stmts,
                                bool OnlyStatement,
                                Token &Next,
                                StmtResult &SR) {

  //IdentifierInfo* Name = Tok.getIdentifierInfo();
  //SourceLocation NameLoc = Tok.getLocation();

  /*
  // scout - detect the forall shorthand, e.g:
  // m.a[1..width-2][1..height-2] = MAX_TEMP;
  if(isScoutLang()) {
    if(Actions.isScoutSource(NameLoc)){
      if(GetLookAheadToken(1).is(tok::period) &&
         GetLookAheadToken(2).is(tok::identifier) &&
         GetLookAheadToken(3).is(tok::l_square)){

        LookupResult
        Result(Actions, Name, NameLoc, Sema::LookupOrdinaryName);

        Actions.LookupName(Result, getCurScope());

        if(Result.getResultKind() == LookupResult::Found){
          if(VarDecl* vd = dyn_cast<VarDecl>(Result.getFoundDecl())){
            if(isa<MeshType>(vd->getType().getCanonicalType().getTypePtr())){
              SR = ParseForAllShortStatement(Name, NameLoc, vd);
              return true;
            }
          }
        }
      }
    }
  }
  */
  return false;
}


StmtResult Parser::ParseForallArrayStatement(ParsedAttributes &attrs) {
  assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");

  SourceLocation ForallLoc = ConsumeToken();

  IdentifierInfo* InductionVarII[3] = {0,0,0};
  SourceLocation InductionVarLoc[3];

  size_t count;
  // parse up to 3 identifiers
  for(size_t i = 0; i < 3; ++i) {
    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_brace, false, true); //multiline skip, don't consume brace
      return StmtError();
    }

    InductionVarII[i] = Tok.getIdentifierInfo();
    InductionVarLoc[i] = ConsumeToken();

    count = i + 1;

    if(Tok.is(tok::kw_in)){
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_forall_expected_comma_or_kw_in);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }
    ConsumeToken();
  } //end for i (identifiers)

  if(Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_forall_expected_kw_in);
    SkipUntil(tok::r_brace, false, true);
    return StmtError();
  }

  ConsumeToken();

  if(Tok.isNot(tok::l_square)){
    Diag(Tok, diag::err_expected_lsquare);
    SkipUntil(tok::r_brace, false, true);
    return StmtError();
  }

  ConsumeBracket();

  Expr* Start[3] = {0,0,0};
  Expr* End[3] = {0,0,0};
  Expr* Stride[3] = {0,0,0};

  // parse up to 3 (start:end:stride) ranges
  for(size_t i = 0; i < 3; ++i) {
    if(Tok.is(tok::coloncolon)) { // don't allow ::
      Diag(Tok, diag::err_forall_array_invalid_end);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    } else {

      // parse start
      if(Tok.is(tok::colon)) {
        Start[i] = IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 0),
                    Actions.Context.IntTy, ForallLoc);
      } else {
        ExprResult StartResult = ParseAssignmentExpression();
        if(StartResult.isInvalid()){
          Diag(Tok, diag::err_forall_array_invalid_start);
          SkipUntil(tok::r_brace, false, true);
          return StmtError();
        }
        Start[i] = StartResult.get();
      } // end if is :
      ConsumeToken();

      // parse end
      if(Tok.is(tok::colon)) {
        Diag(Tok, diag::err_forall_array_invalid_end);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      } else {
        ExprResult EndResult = ParseAssignmentExpression();
        if(EndResult.isInvalid()){
          Diag(Tok, diag::err_forall_array_invalid_end);
          SkipUntil(tok::r_brace, false, true);
          return StmtError();
        }
        End[i] = EndResult.get();
      }
      ConsumeToken();

    }

    // parse stride
    if(Tok.is(tok::comma) || Tok.is(tok::r_square)){
      // note: non-zero stride is used to denote this dimension exists
      Stride[i] =
      IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 1),
                             Actions.Context.IntTy, ForallLoc);
    } else {
      ExprResult StrideResult = ParseAssignmentExpression();
      if(StrideResult.isInvalid()){
        Diag(Tok, diag::err_forall_array_invalid_stride);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      }
      Stride[i] = StrideResult.get();
    }

    if(Tok.isNot(tok::comma)){
      if(i != count - 1){
        Diag(Tok, diag::err_forall_array_mismatch);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      }
      break;
    }
    ConsumeToken();

  } // end for i (ranges)

  if(Tok.isNot(tok::r_square)){
    Diag(Tok, diag::err_expected_rsquare);
    SkipUntil(tok::r_brace, false, true);
    return StmtError();
  }

  ConsumeBracket();

  unsigned ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
  Scope::DeclScope | Scope::ControlScope;

  ParseScope ForAllScope(this, ScopeFlags);

  for(size_t i = 0; i < count; ++i){
    if(!InductionVarII[i]){
      break;
    }

    if(!Actions.ActOnForallArrayInductionVariable(getCurScope(),
        InductionVarII[i],
        InductionVarLoc[i])){
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
  Actions.ActOnForallArrayStmt(InductionVarII, InductionVarLoc,
      Start, End, Stride,
      ForallLoc, Body);

  return ForallArrayResult;
}


