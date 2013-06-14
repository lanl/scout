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
#include "clang/AST/scout/MeshDecls.h"
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

// scout - Stmts
StmtResult Parser::ParseForAllStatement(ParsedAttributes &attrs, bool ForAll) {
  if(ForAll)
    assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");
  else
    assert(Tok.is(tok::kw_renderall) && "Not a renderall stmt!");

  SourceLocation ForAllLoc = ConsumeToken();  // eat the 'forall' / 'renderall'

  tok::TokenKind VariableType = Tok.getKind();

  bool elements = false;

  ForAllStmt::ForAllType FT;
  switch(VariableType) {
    case tok::kw_cells:
      FT = ForAllStmt::Cells;
      break;
    case tok::kw_edges:
      FT = ForAllStmt::Edges;
      break;
    case tok::kw_vertices:
      FT = ForAllStmt::Vertices;
      break;
    case tok::kw_elements:
      if(!ForAll){
        elements = true;
        break;
      }
      // fall through if this is a forall
    default: {
      Diag(Tok, diag::err_expected_vertices_cells);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }

  ConsumeToken();

  unsigned ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
  Scope::DeclScope | Scope::ControlScope;

  ParseScope ForAllScope(this, ScopeFlags);

  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo* LoopVariableII = Tok.getIdentifierInfo();
  SourceLocation LoopVariableLoc = Tok.getLocation();

  ConsumeToken();

  if (elements) {
    if (Tok.isNot(tok::kw_in)) {
      Diag(Tok, diag::err_expected_in_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }
  } else {
    if (Tok.isNot(tok::kw_of)) {
      Diag(Tok, diag::err_expected_of_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }

  ConsumeToken();

  IdentifierInfo* MeshII = 0;
  SourceLocation MeshLoc;
  VarDecl* MVD = 0;

  MemberExpr* ElementMember = 0;
  Expr* ElementColor = 0;
  Expr* ElementRadius = 0;

  const UniformMeshType *MT;

  IdentifierInfo* CameraII = 0;
  SourceLocation CameraLoc;

  if (elements) {
    if (MemberExpr* me = dyn_cast<MemberExpr>(ParseExpression().get())) {
      if (MeshFieldDecl* fd = dyn_cast<MeshFieldDecl>(me->getMemberDecl())) {

        if (const ArrayType* at =
           dyn_cast<ArrayType>(fd->getType().getTypePtr())) {
            (void)at; //suppress warning;
        } else {
          Diag(Tok, diag::err_not_array_renderall_elements);
          SkipUntil(tok::semi);
          return StmtError();
        }

        if (fd->isCellLocated()) {
          ElementMember = me;
        }
      }
    }

    if (!ElementMember) {
      Diag(Tok, diag::err_expected_mesh_field);
      SkipUntil(tok::semi);
      return StmtError();
    }

    if (Tok.isNot(tok::kw_as)) {
      Diag(Tok, diag::err_expected_as_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }

    ConsumeToken();

    if (Tok.isNot(tok::kw_spheres)) {
      Diag(Tok, diag::err_expected_spheres_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }

    ConsumeToken();

    FT = ForAllStmt::ElementSpheres;

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen);
      SkipUntil(tok::semi);
      return StmtError();
    }

    ConsumeParen();

    for(unsigned i = 0; i < 2; ++i) {
      if (Tok.is(tok::r_paren)) {
        break;
      }

      if (Tok.is(tok::identifier)) {
        IdentifierInfo* II = Tok.getIdentifierInfo();
        SourceLocation IILoc = Tok.getLocation();

        ConsumeToken();

        if (Tok.isNot(tok::equal)) {
          Diag(Tok, diag::err_expected_equal_after_element_default);
          SkipUntil(tok::semi);
          return StmtError();
        }

        ConsumeToken();

        ExprResult result = ParseAssignmentExpression();

        if (result.isInvalid()) {
          return StmtError();
        }

        if (II->getName() == "radius") {

          if (ElementRadius) {
            Diag(IILoc, diag::err_duplicate_radius_default);
            SkipUntil(tok::semi);
            return StmtError();
          }

          ElementRadius = result.get();

        } else if(II->getName() == "color") {

          if (ElementColor){
            Diag(Tok, diag::err_duplicate_color_default);
            SkipUntil(tok::semi);
            return StmtError();
          }

          ElementColor = result.get();
        } else {
          Diag(IILoc, diag::err_invalid_default_element);
          SkipUntil(tok::semi);
          return StmtError();
        }
      }
    }

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::semi);
      return StmtError();
    }

    ConsumeParen();
  } else {

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return StmtError();
    }

    MeshII = Tok.getIdentifierInfo();
    MeshLoc = Tok.getLocation();

    ConsumeToken();
  }

  bool success = false;
  if (ForAll) {
    success = Actions.ActOnForAllLoopVariable(getCurScope(),
                                              VariableType,
                                              LoopVariableII,
                                              LoopVariableLoc,
                                              MeshII,
                                              MeshLoc);
  } else {
    if (elements) {
      const MeshType* mt2 =
      Actions.ActOnRenderAllElementsVariable(getCurScope(),
                                             ElementMember,
                                             VariableType,
                                             LoopVariableII,
                                             LoopVariableLoc);

      MT = cast<UniformMeshType>(mt2);

      if (MT) {
        success = true;
      }

    } else {

      success = Actions.ActOnRenderAllLoopVariable(getCurScope(),
                                                   VariableType,
                                                   LoopVariableII,
                                                   LoopVariableLoc,
                                                   MeshII,
                                                   MeshLoc);
    }
  }

  if (!success) {
    return StmtError();
  }


  Expr* Op;
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;

  // Lookup the meshtype and store it for the ForAllStmt Constructor.
  LookupResult LR(Actions, MeshII, MeshLoc, Sema::LookupOrdinaryName);
  Actions.LookupName(LR, getCurScope());
  MVD = cast<VarDecl>(LR.getFoundDecl());

  if (!elements) {
    Op = 0;

    MT =
    dyn_cast<UniformMeshType>(MVD->getType().getCanonicalType().
                          getNonReferenceType().getTypePtr());
    size_t FieldCount = 0;
    const UniformMeshDecl* MD = MT->getDecl();

    //Sema::ContextRAII contextRAII(Actions, const_cast<UniformMeshDecl*>(MD));

    for(MeshDecl::mesh_field_iterator FI = MD->mesh_field_begin(),
        FE = MD->mesh_field_end(); FI != FE; ++FI) {
      MeshFieldDecl* FD = *FI;

      switch(FT) {
        case ForAllStmt::Cells:
          if (!FD->isImplicit() && FD->isCellLocated()) {
            ++FieldCount;
          }
          break;
        case ForAllStmt::Edges:
          if (!FD->isImplicit() && FD->isEdgeLocated()) {
            ++FieldCount;
          }
          break;
        case ForAllStmt::Vertices:
          if (!FD->isImplicit() && FD->isVertexLocated()) {
            ++FieldCount;
          }
          break;
        case ForAllStmt::ElementSpheres:
        case ForAllStmt::Array:
        case ForAllStmt::Faces:
          assert(false && "unimplemented ForAllStmt case");
          break;
      }
    }

    if (FieldCount == 0) {
      switch(FT) {
        case ForAllStmt::Cells:
          Diag(ForAllLoc, diag::warn_no_cells_fields_forall);
          break;
        case ForAllStmt::Edges:
          Diag(ForAllLoc, diag::warn_no_edges_fields_forall);
          break;
        case ForAllStmt::Vertices:
          Diag(ForAllLoc, diag::warn_no_vertices_fields_forall);
          break;
        case ForAllStmt::ElementSpheres:
        case ForAllStmt::Array:
        case ForAllStmt::Faces:
          break;
      }
    }

    // If 3D and forall type is cell(volume renderall), it can accept a camera and window
    // ala "with camera onto win", where camera and window were
    // defined previously.  If none are given it can do a default, but that is
    // probably not going to be a good thing.  You may not see the volume if the
    // camera is not pointing at it.

    if (!ForAll && (MT->dimensions().size() == 3) && (FT == ForAllStmt::Cells)) {
      if (Tok.is(tok::kw_with)) {
        ConsumeToken();
        if(Tok.isNot(tok::identifier)){
          Diag(Tok, diag::err_expected_ident);
          SkipUntil(tok::semi);
          return StmtError();
        }

        CameraII = Tok.getIdentifierInfo();
        CameraLoc = Tok.getLocation();
        ConsumeToken();
      }
    }

    if(Tok.is(tok::kw_where)){
      ConsumeToken();
      if(Tok.isNot(tok::l_paren)){
        Diag(Tok, diag::err_invalid_forall_op);
        SkipUntil(tok::r_brace, false, true); //multiline skip, don't consume brace
        return StmtError();
      }

      LParenLoc = ConsumeParen();

      ExprResult OpResult = ParseExpression();
      if(OpResult.isInvalid()){
        Diag(Tok, diag::err_invalid_forall_op);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      }

      Op = OpResult.get();

      if(Tok.isNot(tok::r_paren)){
        Diag(Tok, diag::err_expected_rparen);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      }
      RParenLoc = ConsumeParen();
    }
    else{
      Op = 0;
    }

  }

  // Check if this is a volume renderall.  If the mesh is
  // three-dimensional and has cells as the ForAllType,
  // then we branch off here into other code to handle it.

  if (!ForAll && (MT->dimensions().size() == 3) && (FT == ForAllStmt::Cells)) {
    return(ParseVolumeRenderAll(getCurScope(), ForAllLoc, attrs, MeshII, MVD,
          CameraII, CameraLoc, Op, LParenLoc, RParenLoc));
  }

  SourceLocation BodyLoc = Tok.getLocation();

  StmtResult BodyResult(ParseStatement());

  if(BodyResult.isInvalid()){
    if(ForAll)
      Diag(Tok, diag::err_invalid_forall_body);
    else
      Diag(Tok, diag::err_invalid_renderall_body);
    SkipUntil(tok::semi);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();

  StmtResult ForAllResult;

  InsertCPPCode("^(void* m, int* i, int* j, int* k){}", BodyLoc);
  BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
  assert(Block && "expected a block expression");
  Block->getBlockDecl()->setBody(cast<class CompoundStmt>(Body));

  if(ForAll){
    ForAllResult = Actions.ActOnForAllStmt(ForAllLoc, FT, MT, MVD,
                                           LoopVariableII, MeshII, LParenLoc,
                                           Op, RParenLoc, Body, Block);
    if(!ForAllResult.isUsable())
      return StmtError();
  } else { //renderall
    assert(!StmtsStack.empty());

    MeshType::MeshDimensionVec dims = MT->dimensions();

    assert(dims.size() >= 1);
#ifndef SC_USE_RT_REWRITER // for testing rewriter
    std::string bc;
    bc = "__scrt_renderall_uniform_begin(";
    bc += MVD->getName().str() + ".width, ";
    bc += MVD->getName().str() + ".height, ";
    bc += MVD->getName().str() + ".depth);";

    InsertCPPCode(bc, Tok.getLocation());

    StmtResult BR = ParseStatementOrDeclaration(*StmtsStack.back(), true).get();

    StmtsStack.back()->push_back(BR.get());

    InsertCPPCode("__scrt_renderall_end();", BodyLoc);
#endif
    ForAllResult = Actions.ActOnRenderAllStmt(ForAllLoc, FT, MT, MVD,
                                              LoopVariableII, MeshII, LParenLoc,
                                              Op, RParenLoc, Body, Block);
  }

  ForAllStmt *FAS;
  if(ForAllResult.get()->getStmtClass() == Stmt::RenderAllStmtClass) {
    RenderAllStmt* RAS = cast<RenderAllStmt>(ForAllResult.get());

    if(elements){
      RAS->setElementMember(ElementMember);
      RAS->setElementColor(ElementColor);
      RAS->setElementRadius(ElementRadius);
    }

    FAS = cast<ForAllStmt>(RAS);
  } else {
    FAS = cast<ForAllStmt>(ForAllResult.get());
  }

  MeshType::MeshDimensionVec dims = MT->dimensions();
  ASTContext &C = Actions.getASTContext();
  Expr *zero = IntegerLiteral::Create(C, llvm::APInt(32, 0),
                                      C.IntTy, ForAllLoc);
  Expr *one = IntegerLiteral::Create(C, llvm::APInt(32, 1),
                                      C.IntTy, ForAllLoc);
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    FAS->setStart(i, zero);
    FAS->setEnd(i, dims[i]);
    FAS->setStride(i, one);
  }
  return ForAllResult;
}

StmtResult
Parser::ParseForAllShortStatement(IdentifierInfo* Name,
                                  SourceLocation NameLoc,
                                  VarDecl* VD){
  ConsumeToken();

  assert(Tok.is(tok::period) && "expected period");
  ConsumeToken();

  assert(Tok.is(tok::identifier) && "expected identifier");

  IdentifierInfo* FieldName = Tok.getIdentifierInfo();
  SourceLocation FieldLoc = ConsumeToken();
  (void)FieldLoc; //suppress warning

  Expr* XStart = 0;
  Expr* XEnd = 0;
  Expr* YStart = 0;
  Expr* YEnd = 0;
  Expr* ZStart = 0;
  Expr* ZEnd = 0;

  Actions.SCLStack.push_back(VD);

  for(size_t i = 0; i < 3; ++i){

    assert(Tok.is(tok::l_square) && "expected l_square");
    ConsumeBracket();

    ExprResult Start = ParseExpression();
    if(Start.isInvalid()){
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    if(Tok.isNot(tok::periodperiod)){
      Diag(Tok, diag::err_expected_periodperiod);
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    ConsumeToken();

    ExprResult End = ParseExpression();
    if(End.isInvalid()){
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    switch(i){
      case 0:
      {
        XStart = Start.get();
        XEnd = End.get();
        break;
      }
      case 1:
      {
        YStart = Start.get();
        YEnd = End.get();
        break;
      }
      case 2:
      {
        ZStart = Start.get();
        ZEnd = End.get();
        break;
      }
    }

    if(Tok.isNot(tok::r_square)){
      Diag(Tok, diag::err_expected_rsquare);
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    ConsumeBracket();

    if(Tok.isNot(tok::l_square)){
      break;
    }
  }

  if(Tok.isNot(tok::equal) &&
     Tok.isNot(tok::plusequal) &&
     Tok.isNot(tok::minusequal) &&
     Tok.isNot(tok::starequal) &&
     Tok.isNot(tok::slashequal)){
    Diag(Tok, diag::err_expected_forall_binary_op);
    SkipUntil(tok::semi);
    Actions.SCLStack.pop_back();
    return StmtError();
  }

  std::string code = FieldName->getName().str() + " " + TokToStr(Tok) + " ";

  SourceLocation CodeLoc = ConsumeToken();

  ExprResult rhs = ParseExpression();

  if(rhs.isInvalid()){
    SkipUntil(tok::semi);
    Actions.SCLStack.pop_back();
    return StmtError();
  }

  code += ToCPPCode(rhs.get());

  InsertCPPCode(code, CodeLoc);

  Stmt* Body = ParseStatement().get();

  InsertCPPCode("^(void* m, int* i, int* j, int* k){}", CodeLoc);

  BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
  assert(Block && "expected a block expression");
  class CompoundStmt* CB =
  new (Actions.Context) class CompoundStmt(CodeLoc);

  CB->setStmts(Actions.Context, &Body, 1);
  Block->getBlockDecl()->setBody(CB);

  // Lookup the meshtype and store it for the ForAllStmt Constructor.
  LookupResult LR(Actions, Name, NameLoc, Sema::LookupOrdinaryName);
  Actions.LookupName(LR, getCurScope());
  VarDecl* MVD = cast<VarDecl>(LR.getFoundDecl());
  const MeshType *MT = cast<MeshType>(MVD->getType().getCanonicalType());

  StmtResult ForAllResult =
  Actions.ActOnForAllStmt(NameLoc,
                          ForAllStmt::Cells,
                          MT,
                          MVD,
                          &Actions.Context.Idents.get("c"),
                          Name,
                          NameLoc,
                          0, NameLoc, Body, Block);

  ForAllStmt* FAS = cast<ForAllStmt>(ForAllResult.get());

  FAS->setXStart(XStart);
  FAS->setXEnd(XEnd);
  FAS->setYStart(YStart);
  FAS->setYEnd(YEnd);
  FAS->setZStart(ZStart);
  FAS->setZEnd(ZEnd);

  return ForAllResult;
}

StmtResult Parser::ParseForAllArrayStatement(ParsedAttributes &attrs){
  assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");

  SourceLocation ForAllLoc = ConsumeToken();

  IdentifierInfo* IVII[3] = {0,0,0};
  SourceLocation IVSL[3];

  size_t count;
  // parse up to 3 identifiers
  for(size_t i = 0; i < 3; ++i) {
    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_brace, false, true); //multiline skip, don't consume brace
      return StmtError();
    }

    IVII[i] = Tok.getIdentifierInfo();
    IVSL[i] = ConsumeToken();

    count = i + 1;

    if(Tok.is(tok::kw_in)){
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_expected_comma_or_in_kw);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    }
    ConsumeToken();
  } //end for i (identifiers)

  if(Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_expected_in_kw);
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
      Diag(Tok, diag::err_invalid_end_forall_array);
      SkipUntil(tok::r_brace, false, true);
      return StmtError();
    } else {

      // parse start
      if(Tok.is(tok::colon)) {
        Start[i] = IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 0),
                    Actions.Context.IntTy, ForAllLoc);
      } else {
        ExprResult StartResult = ParseAssignmentExpression();
        if(StartResult.isInvalid()){
          Diag(Tok, diag::err_invalid_start_forall_array);
          SkipUntil(tok::r_brace, false, true);
          return StmtError();
        }
        Start[i] = StartResult.get();
      } // end if is :
      ConsumeToken();

      // parse end
      if(Tok.is(tok::colon)) {
        Diag(Tok, diag::err_invalid_end_forall_array);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      } else {
        ExprResult EndResult = ParseAssignmentExpression();
        if(EndResult.isInvalid()){
          Diag(Tok, diag::err_invalid_end_forall_array);
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
                             Actions.Context.IntTy, ForAllLoc);
    } else {
      ExprResult StrideResult = ParseAssignmentExpression();
      if(StrideResult.isInvalid()){
        Diag(Tok, diag::err_invalid_stride_forall_array);
        SkipUntil(tok::r_brace, false, true);
        return StmtError();
      }
      Stride[i] = StrideResult.get();
    }

    if(Tok.isNot(tok::comma)){
      if(i != count - 1){
        Diag(Tok, diag::err_mismatch_forall_array);
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
    if(!IVII[i]){
      break;
    }

    if(!Actions.ActOnForAllArrayInductionVariable(getCurScope(),
                                                  IVII[i],
                                                  IVSL[i])){
      return StmtError();
    }
  }

  StmtResult BodyResult(ParseStatement());
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_forall_body);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();

  InsertCPPCode("^(void* m, int* i, int* j, int* k){}", ForAllLoc);

  BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
  assert(Block && "expected a block expression");
  Block->getBlockDecl()->setBody(cast<class CompoundStmt>(Body));

  StmtResult ForAllArrayResult =
  Actions.ActOnForAllArrayStmt(ForAllLoc, Body, Block);

  Stmt* stmt = ForAllArrayResult.get();

  ForAllArrayStmt* FA = dyn_cast<ForAllArrayStmt>(stmt);

  for(size_t i = 0; i < 3; ++i){
    // non-zero stride is used to denote this dimension exists
    if(!Stride[i]){
      break;
    }

    FA->setStart(i, Start[i]);
    FA->setEnd(i, End[i]);
    FA->setStride(i, Stride[i]);
    FA->setInductionVar(i, IVII[i]);
  }

  return ForAllArrayResult;
}

StmtResult Parser::ParseVolumeRenderAll(Scope* scope,
    SourceLocation VolRenLoc, ParsedAttributes &attrs,
    IdentifierInfo* MeshII, VarDecl* MVD,
    IdentifierInfo* CameraII, SourceLocation CameraLoc, Expr* Op,
    SourceLocation OpLParenLoc, SourceLocation OpRParenLoc){

  ParseScope CompoundScope(this, Scope::DeclScope);
  StmtVector Stmts;
  StmtResult R;
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
      Tok.getLocation(),
      "in volume renderall statement ('{}')");

  // Now parse Body for transfer function closure.
  // We want to parse it here -- will give correct line numbers
  // for errors and warnings if we do so.
  StmtResult BodyResult(ParseStatement());

  // Not sure why it doesn't deem it invalid when an error is
  // found in ParseStatement

  // We end up getting an error and warning later, since it doesn't seem to
  // return here as I would expect.
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_renderall_body);
    SkipUntil(tok::r_brace);
    return StmtError();
  }

  class CompoundStmt* compoundStmt;
  compoundStmt = dyn_cast<class CompoundStmt>(BodyResult.get());
  SourceLocation RBraceLoc = compoundStmt->getRBracLoc();

  // TBD do more in here

  return Actions.ActOnVolumeRenderAllStmt(scope, VolRenLoc, LBraceLoc, RBraceLoc,
      MeshII, MVD, CameraII, CameraLoc, Stmts, compoundStmt, false);

}
