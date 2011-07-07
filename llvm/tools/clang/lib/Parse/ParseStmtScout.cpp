//===----------------------------------------------------------------------===//
//
// ndm - This file implements the Scout Stmt parsing methods.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"

#include "clang/AST/StmtScout.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/Sema.h"

using namespace clang;

StmtResult Parser::ParseForAllStatement(ParsedAttributes &attrs) {
  /*
  assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");
  SourceLocation ForAllLoc = ConsumeToken();  // eat the 'forall'.
  
  ForAllStmt::ForAllType Type;
  
  switch(Tok.getKind()){
    case tok::kw_cells:
      Type = ForAllStmt::Cells;
      break;
    case tok::kw_vertices:
      Type = ForAllStmt::Vertices;
      break; 
    default: {
      Diag(Tok, diag::err_expected_vertices_cells);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }
  
  ConsumeToken();
  
  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  Expr* Ind = ParseExpression().get();
  
  if(Tok.isNot(tok::kw_of)){
    Diag(Tok, diag::err_expected_of_kw);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  ConsumeToken();
  
  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  Expr* Mesh = ParseExpression().get();
  
  Expr* Op;
  
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;
  
  if(Tok.is(tok::l_paren)){
    LParenLoc = ConsumeParen();
    ExprResult OpResult = ParseExpression();
    if(OpResult.isInvalid()){
      Diag(Tok, diag::err_invalid_forall_op);
      SkipUntil(tok::l_brace);
      ConsumeToken();
      return StmtError();
    }
    if(Tok.isNot(tok::r_paren)){
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::l_brace);
      ConsumeToken();
      return StmtError();
    }
    RParenLoc = ConsumeParen();
  }
  else{
    Op = 0; 
  }
  
  StmtResult BodyResult = ParseStatement();
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_forall_body);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  Stmt* Body = BodyResult.get();
  
  
  return Actions.ActOnForAllStmt(ForAllLoc, Type, Ind, Mesh, LParenLoc,
                                 Op, RParenLoc, Body);
  */
}

StmtResult Parser::ParseRenderAllStatement(ParsedAttributes &attrs) {
  assert(Tok.is(tok::kw_renderall) && "Not a renderall stmt!");
  SourceLocation RenderAllLoc = ConsumeToken();  // eat the 'renderall'.
 
  RenderAllStmt::RenderAllType Type;
  
  switch(Tok.getKind()){
    case tok::kw_faces:
      Type = RenderAllStmt::Faces;
      break;
    case tok::kw_edges:
      Type = RenderAllStmt::Edges;
      break;
    case tok::kw_cells:
      Type = RenderAllStmt::Cells;
      break;
    default: {
      Diag(Tok, diag::err_expected_faces_edges_cells);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }
  
  ConsumeToken();
  
  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }
    
  Expr* Ind = ParseExpression().get();
  
  if(Tok.isNot(tok::kw_of)){
    Diag(Tok, diag::err_expected_of_kw);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  ConsumeToken();
  
  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  Expr* Mesh = ParseExpression().get();
  
  StmtResult BodyResult = ParseStatement();
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_renderall_body);
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  Stmt* Body = BodyResult.get();
  
  return Actions.ActOnRenderAllStmt(RenderAllLoc, Type, Ind, Mesh, Body);
}

