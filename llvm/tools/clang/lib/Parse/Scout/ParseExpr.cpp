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
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

static ForallMeshStmt::MeshElementType setMeshElementType(tok::TokenKind kind){
   switch(kind){
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

ExprResult Parser::ParseScoutQueryExpression(){
  // Upon entry we expect the input token to be on the 'from'
  // keyword -- we'll throw an assertion in just to make sure
  // we help maintain consistency from the caller(s).
  assert(Tok.getKind() == tok::kw_from && "expected input token to be 'from'");
  
  // Swallow the 'from' token...
  SourceLocation FromLoc = ConsumeToken();
  
  // At this point we should be sitting at the mesh element keyword
  // that identifies the locations on the mesh that are to be queried
  // over.  Keep track of the element token and its location (for later
  // use).  Also set the mesh element type we're processing so we can
  // refer to it later w/out having to query/translate token types...
  tok::TokenKind ElementToken = Tok.getKind();
  ConsumeToken();
  
  ForallMeshStmt::MeshElementType MeshElementType = setMeshElementType(ElementToken);
  if (MeshElementType == ForallMeshStmt::Undefined){
    Diag(Tok, diag::err_query_expected_mesh_element_kw);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  // We consumed the element token above and should now be
  // at the element identifier portion of the query; make
  // sure we have a valid identifier and bail if not...
  if (Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  IdentifierInfo* ElementIdentInfo = Tok.getIdentifierInfo();
  SourceLocation  ElementIdentLoc  = Tok.getLocation();
  ConsumeToken();
  
  // Next we should encounter the 'in' keyword...
  if (Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_forall_expected_kw_in);
    SkipUntil(tok::semi);
    return ExprError();
  }
  ConsumeToken();
  
  //if we are in scc-mode and in a function where the mesh was
  // passed as a parameter we will have a star here.
  bool meshptr = false;
  if(getLangOpts().ScoutC){
    if(Tok.is(tok::star)){
      ConsumeToken();
      meshptr = true;
    }
  }
  
  // Finally, we are at the identifier that specifies the mesh
  // that we are querying over.
  if (Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  IdentifierInfo* MeshIdentInfo = Tok.getIdentifierInfo();
  SourceLocation MeshIdentLoc  = Tok.getLocation();
  
  VarDecl* VD = LookupScoutVarDecl(MeshIdentInfo, MeshIdentLoc);
  
  if(VD == 0){
    return ExprError();
  }
  
  // If we are in scc-mode and inside a function then make sure
  // we have a *
  if(getLangOpts().ScoutC && isa<ParmVarDecl>(VD) && meshptr == false){
    Diag(Tok,diag::err_expected_star_mesh);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  const MeshType* RefMeshType = LookupMeshType(VD, MeshIdentInfo);
  
  if(RefMeshType == 0){
    Diag(MeshIdentLoc, diag::err_expected_a_mesh_type);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  DeclStmt* Init; //declstmt for forall implicit variable
  bool success = Actions.ActOnForallMeshRefVariable(getCurScope(),
                                                    MeshIdentInfo, MeshIdentLoc,
                                                    MeshElementType,
                                                    ElementIdentInfo,
                                                    ElementIdentLoc,
                                                    RefMeshType,
                                                    VD, &Init);
  if(!success){
    return ExprError();
  }
  
  ConsumeToken();
  
  MeshElementTypeDiag(MeshElementType, RefMeshType, MeshIdentLoc);
  
  if (Tok.isNot(tok::kw_select)){
    Diag(Tok, diag::err_query_expected_kw_select);
    SkipUntil(tok::semi);
    return ExprError();
  }
  SourceLocation SelectLoc = ConsumeToken();
  (void)SelectLoc; // suppress warning 
 
  ExprResult FieldResult = ParseExpression();
  
  if(FieldResult.isInvalid() || !isa<MemberExpr>(FieldResult.get())){
    Diag(Tok, diag::err_invalid_query_field);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  if (Tok.isNot(tok::kw_where)){
    Diag(Tok, diag::err_query_expected_kw_where);
    SkipUntil(tok::semi);
    return ExprError();
  }
  SourceLocation WhereLoc = ConsumeToken();
  (void)WhereLoc; // suppress warning 

  ExprResult PredicateResult = ParseExpression();
  
  if(PredicateResult.isInvalid()){
    Diag(Tok, diag::err_invalid_query_predicate);
    SkipUntil(tok::semi);
    return ExprError();
  }
  
  return Actions.ActOnQueryExpr(FromLoc,
                                VD,
                                FieldResult.get(),
                                PredicateResult.get());
}

ExprResult Parser::ParseSpecExpression(){
  if(Tok.is(tok::l_brace)){
    return ParseSpecObjectExpression();
  }
  else if(Tok.is(tok::l_square)){
    return ParseSpecArrayExpression();
  }
  
  return ParseSpecValueExpression();
}

ExprResult Parser::ParseSpecObjectExpression(){
  assert(Tok.is(tok::l_brace) && "expected '{'");

  ConsumeBrace();
  
  SpecObjectExpr* obj =
  cast<SpecObjectExpr>(Actions.ActOnSpecObjectExpr(Tok.getLocation()).get());
  
  for(;;){
    std::string key;
    
    if(Tok.is(tok::identifier)){
      IdentifierInfo* IdentInfo = Tok.getIdentifierInfo();
      key = IdentInfo->getName().str();
      ConsumeToken();
    }
    else if(Tok.is(tok::string_literal)){
      ExprResult StringResult = ParseStringLiteralExpression();
      StringLiteral* literal = cast<StringLiteral>(StringResult.get());
      key = literal->getString().str();
    }
    else{
      Diag(Tok, diag::err_spec_invalid_object_key);
      return ExprError();
    }
    
    if(Tok.isNot(tok::colon)){
      Diag(Tok, diag::err_spec_invalid_expected) << ":";
      return ExprError();
    }
    
    ConsumeToken();
    
    ExprResult valueResult = ParseSpecExpression();
    
    if(valueResult.isInvalid()){
      return ExprError();
    }
    
    SpecExpr* value = cast<SpecExpr>(valueResult.get());
    
    obj->insert(key, value);
    
    if(Tok.is(tok::r_brace)){
      ConsumeBrace();
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_spec_invalid_expected) << ",";
      return ExprError();
    }
    
    ConsumeToken();
  }
  
  return obj;
}

ExprResult Parser::ParseSpecValueExpression(){
  ExprResult result = ParseAssignmentExpression();
  
  if(result.isInvalid()){
    return ExprError();
  }
  
  return Actions.ActOnSpecValueExpr(result.get());
}

ExprResult Parser::ParseSpecArrayExpression(){
  assert(Tok.is(tok::l_square) && "expected '['");
  
  ConsumeBracket();
  
  SpecArrayExpr* array =
  cast<SpecArrayExpr>(Actions.ActOnSpecArrayExpr(Tok.getLocation()).get());
  
  for(;;){
    ExprResult valueResult = ParseSpecExpression();
    
    if(valueResult.isInvalid()){
      return ExprError();
    }
    
    SpecExpr* value = cast<SpecExpr>(valueResult.get());

    array->add(value);
    
    if(Tok.is(tok::r_square)){
      ConsumeBracket();
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_spec_invalid_expected) << ",";
      return ExprError();
    }
    
    ConsumeToken();
  }
  
  return array;
}
