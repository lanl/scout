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
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

void Parser::ParseMeshDeclaration(ParsingDeclSpec &DS,
                                  FieldCallback &Fields) {
  
  ParseSpecifierQualifierList(DS);

  // Read mesh-declarators until we find the semicolon.
  bool FirstDeclarator = true;
  while (1) {
    ParsingDeclRAIIObject PD(*this, ParsingDeclRAIIObject::NoParent);
    ParsingFieldDeclarator DeclaratorInfo(*this, DS);

    // Attributes are only allowed here on successive declarators.
    if (!FirstDeclarator)
      MaybeParseGNUAttributes(DeclaratorInfo.D);

    ParseDeclarator(DeclaratorInfo.D);

    // If attributes exist after the declarator, parse them.
    MaybeParseGNUAttributes(DeclaratorInfo.D);

    // We're done with this declarator;  invoke the callback.
    Fields.invoke(DeclaratorInfo);

    // If we don't have a comma, it is either the end of the list (a ';')
    // or an error, bail out.
    if (Tok.isNot(tok::comma)){
      return;
    }

    // Consume the comma.
    ConsumeToken();

    FirstDeclarator = false;
  }
}

// Tail end of mesh variable declaration (the bracket and beyond)
void Parser::ParseMeshVarBracketDeclarator(Declarator &D) {

  // get type info of this object
  DeclSpec& DS = D.getMutableDeclSpec();

  ParsedType parsedType = DS.getRepAsType();
  const UniformMeshType* umt = dyn_cast<UniformMeshType>(parsedType.get().getTypePtr());
  if (umt) {
    BalancedDelimiterTracker T(*this, tok::l_square);
    T.consumeOpen();

    // for uniform type it can be a comma-separated list of dimensions
    // parse mesh dimensions, e.g: [512,512]

    MeshType::MeshDimensions dims;

    ExprResult NumElements;

    for(;;) {
      NumElements = ParseConstantExpression(); // consumes it too
      // If there was an error parsing the assignment-expression, recover.
      // Maybe should print a diagnostic, tho.
      if (NumElements.isInvalid()) {
        // If the expression was invalid, skip it.
        SkipUntil(tok::r_square);
        StmtError();
      }
      dims.push_back(NumElements.get());
      
      if (Tok.is(tok::r_square)) {
        break;
      }

      if (Tok.is(tok::eof)) {
        Diag(Tok, diag::err_expected_lsquare);
        StmtError();
      }

      if (Tok.isNot(tok::comma)) {
        Diag(Tok, diag::err_expected_comma);
        SkipUntil(tok::r_square);
        SkipUntil(tok::semi);
        StmtError();
      }

      ConsumeToken();
    }

    T.consumeClose();

    // set dims on type
    ParsedAttributes attrs(AttrFactory);

    DeclaratorChunk DC =
      DeclaratorChunk::getUniformMesh(dims, T.getOpenLocation(), T.getCloseLocation());

    D.AddTypeInfo(DC, attrs, T.getCloseLocation());

  } // else if unstructured or any other mesh type, do something else
}

// scout - tail end of mesh variable declaration (the parenthesis and beyond)
void Parser::ParseMeshVarParenDeclarator(Declarator &D) {

  // get type info of this object
  DeclSpec& DS = D.getMutableDeclSpec();

  ParsedType parsedType = DS.getRepAsType();
  const UnstructuredMeshType* unsMT = dyn_cast<UnstructuredMeshType>(parsedType.get().getTypePtr());
  if(unsMT){

    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();

    // for unstructured type it can be a string for a filename from
    // which to read the geometry and values, e.g. MyMeshType mymesh("mesh_info.txt");

    ExprResult unsMeshFileName;

    if (isTokenStringLiteral()) {
      unsMeshFileName = ParseStringLiteralExpression(); // consumes it too

      // If there was an error parsing the assignment-expression, recover.
      // Maybe should print a diagnostic, tho.
      if (!unsMeshFileName.isUsable()) {
        // If the expression was invalid, skip it.
        SkipUntil(tok::r_paren);
        StmtError();
      }
    } else {
      Diag(Tok, diag::err_expected_string_literal);
      StmtError();
    }

    T.consumeClose();

    // set mesh file info for type
    ParsedAttributes attrs(AttrFactory);

    DeclaratorChunk DC = DeclaratorChunk::getUnstructuredMesh(unsMeshFileName.get(),
        T.getOpenLocation(), T.getCloseLocation());

    D.AddTypeInfo(DC, attrs, T.getCloseLocation());

  } // else if any other mesh type, do something else
}



// parse a mesh parameter declaration
// assumes on entry that the token stream looks like:
// [], [:], [::], and that we have already parsed a mesh type
//
void Parser::ParseMeshParameterDeclaration(DeclSpec& DS) {

  ParsedType parsedType = DS.getRepAsType();
  const MeshType* mt;
  mt = dyn_cast<MeshType>(parsedType.get().getTypePtr());
  assert(mt && "expected mesh type");
  const UniformMeshType *umt = reinterpret_cast<const UniformMeshType *>(mt);

  ConsumeBracket();
  size_t numDims;
  if(Tok.is(tok::r_square)) {
    numDims = 1;
  } else if(Tok.is(tok::colon)) {
    numDims = 2;
    ConsumeToken();
  } else if(Tok.is(tok::coloncolon)) {
    numDims = 3;
    ConsumeToken();
  } else {
    Diag(Tok, diag::err_expected_mesh_param_token);
    SkipUntil(tok::r_square);
    return;
  }

  if (Tok.isNot(tok::r_square)){
    Diag(Tok, diag::err_expected_mesh_param_token);
    SkipUntil(tok::r_square);
    return;
  } else {
    ConsumeBracket();
  }

  // SC_TODO - does this only handle integer constants?  Would be 
  // nice to support variable sizes for mesh dimensions. 
  MeshType::MeshDimensions dims;
  for(size_t i = 0; i < numDims; ++i) {
    dims.push_back(Actions.ActOnIntegerConstant(Tok.getLocation(), 0).get());
  }

  // SC_TODO: possible alignment problem?
  UniformMeshType* mdt = new UniformMeshType(umt->getDecl());

  mdt->setDimensions(dims);
  parsedType.set(QualType(mdt, 0));
  DS.UpdateTypeRep(parsedType);
}

void Parser::ParseWindowBracketDeclarator(Declarator &D) {
  
  // We've already seen the opening square bracket prior to calling
  // this function.  Set up the balanced delimiter tracker to take
  // care of the closing bracket details for us...
  BalancedDelimiterTracker T(*this, tok::l_square);
  T.consumeOpen();

  ExprResult NumElements;  
  llvm::SmallVector<Expr*, 2> Dims;
  
  while(1) {
    NumElements = ParseConstantExpression();
    if (NumElements.isInvalid()) {
      D.setInvalidType(true);
      SkipUntil(tok::r_square, StopAtSemi);
      return;
    } 
    Dims.push_back(NumElements.get());

    if (Dims.size() == 1) {
      if (Tok.isNot(tok::comma)) {
        Diag(Tok, diag::err_expected_comma);
        Diag(Tok, diag::warn_render_targets_2d);
        D.setInvalidType(true);
        SkipUntil(tok::r_square, StopAtSemi);
        return;
      }
      ConsumeToken();
    } else if (Dims.size() == 2) {
      if (Tok.isNot(tok::r_square)) {
        Diag(Tok, diag::err_expected_rsquare);
        D.setInvalidType(true);
        if (Tok.is(tok::comma))
          Diag(Tok, diag::warn_render_targets_2d);          
        SkipUntil(tok::r_square, StopAtSemi);        
        return;
      }
      T.consumeClose();
      break;
    } else {
      Diag(Tok, diag::err_too_many_dims);
      D.setInvalidType(true);
      SkipUntil(tok::semi);
      return;
    }
  }
  
  ParsedAttributes attrs(AttrFactory);
  MaybeParseCXX11Attributes(attrs);  
  D.AddTypeInfo(DeclaratorChunk::getWindow(Dims, T.getOpenLocation(),
                                           T.getCloseLocation()),
                attrs, T.getCloseLocation());
}

void Parser::ParseImageBracketDeclarator(Declarator &D) {
  
  // We've already seen the opening square bracket prior to calling
  // this function.  Set up the balanced delimiter tracker to take
  // care of the closing bracket details for us...
  BalancedDelimiterTracker T(*this, tok::l_square);
  T.consumeOpen();

  ExprResult NumElements;  
  llvm::SmallVector<Expr*, 2> Dims;
  
  while(1) {
    NumElements = ParseConstantExpression();
    if (NumElements.isInvalid()) {
      D.setInvalidType(true);
      SkipUntil(tok::r_square, StopAtSemi);
      return;
    } 
    Dims.push_back(NumElements.get());

    if (Dims.size() == 1) {
      if (Tok.isNot(tok::comma)) {
        Diag(Tok, diag::err_expected_comma);
        Diag(Tok, diag::warn_render_targets_2d);
        D.setInvalidType(true);
        SkipUntil(tok::r_square, StopAtSemi);
        return;
      }
      ConsumeToken();
    } else if (Dims.size() == 2) {
      if (Tok.isNot(tok::r_square)) {
        Diag(Tok, diag::err_expected_rsquare);
        D.setInvalidType(true);
        if (Tok.is(tok::comma))
          Diag(Tok, diag::warn_render_targets_2d);          
        SkipUntil(tok::r_square, StopAtSemi);        
        return;
      }
      T.consumeClose();
      break;
    } else {
      Diag(Tok, diag::err_too_many_dims);
      D.setInvalidType(true);
      SkipUntil(tok::semi);
      return;
    }
  }
  
  ParsedAttributes attrs(AttrFactory);
  MaybeParseCXX11Attributes(attrs);  
  D.AddTypeInfo(DeclaratorChunk::getImage(Dims, T.getOpenLocation(),
                                          T.getCloseLocation()),
                attrs, T.getCloseLocation());
}



// scout - parse a camera declaration
// return true on success
// these look like:

//camera cam {
//  near = 70.0;
//  far = 500.0;
//  fov = 40.0;
//  pos = float3(350.0, -100.0, 650.0);
//  lookat = float3(350.0, 200.0, 25.0);
//  up = float3(-1.0, 0.0, 0.0);
//};

#if 0
StmtResult
Parser::ParseCameraDeclaration(StmtVector &Stmts,
                               bool OnlyStatement){
  ConsumeToken();

  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    ConsumeToken();
    return StmtError();
  }

  IdentifierInfo* Name = Tok.getIdentifierInfo();
  SourceLocation NameLoc = ConsumeToken();

  if(Tok.isNot(tok::l_brace)){
    Diag(Tok, diag::err_expected_lbrace);

    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    ConsumeToken();
    return StmtError();
  }

  ConsumeBrace();

  typedef std::map<std::string, Expr*> ArgExprMap;

  ArgExprMap argExprMap;

  bool error = false;

  for(;;){
    if(Tok.is(tok::r_brace) || Tok.is(tok::eof)){
      break;
    }

    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);

      SkipUntil(tok::r_brace);
      SkipUntil(tok::semi);
      ConsumeToken();
      return StmtError();
    }

    IdentifierInfo* Arg = Tok.getIdentifierInfo();
    SourceLocation ArgLoc = ConsumeToken();
    (void)ArgLoc; //suppress warning

    if(Tok.isNot(tok::equal)){
      Diag(Tok, diag::err_expected_equal_after) << Arg->getName();

      SkipUntil(tok::r_brace);
      SkipUntil(tok::semi);
      ConsumeToken();
      return StmtError();
    }

    ConsumeToken();

    ExprResult argResult = ParseExpression();
    if(argResult.isInvalid()){
      error = true;
    }

    argExprMap[Arg->getName()] = argResult.get();

    if(Tok.isNot(tok::semi)){
      Diag(Tok, diag::err_expected_semi_camera_arg);

      SkipUntil(tok::r_brace);
      SkipUntil(tok::semi);
      ConsumeToken();
      return StmtError();
    }

    ConsumeToken();
  }

  assert(Tok.is(tok::r_brace) && "expected r_brace");

  ConsumeBrace();

  assert(Tok.is(tok::semi) && "expected semi");

  ConsumeToken();

  if(error){
    return StmtError();
  }

  std::string code;

  code = "scout::glCamera " + Name->getName().str() + "(";

  // put together the constructor call for glCamera

  ArgExprMap::iterator itr = argExprMap.find("fov");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "fov";
    error = true;
  }

  code += ToCPPCode(itr->second);
  code += ",";

  itr = argExprMap.find("pos");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "pos";
    error = true;
  }

  code += "scout::glfloat3(";
  code += ToCPPCode(itr->second) + ".x, ";
  code += ToCPPCode(itr->second) + ".y, ";
  code += ToCPPCode(itr->second) + ".z ";
  code += "), ";

  itr = argExprMap.find("lookat");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "lookat";
    error = true;
  }

  code += "scout::glfloat3(";
  code += ToCPPCode(itr->second) + ".x, ";
  code += ToCPPCode(itr->second) + ".y, ";
  code += ToCPPCode(itr->second) + ".z ";
  code += "), ";

  itr = argExprMap.find("up");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "up";
    error = true;
  }

  code += "scout::glfloat3(";
  code += ToCPPCode(itr->second) + ".x, ";
  code += ToCPPCode(itr->second) + ".y, ";
  code += ToCPPCode(itr->second) + ".z ";
  code += "), ";

  itr = argExprMap.find("near");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "near";
    error = true;
  }

  code += ToCPPCode(itr->second);
  code += ", ";

  itr = argExprMap.find("far");
  if(itr == argExprMap.end()){
    Diag(Tok, diag::err_missing_field_window_decl) << "far";
    error = true;
  }

  code += ToCPPCode(itr->second);
  code += ");";

  if(error){
    return StmtError();
  }

  InsertCPPCode(code, NameLoc);

  return ParseStatementOrDeclaration(Stmts, OnlyStatement);
}
#endif
