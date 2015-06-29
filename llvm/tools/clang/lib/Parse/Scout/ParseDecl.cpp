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
#include "clang/AST/ASTContext.h"

using namespace clang;

bool Parser::ParseMeshSpecifier(DeclSpec &DS,
                                const ParsedTemplateInfo &TI) {
  
  tok::TokenKind MeshType = Tok.getKind();
  SourceLocation MeshTypeLocation = ConsumeToken();
  
  if (Tok.isNot(tok::kw_mesh)) {
    Diag(Tok, diag::err_expected_mesh_kw);
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return false;
  }
  
  SourceLocation MeshLocation = ConsumeToken();
  
  // parse mesh name
  IdentifierInfo* Name;
  SourceLocation NameLoc;
  if (Tok.is(tok::identifier)) {
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  } else {
    Diag(Tok, diag::err_expected_ident);
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return false;
  }
  
  if (Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_lbrace);
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return false;
  }
  
  TemplateParameterLists* TemplateParams = TI.TemplateParams;
  MultiTemplateParamsArg TParams;
  if (TemplateParams) {
    TParams = MultiTemplateParamsArg(&(*TemplateParams)[0],
                                     TemplateParams->size());
  }
  
  const char *PrevSpec = 0;
  unsigned DiagID;
  
  const clang::PrintingPolicy &Policy =
  Actions.getASTContext().getPrintingPolicy();
  
  switch(MeshType) {
      
    case tok::kw_uniform: {
      UniformMeshDecl *UMD;
      UMD = static_cast<UniformMeshDecl*>(
                                          Actions.ActOnMeshDefinition(getCurScope(),
                                                                      MeshType, MeshTypeLocation,
                                                                      Name, NameLoc, TParams));
      UMD->completeDefinition();
      if (ParseMeshBody(MeshLocation, UMD, DeclSpec::TST_uniform_mesh)) {
        DS.SetTypeSpecType(DeclSpec::TST_uniform_mesh,
                           MeshLocation, PrevSpec,
                           DiagID, UMD, true, Policy);
        return true;
      } else {
        return false;
      }
    }
      break;

    case tok::kw_ALE: {
      ALEMeshDecl *AMD;
      AMD = static_cast<ALEMeshDecl*>(
                                          Actions.ActOnMeshDefinition(getCurScope(),
                                                                      MeshType, MeshTypeLocation,
                                                                      Name, NameLoc, TParams));
      AMD->completeDefinition();
      if (ParseMeshBody(MeshLocation, AMD, DeclSpec::TST_ALE_mesh)) {
        DS.SetTypeSpecType(DeclSpec::TST_ALE_mesh,
                           MeshLocation, PrevSpec,
                           DiagID, AMD, true, Policy);
        return true;
      } else {
        return false;
      }
    }
      break;
      
    case tok::kw_rectilinear: {
      RectilinearMeshDecl *RMD;
      RMD = static_cast<RectilinearMeshDecl*>(
                                              Actions.ActOnMeshDefinition(getCurScope(),
                                                                          MeshType, MeshTypeLocation,
                                                                          Name, NameLoc, TParams));
      RMD->completeDefinition();
      if (ParseMeshBody(MeshLocation, RMD, DeclSpec::TST_rectilinear_mesh)) {
        DS.SetTypeSpecType(DeclSpec::TST_rectilinear_mesh,
                           MeshLocation, PrevSpec,
                           DiagID, RMD, true, Policy);
        return true;
      } else {
        return false;
      }
    }
      break;
      
    case tok::kw_structured: {
      StructuredMeshDecl *SMD;
      SMD = static_cast<StructuredMeshDecl*>(
                                             Actions.ActOnMeshDefinition(getCurScope(),
                                                                         MeshType, MeshTypeLocation,
                                                                         Name, NameLoc, TParams));
      SMD->completeDefinition();
      if (ParseMeshBody(MeshLocation, SMD, DeclSpec::TST_structured_mesh)) {
        DS.SetTypeSpecType(DeclSpec::TST_structured_mesh,
                           MeshLocation, PrevSpec,
                           DiagID, SMD, true, Policy);
        return true;
      } else {
        return false;
      }
    }
      break;
      
    case tok::kw_unstructured: {
      UnstructuredMeshDecl *USMD;
      USMD = static_cast<UnstructuredMeshDecl*>(
                                                Actions.ActOnMeshDefinition(getCurScope(),
                                                                            MeshType, MeshTypeLocation,
                                                                            Name, NameLoc, TParams));
      USMD->completeDefinition();
      if (ParseMeshBody(MeshLocation, USMD, DeclSpec::TST_unstructured_mesh)) {
        DS.SetTypeSpecType(DeclSpec::TST_unstructured_mesh,
                           MeshLocation, PrevSpec, 
                           DiagID, USMD, true, Policy);
        return true;
      } else {
        return false;
      }
    }
      break;
      
    default:
      llvm_unreachable("unrecognized mesh token kind");
      return false;
      break;
  }
}

bool Parser::ParseMeshBody(SourceLocation StartLoc, MeshDecl* Dec, TypeSpecifierType typeSpecType) {
  
  PrettyDeclStackTraceEntry CrashInfo(Actions, Dec, StartLoc,
                                      "parsing Scout mesh body");
  
  SourceLocation LBraceLoc = ConsumeBrace();
  
  ParseScope MeshScope(this, Scope::ClassScope|Scope::DeclScope);
  Actions.ActOnMeshStartDefinition(getCurScope(), Dec);
  bool valid = true;
  
  llvm::SmallVector<Decl *, 32> FieldDecls;

  // If it's an ALEmesh type, before we actually parse the fields, 
  // add in some built-in ones for variable vertex positions.

  //ParsingDeclSpec DS(*this);
  //const Type *Ty = DS.getRepAsType().get().getCanonicalType().getTypePtr();
  if (typeSpecType==TST_ALE_mesh) {

    // x movable position field
    Decl* field = Actions.ActOnBuiltinMeshField(Dec, StringRef("__x"), Actions.Context.FloatTy->getTypePtr());
    MeshFieldDecl* FDecl = cast<MeshFieldDecl>(field);
    FDecl->setImplicit(false);
    FDecl->setVertexLocated(true);
    // SC_TODO - is this a potential bug?  FIXME -- PM
    //FDecl->setExternAlloc(externAlloc);
    FieldDecls.push_back(field);
    // not sure I need this
    // FD.complete(field);

    // y movable position field
    field = Actions.ActOnBuiltinMeshField(Dec, StringRef("__y"), Actions.Context.FloatTy->getTypePtr());
    FDecl = cast<MeshFieldDecl>(field);
    FDecl->setImplicit(false);
    FDecl->setVertexLocated(true);
    FieldDecls.push_back(field);

    // z movable position field
    field = Actions.ActOnBuiltinMeshField(Dec, StringRef("__z"), Actions.Context.FloatTy->getTypePtr());
    FDecl = cast<MeshFieldDecl>(field);
    FDecl->setImplicit(false);
    FDecl->setVertexLocated(true);
    FieldDecls.push_back(field);

  }

  while(Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    
    if (Tok.is(tok::kw_cells)) {
      MFK = Cell;
      ConsumeToken();
      Dec->setHasCellData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << "cells";
        SkipUntil(tok::r_brace, StopAtSemi|StopBeforeMatch);
      }
      ConsumeToken();
      
    } else if (Tok.is(tok::kw_vertices)) {
      MFK = Vertex;
      ConsumeToken();
      Dec->setHasVertexData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << "vertices";
        SkipUntil(tok::r_brace, StopAtSemi|StopBeforeMatch);
      }
      ConsumeToken();
    } else if (Tok.is(tok::kw_faces)) {
      MFK = Face;
      ConsumeToken();
      Dec->setHasFaceData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << " faces";
        SkipUntil(tok::r_brace, StopAtSemi|StopBeforeMatch);
      }
      ConsumeToken();
    } else if (Tok.is(tok::kw_edges)) {
      MFK = Edge;
      ConsumeToken();
      Dec->setHasEdgeData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << " edges";
        SkipUntil(tok::r_brace, StopAtSemi|StopBeforeMatch);
      }
      ConsumeToken();
    }
    
    ParsingDeclSpec DS(*this);
    
    auto CFieldCallback = [&](ParsingFieldDeclarator &FD) {
      // Install the declarator into the current MeshDecl.
      Decl* Field = Actions.ActOnMeshField(getCurScope(), Dec,
                                           FD.D.getDeclSpec().getSourceRange().getBegin(),
                                           FD.D);
      
      MeshFieldDecl* FDecl = cast<MeshFieldDecl>(Field);
      
      FDecl->setImplicit(false);
      
      if (getMeshFieldKind() == Cell) {
        FDecl->setCellLocated(true);
      } else if (getMeshFieldKind() == Vertex) {
        FDecl->setVertexLocated(true);
      } else if (getMeshFieldKind() == Edge) {
        FDecl->setEdgeLocated(true);
      } else if (getMeshFieldKind() == Face) {
        FDecl->setFaceLocated(true);
      } else {
        FDecl->setCellLocated(false);
        FDecl->setVertexLocated(false);
        FDecl->setEdgeLocated(false);
        FDecl->setFaceLocated(false);
      }
      // SC_TODO - is this a potential bug?  FIXME -- PM
      //FDecl->setExternAlloc(externAlloc);
      FieldDecls.push_back(Field);
      FD.complete(Field);
    };
    
    if (Tok.getKind() == tok::kw_extern) {
      Diag(Tok, diag::err_extern_mesh_field);
      ConsumeToken();
    }
  
    // Parses mesh field declarations using callback 
    ParseMeshDeclaration(DS, CFieldCallback);
    
    
    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (Tok.is(tok::r_brace)) {
      ExpectAndConsume(tok::semi, diag::ext_expected_semi_decl_list);
    } else {
      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list);
      SkipUntil(tok::r_brace, StopAtSemi|StopBeforeMatch);
      if (Tok.is(tok::semi)) {
        ConsumeToken();
      }
    }
  }
  
  if (FieldDecls.empty()) {
    Diag(LBraceLoc, diag::warn_empty_mesh);
  }
  
  if (Tok.is(tok::r_brace)) {
    ConsumeBrace();
  } else {
    Diag(Tok, diag::err_expected_rbrace);
  }
  
  MeshScope.Exit();
  
  if (!valid) {
    return false;
  }
  
  return Actions.ActOnMeshFinishDefinition(getCurScope(), Dec, StartLoc);
  
}

void Parser::ParseMeshDeclaration(ParsingDeclSpec &DS,
  llvm::function_ref<void(ParsingFieldDeclarator &)> FieldsCallback) {
  
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
    FieldsCallback(DeclaratorInfo);

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
// for Uniform Mesh
void Parser::ParseMeshVarBracketDeclarator(Declarator &D) {
  // get type info of this object
  DeclSpec& DS = D.getMutableDeclSpec();

  const Type *Ty = DS.getRepAsType().get().getCanonicalType().getTypePtr();
  if (Ty->isUniformMeshType() || Ty->isALEMeshType()) {
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
        D.setInvalidType(true);
        SkipUntil(tok::r_square, StopAtSemi);
        StmtError();
        return;
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

    DeclaratorChunk DC;
    if (Ty->isUniformMeshType()) {
      DC = DeclaratorChunk::getUniformMesh(dims, T.getOpenLocation(), T.getCloseLocation());
    }
    if (Ty->isALEMeshType()) {
      DC = DeclaratorChunk::getALEMesh(dims, T.getOpenLocation(), T.getCloseLocation());
    }

    D.AddTypeInfo(DC, attrs, T.getCloseLocation());

  } // else if unstructured or any other mesh type, do something else
}

// scout - tail end of mesh variable declaration (the parenthesis and beyond)
// for Unstructured Mesh
void Parser::ParseMeshVarParenDeclarator(Declarator &D) {

  // get type info of this object
  DeclSpec& DS = D.getMutableDeclSpec();

  const Type *Ty = DS.getRepAsType().get().getCanonicalType().getTypePtr();
  if(Ty->isUnstructuredMeshType()) {

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
      // We really should never make it here...
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
      // We really should never make it here... 
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

// only allow pass by pointer for scc
void Parser::ParseMeshParameterDeclaration(DeclSpec& DS) {

  ParsedType parsedType = DS.getRepAsType();
  if(!isa<MeshType>(parsedType.get().getTypePtr()))
        assert(false && "expected mesh type");

  if(getLangOpts().ScoutC && Tok.isNot(tok::star)) {
    Diag(Tok, diag::err_expected_mesh_param_star);
    SkipUntil(tok::semi);
    return;
  }
}

bool Parser::ParseFrameSpecifier(DeclSpec &DS) {
  assert(Tok.is(tok::kw_frame) && "expected frame keyword");
  
  SourceLocation FrameLoc = ConsumeToken();
  
  IdentifierInfo* Name;
  SourceLocation NameLoc;
  
  if(Tok.is(tok::identifier)){
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  }
  else{
    Diag(Tok, diag::err_expected_ident);
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return false;
  }
  
  if(Tok.isNot(tok::l_brace)){
    Diag(Tok, diag::err_frame_expected_specifier);
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return false;
  }
  
  FrameDecl* FD =
  static_cast<FrameDecl*>(Actions.ActOnFrameDefinition(getCurScope(), FrameLoc,
                                                       Name, NameLoc));
  
  ParseScope FrameScope(this, Scope::ControlScope|Scope::DeclScope);
  
  Actions.InitFrameDefinitions(getCurScope(), FD);
  
  ExprResult Result = ParseSpecObjectExpression();
  if(Result.isInvalid()){
    Actions.PopFrameContext(FD);
    return false;
  }
  
  if(!Actions.InitFrame(getCurScope(), FD, Result.get())){
    Actions.PopFrameContext(FD);
    return false;
  }

  FrameScope.Exit();
  
  Actions.ActOnFrameFinishDefinition(FD);
  
  Actions.PopFrameContext(FD);
  
  const char* PrevSpec = 0;
  unsigned DiagID;
  
  const clang::PrintingPolicy &Policy =
  Actions.getASTContext().getPrintingPolicy();
  
  DS.SetTypeSpecType(DeclSpec::TST_frame,
                     FrameLoc, PrevSpec,
                     DiagID, FD, true, Policy);
  
  return true;
}
