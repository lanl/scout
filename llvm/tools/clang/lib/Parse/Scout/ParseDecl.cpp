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
// for Uniform Mesh
void Parser::ParseMeshVarBracketDeclarator(Declarator &D) {

  // get type info of this object
  DeclSpec& DS = D.getMutableDeclSpec();

  const Type *Ty = DS.getRepAsType().get().getCanonicalType().getTypePtr();
  if (Ty->isUniformMeshType()) {
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


// Set the parsed type to be the correct mesh type and rank
// we need this so we can do the forall codegen.
void Parser:: SetMeshParameterParsedType(size_t rank, ParsedType *parsedType) {

  // just set dimensions to zero we only need the rank to be correct
  MeshType::MeshDimensions dims;
  for(size_t i = 0; i < rank; ++i) {
    dims.push_back(Actions.ActOnIntegerConstant(Tok.getLocation(), 0).get());
  }

  // set the mesh type and dimesions
  const MeshType* MT = dyn_cast<MeshType>(parsedType->get().getTypePtr());
  if(MT->isUniformMeshType()) {
    UniformMeshType *UMT = const_cast<UniformMeshType *>(cast<UniformMeshType>(MT));
    UMT->setDimensions(dims);
    parsedType->set(QualType(UMT, 0));
  } else if(MT->isRectilinearMeshType()) {
    RectilinearMeshType *RMT = const_cast<RectilinearMeshType *>(cast<RectilinearMeshType>(MT));
    RMT->setDimensions(dims);
    parsedType->set(QualType(RMT, 0));
  } else if(MT->isStructuredMeshType()) {
    StructuredMeshType *SMT = const_cast<StructuredMeshType *>(cast<StructuredMeshType>(MT));
    SMT->setDimensions(dims);
    parsedType->set(QualType(SMT, 0));
  } else if(MT->isUnstructuredMeshType()) {
    UnstructuredMeshType *USMT = const_cast<UnstructuredMeshType *>(cast<UnstructuredMeshType>(MT));
    USMT->setDimensions(dims);
    parsedType->set(QualType(USMT, 0));
  }
}

void Parser::ParseMeshParameterDeclaration(DeclSpec& DS) {

  if (Tok.is(tok::l_square)) {
    ParsedType parsedType = DS.getRepAsType();

    if(!isa<MeshType>(parsedType.get().getTypePtr()))
      assert(false && "expected mesh type");

    ConsumeBracket();
    size_t rank;
    if(Tok.is(tok::r_square)) {
      rank = 1;
    } else if(Tok.is(tok::colon)) {
      rank = 2;
      ConsumeToken();
    } else if(Tok.is(tok::coloncolon)) {
      rank = 3;
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_mesh_param_token);
      SkipUntil(tok::r_square);
      return;
    }

    if (Tok.isNot(tok::r_square)) {
      Diag(Tok, diag::err_expected_mesh_param_token);
      SkipUntil(tok::r_square);
      return;
    } else {
      ConsumeBracket();
    }

    // Set the parsed type to be the correct mesh type and rank
    SetMeshParameterParsedType(rank, &parsedType);
    DS.UpdateTypeRep(parsedType);
  }
  if (Tok.isNot(tok::amp) && Tok.isNot(tok::star)) {
    Diag(Tok, diag::err_expected_mesh_param_star_amp);
    SkipUntil(tok::r_paren);
    return;
  }
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

