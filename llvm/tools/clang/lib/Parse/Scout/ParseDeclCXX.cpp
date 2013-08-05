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
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;


bool Parser::ParseMeshSpecifier(DeclSpec &DS, 
                                const ParsedTemplateInfo &TemplateInfo) {

  // the current lookahead token is tok::kw_uniform, tok::kw_rectlinear,
  // tok::kw_structured, or tok::kw_unstructured
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

  TemplateParameterLists* TemplateParams = TemplateInfo.TemplateParams;
  MultiTemplateParamsArg TParams;
  if (TemplateParams) {
    TParams = MultiTemplateParamsArg(&(*TemplateParams)[0], 
                                     TemplateParams->size());
  }

  MeshDecl* Dec;
  Dec = static_cast<MeshDecl*>(Actions.ActOnMeshDefinition(getCurScope(),
                               MeshType, MeshTypeLocation,
                               Name, NameLoc, TParams));
  Dec->completeDefinition();
  bool valid = ParseMeshBody(MeshLocation, Dec);
  if (valid) {
    unsigned DiagID;
    const char* PrevSpec;
    DS.SetTypeSpecType(DeclSpec::TST_uniform_mesh, MeshLocation, PrevSpec,
                       DiagID, Dec, true);
    return true;
  }

  return false;
}

// scout - Scout Mesh
// parse the body of a definition of a mesh, e.g:
// uniform mesh MyMesh {
///     <BODY>
// }
// return true on success

bool Parser::ParseMeshBody(SourceLocation StartLoc, MeshDecl* Dec) {

  PrettyDeclStackTraceEntry CrashInfo(Actions, Dec, StartLoc,
                                      "parsing Scout mesh body");

  SourceLocation LBraceLoc = ConsumeBrace();

  ParseScope MeshScope(this, Scope::ClassScope|Scope::DeclScope);
  Actions.ActOnMeshStartDefinition(getCurScope(), Dec);
  bool valid = true;

  llvm::SmallVector<Decl *, 32> FieldDecls;

  while(Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {

    if (Tok.is(tok::kw_cells)) {
      llvm::errs() << "mesh has cells\n";
      ConsumeToken();
      Dec->setHasCellData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << "cells";
        SkipUntil(tok::r_brace, true, true);
      }
      ConsumeToken();

    } else if (Tok.is(tok::kw_vertices)) {
      ConsumeToken();
      Dec->setHasVertexData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << "vertices";
        SkipUntil(tok::r_brace, true, true);
      }
      ConsumeToken();
    } else if (Tok.is(tok::kw_faces)) {
      ConsumeToken();
      Dec->setHasFaceData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << " faces";
        SkipUntil(tok::r_brace, true, true);
      }
    } else if (Tok.is(tok::kw_edges)) {
      ConsumeToken();
      Dec->setHasEdgeData(true);
      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon_after) << " edges";
        SkipUntil(tok::r_brace, true, true);
      }
    }

    ParsingDeclSpec DS(*this);

    struct ScoutFieldCallback : FieldCallback {
      Parser& P;
      Decl* MeshDecl;
      llvm::SmallVectorImpl<Decl*>& FieldDecls;
      bool externAlloc;

      ScoutFieldCallback(Parser& P, Decl* MeshDecl,
                         llvm::SmallVectorImpl<Decl*>& FieldDecls) :
          P(P), MeshDecl(MeshDecl), FieldDecls(FieldDecls) {}

      void invoke(ParsingFieldDeclarator& FD) {
        //llvm::outs() << "FieldLoc " << FieldLoc << "\n";
        // Install the declarator into the current MeshDecl.
        Decl* Field = P.Actions.ActOnMeshField(P.getCurScope(), MeshDecl,
                                   FD.D.getDeclSpec().getSourceRange().getBegin(),
                                   FD.D);

        MeshFieldDecl* FDecl = cast<MeshFieldDecl>(Field);
        FDecl->setImplicit(false);
        // SC_TODO - is this a potential bug?  FIXME -- PM 
        //FDecl->setExternAlloc(externAlloc);
        FieldDecls.push_back(Field);
        FD.complete(Field);
      }

      void setFieldExternAlloc(bool externalloc) {
        externAlloc = externalloc;
      }

    } Callback(*this, Dec, FieldDecls);

    if (Tok.getKind() == tok::kw_extern) {
      Diag(Tok, diag::err_extern_mesh_field);      
      ConsumeToken();      
    }

    ParseMeshDeclaration(DS, Callback);

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (Tok.is(tok::r_brace)) {
      ExpectAndConsume(tok::semi, diag::ext_expected_semi_decl_list);
    } else {
      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list);
      SkipUntil(tok::r_brace, true, true);
      if (Tok.is(tok::semi)) {
        ConsumeToken();
      }
    }
  }

  if (FieldDecls.empty()) {
    Diag(LBraceLoc, diag::warn_empty_mesh);
  }

  // scout - MERGE
  //SourceLocation RBraceLoc = MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  assert(Tok.is(tok::r_brace));
  ConsumeBrace();

  MeshScope.Exit();

  if (!valid) {
    return false;
  }

  return Actions.ActOnMeshFinish(StartLoc, Dec);
}

