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

#include "clang/AST/Scout/MeshDecl.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Parse/ParseDiagnostic.h"
#include <algorithm>
#include <cstring>
#include <functional>
using namespace clang;
using namespace sema;


Decl* Sema::ActOnMeshDefinition(Scope* S,
                                tok::TokenKind MeshType,
                                SourceLocation KWLoc,
                                IdentifierInfo* Name,
                                SourceLocation NameLoc,
                                MultiTemplateParamsArg TemplateParameterLists) {

  LookupResult LR(*this, Name, NameLoc, LookupTagName, Sema::NotForRedeclaration);


  switch(MeshType) {

    case tok::kw_uniform: {
      UniformMeshDecl* MD;
      MD = UniformMeshDecl::Create(Context, CurContext,
                                   KWLoc, NameLoc, Name, 0);
      PushOnScopeChains(MD, S, true);
      return MD;
    }

    case tok::kw_structured: {
      StructuredMeshDecl* USMD;
      USMD = StructuredMeshDecl::Create(Context, CurContext,
                                        KWLoc, NameLoc, Name, 0);
      PushOnScopeChains(USMD, S, true);
      return USMD;
    }

    case tok::kw_rectilinear:
      Diag(NameLoc, diag::err_mesh_not_implemented);
      return NULL;
      break;

    case tok::kw_unstructured:
      Diag(NameLoc, diag::err_mesh_not_implemented);
      return NULL;
      break;

    default:
      llvm_unreachable("Unknown mesh type");
      return NULL;
      break;
  }

  return NULL;
}

// scout - Scout Mesh field
Decl *Sema::ActOnMeshField(Scope *S, Decl *MeshD, SourceLocation DeclStart,
                           Declarator &D) {
  MeshFieldDecl *Res = HandleMeshField(S, cast_or_null<MeshDecl>(MeshD),
                                   DeclStart, D);
  return Res;
}

// scout - Scout Mesh
void Sema::ActOnMeshStartDefinition(Scope *S, Decl *MeshD) {
  MeshDecl *Mesh = cast<MeshDecl>(MeshD);

  // Enter the mesh context.
  PushDeclContext(S, Mesh);
}

// scout - Scout Mesh
MeshFieldDecl *Sema::HandleMeshField(Scope *S, MeshDecl *Mesh,
                                     SourceLocation DeclStart,
                                     Declarator &D) {

  IdentifierInfo *II = D.getIdentifier();
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();

  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, S);
  QualType T = TInfo->getType();

  // Check to see if this name was declared as a member previously
  LookupResult Previous(*this, II, Loc, LookupMemberName, ForRedeclaration);
  LookupName(Previous, S);
  assert((Previous.empty() || Previous.isOverloadedResult() ||
          Previous.isSingleResult())
         && "Lookup of member name should be either overloaded, single or null");

  // If the name is overloaded then get any declaration else get the single result
  NamedDecl *PrevDecl = Previous.isOverloadedResult() ?
  Previous.getRepresentativeDecl() : Previous.getAsSingle<NamedDecl>();

  if (PrevDecl && !isDeclInScope(PrevDecl, Mesh, S))
    PrevDecl = 0;

  SourceLocation TSSL = D.getSourceRange().getBegin();
  MeshFieldDecl *NewFD;
  NewFD = CheckMeshFieldDecl(II, T, TInfo, Mesh, Loc, TSSL, PrevDecl, &D);

  if (NewFD->isInvalidDecl())
    Mesh->setInvalidDecl();

  if (NewFD->isInvalidDecl() && PrevDecl) {
    // Don't introduce NewFD into scope; there's already something
    // with the same name in the same scope.
  } else if (II) {
    PushOnScopeChains(NewFD, S);
  } else
    Mesh->addDecl(NewFD);

  return NewFD;
}

// scout - Scout Mesh
MeshFieldDecl *Sema::CheckMeshFieldDecl(DeclarationName Name, QualType T,
                                        TypeSourceInfo *TInfo,
                                        MeshDecl *Mesh, SourceLocation Loc,
                                        SourceLocation TSSL,
                                        NamedDecl *PrevDecl,
                                        Declarator *D) {

  IdentifierInfo *II = Name.getAsIdentifierInfo();
  bool InvalidDecl = false;

  if (D) InvalidDecl = D->isInvalidType();

  if (T.isNull()) {
    InvalidDecl = true;
    T = Context.IntTy;
  }

  QualType EltTy = Context.getBaseElementType(T);
  if (!EltTy->isDependentType() &&
      RequireCompleteType(Loc, EltTy, diag::err_field_incomplete)) {
    Mesh->setInvalidDecl();
    InvalidDecl = true;
  }

  if (!InvalidDecl && RequireNonAbstractType(Loc, T,
                                             diag::err_abstract_type_in_decl,
                                             AbstractFieldType))
    InvalidDecl = true;

  // add mesh members
  MeshFieldDecl *NewFD = MeshFieldDecl::Create(Context, Mesh, TSSL, Loc, II, T, TInfo,
                                               0, true, ICIS_NoInit);

  if (InvalidDecl)
    NewFD->setInvalidDecl();

  if (PrevDecl && !isa<MeshDecl>(PrevDecl)) {
    Diag(Loc, diag::err_duplicate_member) << II;
    Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
    NewFD->setInvalidDecl();
  }

  if (D)
    ProcessDeclAttributes(TUScope, NewFD, *D);

  NewFD->setAccess(AS_public);
  return NewFD;
}

// scout - Mesh
// return true on success

bool Sema::ActOnMeshFinish(SourceLocation Loc, MeshDecl* Mesh){
  PopDeclContext(); // need this or we get BlockDecl in MeshDecl
  return IsValidDeclInMesh(Mesh);
}

bool Sema::IsValidMeshField(MeshFieldDecl* MFD){

  QualType QT = MFD->getType();
  const Type* T = QT.getTypePtr();

  // We don't allow pointers in the mesh description (this helps us
  // avoid aliasing issues in the mesh-oriented loops).
  if (T->isPointerType()) {
    Diag(MFD->getSourceRange().getBegin(), diag::err_pointer_field_mesh);
    return false;
  }

  if(const MeshType* MT = dyn_cast<MeshType>(T)){
    if (!IsValidDeclInMesh(MT->getDecl())) {
      Diag(MFD->getSourceRange().getBegin(),
           diag::err_pointer_field_mesh);
      return false;
    }
  } else if (const RecordType* RT = dyn_cast<RecordType>(T)) {
    if (!IsValidDeclInMesh(RT->getDecl())) {
      Diag(MFD->getSourceRange().getBegin(),
           diag::err_pointer_field_mesh);
      return false;
    }
  }
  return true;
}


bool Sema::IsValidDeclInMesh(Decl* D){

  if (MeshDecl* MD = dyn_cast<MeshDecl>(D)) {

    if (! MD->hasValidFieldData()) {
      Diag(MD->getSourceRange().getBegin(),
           diag::err_mesh_has_no_elements);
    }

    for(MeshDecl::field_iterator itr = MD->field_begin(),
        itrEnd = MD->field_end(); itr != itrEnd; ++itr){
      MeshFieldDecl* MFD = *itr;
      if (!IsValidMeshField(MFD)) {
        return false;
      }
    }
    return true;
  } else {
    assert(false && "passed non-mesh decl for validation");
  }

}