//===----------------------------------------------------------------------===//
//
//  ndm - This file implements semantic analysis and AST building for 
//  Scout declarations/definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"

#include "clang/AST/DeclScout.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/DeclBase.h"

#include <iostream>

using namespace clang;
using namespace sema;

Decl* Sema::ActOnMeshDefinition(Scope* S,
                                tok::TokenKind MeshType, 
                                SourceLocation KWLoc,
                                IdentifierInfo* Name,
                                SourceLocation NameLoc){
  
  LookupResult LR(*this, Name, NameLoc, LookupTagName, Sema::NotForRedeclaration);
  
  DeclContext* DC = Context.getTranslationUnitDecl();
  
  std::cout << "creating mesh decl" << std::endl;
  
  // ndm - do we need to pass PrevDecl as the last parameter?
  
  return MeshDecl::Create(Context, Decl::Mesh, DC, 
                          KWLoc, NameLoc, 
                          Name, 0);
  
}

void Sema::ActOnMeshStartDefinition(Scope *S, Decl *MeshD) {

  std::cout << "a1" << std::endl;
  MeshDecl *Mesh = cast<MeshDecl>(MeshD);
  std::cout << "a2" << std::endl;
  
  // Enter the tag context.
  PushDeclContext(S, Mesh);
}

FieldDecl *Sema::HandleMeshField(Scope *S, MeshDecl *Mesh,
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
  
  // ndm - TODO - can this be taken out?
  
  // If the name is overloaded then get any declaration else get the single result
  NamedDecl *PrevDecl = Previous.isOverloadedResult() ?
  Previous.getRepresentativeDecl() : Previous.getAsSingle<NamedDecl>();
  
  if (PrevDecl && !isDeclInScope(PrevDecl, Mesh, S))
    PrevDecl = 0;
  
  SourceLocation TSSL = D.getSourceRange().getBegin();
  FieldDecl *NewFD
  = CheckMeshFieldDecl(II, T, TInfo, Mesh, Loc, TSSL, PrevDecl, &D);
  
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

FieldDecl *Sema::CheckMeshFieldDecl(DeclarationName Name, QualType T,
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
  
  bool ZeroWidth = false;
  
  // ndm - ok to pass null bitwidth?
  
  FieldDecl *NewFD = FieldDecl::Create(Context, Mesh, TSSL, Loc, II, T, TInfo,
                                       0, true, false);
  if (InvalidDecl)
    NewFD->setInvalidDecl();
  
  if (PrevDecl && !isa<TagDecl>(PrevDecl)) {
    Diag(Loc, diag::err_duplicate_member) << II;
    Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
    NewFD->setInvalidDecl();
  }
  
  // FIXME: We need to pass in the attributes given an AST
  // representation, not a parser representation.
  if (D)
    // FIXME: What to pass instead of TUScope?
    ProcessDeclAttributes(TUScope, NewFD, *D);
  
  // In auto-retain/release, infer strong retension for fields of
  // retainable type.
  if (getLangOptions().ObjCAutoRefCount && inferObjCARCLifetime(NewFD))
    NewFD->setInvalidDecl();
  
  if (T.isObjCGCWeak())
    Diag(Loc, diag::warn_attribute_weak_on_field);
  
  NewFD->setAccess(AS_public);
  return NewFD;
}

