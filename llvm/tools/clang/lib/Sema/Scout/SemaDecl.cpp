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

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Scout/MeshDecl.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
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

  LookupResult LR(*this, Name, NameLoc, LookupMeshName, Sema::NotForRedeclaration);


  switch(MeshType) {

    case tok::kw_uniform: {
      //llvm::errs() << "create uniform mesh " << Name->getName() << "\n";
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
Decl *Sema::ActOnMeshField(Scope *S, Decl *MeshD,
                           SourceLocation DeclStart,
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
  MeshFieldDecl *NewFD = MeshFieldDecl::Create(Context, Mesh,
                                               TSSL, Loc, II, T,
                                               TInfo, 0, true,
                                               ICIS_NoInit);

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

bool Sema::ActOnMeshFinishDefinition(Scope *S, Decl *MeshD,
                                    SourceLocation RBraceLoc) {
  MeshDecl *Mesh = cast<MeshDecl>(MeshD);
  Mesh->setRBraceLoc(RBraceLoc);

  // Make sure we "complete" the definition even it is invalid.
  if (Mesh->isBeingDefined()) {
    assert(Mesh->isInvalidDecl() && "We should already have completed it");
    if (UniformMeshDecl *UMD = dyn_cast<UniformMeshDecl>(Mesh))
      UMD->completeDefinition();
  }
  // Exit this scope of this mesh's definition.
  PopDeclContext();

  if (!Mesh->isInvalidDecl())
    Consumer.HandleMeshDeclDefinition(Mesh);

  return IsValidMeshDecl(Mesh);
}


// We don't allow pointers in the mesh description (this helps us
  // avoid aliasing issues in the mesh-oriented loops).
bool Sema::isPointerInMesh(const Type *T, SourceLocation Loc) {
  if (T->isPointerType()) {
    Diag(Loc, diag::err_pointer_field_mesh);
    return true;
  }
  return false;
}

// We don't allow pointers inside records the mesh description (this helps us
// avoid aliasing issues in the mesh-oriented loops).
bool Sema::isValidRecordInMesh(const Type *T, SourceLocation Loc) {
  if (const RecordType* RT = dyn_cast<RecordType>(T)) {
    if (!IsValidRecordDeclInMesh(RT->getDecl())) {
      Diag(Loc, diag::err_pointer_field_mesh);
      return false;
    }
  }
  return true;
}

bool Sema::IsValidMeshField(MeshFieldDecl* MFD) {

  const Type* T = MFD->getType().getCanonicalType().getTypePtr();
  SourceLocation Loc = MFD->getSourceRange().getBegin();

  if (isPointerInMesh(T, Loc)) return false;

  // nested mesh
  if(const MeshType* MT = dyn_cast<MeshType>(T)) {
    if (!IsValidMeshDecl(MT->getDecl())) {
      Diag(Loc, diag::err_pointer_field_mesh);
      return false;
    }
  }
  // struct in mesh
  if (!isValidRecordInMesh(T, Loc)) return false;

  return true;
}


bool Sema::IsValidRecordField(FieldDecl* FD) {
  const Type* T = FD->getType().getCanonicalType().getTypePtr();
  SourceLocation Loc = FD->getSourceRange().getBegin();

  if (isPointerInMesh(T, Loc)) return false;
  if (!isValidRecordInMesh(T, Loc)) return false;
  return true;
}

// look through all fields in mesh for validity
bool Sema::IsValidMeshDecl(MeshDecl* MD) {
  if (! MD->hasValidFieldData()) {
    Diag(MD->getSourceRange().getBegin(),
        diag::err_mesh_has_no_elements);
    return false;
  }

  for(MeshDecl::field_iterator itr = MD->field_begin(),
      itrEnd = MD->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* MFD = *itr;
    if (!IsValidMeshField(MFD)) return false;
  }
  return true;
}

// look through all fields  record inside mesh for validity
bool Sema::IsValidRecordDeclInMesh(RecordDecl* RD) {
  for(RecordDecl::field_iterator itr = RD->field_begin(),
      itrEnd = RD->field_end(); itr != itrEnd; ++itr){
    FieldDecl* FD = *itr;
    if (!IsValidRecordField(FD)) return false;
  }
  return true;
}

Decl* Sema::ActOnFrameDefinition(Scope* S,
                                 SourceLocation FrameLoc,
                                 IdentifierInfo* Name,
                                 SourceLocation NameLoc,
                                 MultiTemplateParamsArg TemplateParameterLists){
  
  LookupResult LR(*this, Name, NameLoc, LookupFrameName, Sema::NotForRedeclaration);
  
  FrameDecl* FD =
  FrameDecl::Create(Context, CurContext, FrameLoc, NameLoc, Name, 0);
  
  PushOnScopeChains(FD, S, true);
  
  FD->completeDefinition();
    
  PushDeclContext(S, FD);
  
  QualType vt = Context.getFrameVarType(0);
  
  AddFrameVarType(S, FD, "Timestep", Context.IntTy);
  AddFrameVarType(S, FD, "Temperature", Context.DoubleTy);
  
  AddFrameFunction(S, FD, "sum", vt, {vt});
  
  return FD;
}

void Sema::AddFrameVarType(Scope* Scope,
                           FrameDecl* FD,
                           const char* Name,
                           QualType Type){
  VarDecl* VD =
  VarDecl::Create(Context, FD, SourceLocation(), SourceLocation(),
                  PP.getIdentifierInfo(Name), Type,
                  Context.getTrivialTypeSourceInfo(Type),
                  SC_None);
  FD->addVarType(VD);
  
  PushOnScopeChains(VD, Scope, true);
}

void Sema::AddFrameFunction(Scope* Scope,
                            FrameDecl* FD,
                            const char* Name,
                            QualType RetTy,
                            std::vector<QualType> ArgTys){
  using namespace std;
  
  FunctionProtoType::ExtProtoInfo EPI;

  QualType FT = Context.getFunctionType(RetTy, ArgTys, EPI);
  
  FunctionDecl* F =
  FunctionDecl::Create(Context, FD, SourceLocation(), SourceLocation(),
                       DeclarationName(PP.getIdentifierInfo(Name)),
                       FT, Context.getTrivialTypeSourceInfo(FT),
                       SC_Extern, true);
    
  vector<ParmVarDecl*> params;
  
  int i = 0;
  char nb[16];
  
  for(QualType t : ArgTys){
    sprintf(nb, "p%d", i);
    
    ParmVarDecl* P =
    ParmVarDecl::Create(Context, F, SourceLocation(), SourceLocation(),
                        PP.getIdentifierInfo(nb),
                        t, Context.getTrivialTypeSourceInfo(t),
                        SC_Extern, 0);
    
    params.push_back(P);
    
    ++i;
  }
   
  F->setParams(params);
  
  F->setBody(new (Context) NullStmt(SourceLocation()));
  
  PushOnScopeChains(F, Scope, true);
}

void Sema::PopFrameContext(FrameDecl* F){
  PopDeclContext();
}

bool Sema::InitFrame(Scope* Scope, FrameDecl* F, Expr* SE){
  using namespace std;
  
  bool valid = true;
  
  SpecObjectExpr* Spec = static_cast<SpecObjectExpr*>(SE);
  
  auto& m = Spec->memberMap();
  
  for(auto& itr : m){
    const string& k = itr.first;
    
    if(!SpecExpr::isSymbol(k)){
      Diag(Spec->getKeyLoc(k), diag::err_invalid_frame_spec) << "invalid frame variable";
      valid = false;
    }
    
    SpecExpr* v = itr.second;

    SpecObjectExpr* vo = v->toObject();
    
    if(!vo){
      Diag(v->getLocStart(), diag::err_invalid_frame_spec) << "invalid frame definition";
      valid = false;
    }

    VarDecl* vd = 0;
    
    SpecExpr* t = vo->get("type");

    if(t){
      DeclRefExpr* dr = dyn_cast_or_null<DeclRefExpr>(t->toExpr());
      if(!dr){
        Diag(v->getLocStart(), diag::err_invalid_frame_spec) << "expeceted type specifier";
        valid = false;
      }
      else{
        vd = cast<VarDecl>(dr->getDecl());
        if(!F->hasVarType(vd)){
          Diag(v->getLocStart(), diag::err_invalid_frame_spec) << "invalid type";
          valid = false;
        }
      }
    }
    else{
      vd = F->getVarType(SpecExpr::toUpper(k));
      if(!vd){
        Diag(v->getLocStart(), diag::err_invalid_frame_spec) << "no valid default type";
        valid = false;
      }
    }
    
    if(vd){
      QualType vt = Context.getFrameVarType(vd->getType().getTypePtr());
      
      VarDecl* VD =
      VarDecl::Create(Context, F, SourceLocation(), SourceLocation(),
                      PP.getIdentifierInfo(k), vt,
                      Context.getTrivialTypeSourceInfo(vt),
                      SC_Static);
      
      PushOnScopeChains(VD, Scope, true);
      F->addVar(k, VD);
    }
  }
  
  if(valid){
    F->setSpec(Spec);
  }
  
  return valid;
}

bool Sema::ActOnFrameFinishDefinition(Decl* D){
  FrameDecl* FD = cast<FrameDecl>(D);

  if(!FD->isInvalidDecl()){
    Consumer.HandleFrameDeclDefinition(FD);
  }

  return true;
}
