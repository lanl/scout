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

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include "clang/AST/Scout/MeshDecl.h"

using namespace clang;
using namespace sema;

ExprResult LookupMemberExpr(Sema &S, LookupResult &R,
    ExprResult &BaseExpr, bool &IsArrow,
    SourceLocation OpLoc, CXXScopeSpec &SS,
    Decl *ObjCImpDecl, bool HasTemplateArgs);

ExprResult
BuildFieldReferenceExpr(Sema &S, Expr *BaseExpr, bool IsArrow,
                        SourceLocation OpLoc,
                        const CXXScopeSpec &SS, FieldDecl *Field,
                        DeclAccessPair FoundDecl,
                        const DeclarationNameInfo &MemberNameInfo);
ExprResult
BuildMSPropertyRefExpr(Sema &S, Expr *BaseExpr, bool IsArrow,
                       const CXXScopeSpec &SS,
                       MSPropertyDecl *PD,
                       const DeclarationNameInfo &NameInfo);

static MemberExpr *BuildMemberExpr(
                                   Sema &SemaRef, ASTContext &C, Expr *Base, bool isArrow,
                                   SourceLocation OpLoc, const CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
                                   ValueDecl *Member, DeclAccessPair FoundDecl,
                                   const DeclarationNameInfo &MemberNameInfo, QualType Ty, ExprValueKind VK,
                                   ExprObjectKind OK, const TemplateArgumentListInfo *TemplateArgs = nullptr) {
  assert((!isArrow || Base->isRValue()) && "-> base must be a pointer rvalue");
  MemberExpr *E = MemberExpr::Create(
                                     C, Base, isArrow, OpLoc, SS.getWithLocInContext(C), TemplateKWLoc, Member,
                                     FoundDecl, MemberNameInfo, TemplateArgs, Ty, VK, OK);
  SemaRef.MarkMemberReferenced(E);
  return E;
}

ExprResult
BuildMeshFieldReferenceExpr(Sema &S, Expr *BaseExpr, bool IsArrow,
    const CXXScopeSpec &SS, MeshFieldDecl *Field,
    DeclAccessPair FoundDecl,
    const DeclarationNameInfo &MemberNameInfo) {

  // x.a is an l-value if 'a' has a reference type. Otherwise:
  // x.a is an l-value/x-value/pr-value if the base is (and note
  //   that *x is always an l-value), except that if the base isn't
  //   an ordinary object then we must have an rvalue.
  ExprValueKind VK = VK_LValue;
  ExprObjectKind OK = OK_Ordinary;
  if (!IsArrow) {
    if (BaseExpr->getObjectKind() == OK_Ordinary)
      VK = BaseExpr->getValueKind();
    else
      VK = VK_RValue;
  }

  if (VK != VK_RValue && Field->isBitField())
    OK = OK_BitField;

    // Figure out the type of the member; see C99 6.5.2.3p3, C++ [expr.ref]
  QualType MemberType = Field->getType();
  if (const ReferenceType *Ref = MemberType->getAs<ReferenceType>()) {
    MemberType = Ref->getPointeeType();
    VK = VK_LValue;
  } else {
    QualType BaseType = BaseExpr->getType();
    if (IsArrow) BaseType = BaseType->getAs<PointerType>()->getPointeeType();
    Qualifiers BaseQuals = BaseType.getQualifiers();

    // CVR attributes from the base are picked up by members,
    // except that 'mutable' members don't pick up 'const'.
    if (Field->isMutable()) {
      BaseQuals.removeConst();
    }

    Qualifiers MemberQuals
    = S.Context.getCanonicalType(MemberType).getQualifiers();

    assert(!MemberQuals.hasAddressSpace());


    Qualifiers Combined = BaseQuals + MemberQuals;
    if (Combined != MemberQuals)
      MemberType = S.Context.getQualifiedType(MemberType, Combined);
  }

  S.UnusedPrivateFields.remove(Field);

  ExprResult Base =
  S.PerformObjectMemberConversion(BaseExpr, SS.getScopeRep(),
                                  FoundDecl, Field);
  if (Base.isInvalid())
    return ExprError();

  return BuildMemberExpr(S, S.Context, Base.get(), IsArrow,
                         SourceLocation(), SS,
                         /*TemplateKWLoc=*/SourceLocation(),
                         Field, FoundDecl, MemberNameInfo,
                         MemberType, VK, OK);
}


bool Sema::LookupMeshMemberExpr(LookupResult &R, ExprResult &BaseExpr, SourceLocation OpLoc,
    CXXScopeSpec &SS, const MeshType *MTy, ExprResult &Result) {

  SourceLocation MemberLoc = R.getNameLoc();
  // Scout's mesh element access operations are all through
  // implicit constructs (for now) -- as such we treat all
  // explicit accesses as an error...

  Result = ExprError();

  Expr *E = BaseExpr.get();
  if(ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    E = ICE->getSubExpr();
  }

  DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
  if(!DRE) {
    Diag(MemberLoc, diag::err_illegal_mesh_element_access);
    return false;
  }

  // now, make this parseable to support parsing of plot() builtin
  // and other potential expressions
  
  /*
  if(!isa<ImplicitMeshParamDecl>(DRE->getDecl()) &&
      !isa<ParmVarDecl>(DRE->getDecl())) { // for stencil
    Diag(MemberLoc, diag::err_illegal_mesh_element_access);
    return false;
  }
  */

  if (LookupMemberExprInMesh(*this, R, BaseExpr.get()->getSourceRange(),
      MTy, OpLoc, SS))
    return false;

  // Returning valid-but-null is how we indicate to the caller that
  // the lookup result was filled in.
  Result = nullptr;
  return true;
}

bool Sema::LookupMemberExprInMesh(Sema &SemaRef, LookupResult &R,
                       SourceRange BaseRange, const MeshType *MTy,
                       SourceLocation OpLoc, CXXScopeSpec &SS){

  MeshDecl *MDecl = MTy->getDecl();
  DeclContext *DC = MDecl;

  SemaRef.LookupQualifiedName(R, DC);

  if (R.empty()) {
    SemaRef.Diag(OpLoc, diag::err_invalid_mesh_field) << R.getLookupName();
    return true;
  }

  NamedDecl* ND = R.getFoundDecl();

  assert(isa<MeshFieldDecl>(ND) && "expected a mesh field decl");

  MeshFieldDecl* FD = cast<MeshFieldDecl>(ND);

  if (FD->isCellLocated()) {
    if (!MTy->hasCellData()) {
      SemaRef.Diag(OpLoc, diag::err_invalid_mesh_cells_field) <<
      R.getLookupName();
      return true;
    }
  } else if (FD->isVertexLocated()) {
    if (!MTy->hasVertexData()) {
      SemaRef.Diag(OpLoc, diag::err_invalid_mesh_vertices_field) <<
      R.getLookupName();
      return true;
    }
  } else if (FD->isFaceLocated()) {
    if (!MTy->hasFaceData()) {
      SemaRef.Diag(OpLoc, diag::err_invalid_mesh_faces_field) <<
      R.getLookupName();
      return true;
    }
  } else if (FD->isEdgeLocated()) {
    if(!MTy->hasEdgeData()){
      SemaRef.Diag(OpLoc, diag::err_invalid_mesh_edges_field) <<
      R.getLookupName();
      return true;
    }
  } else {
    assert(false && "unknown mesh field type");
  }

  return false;
}

ExprResult
Sema::BuildMeshMemberReferenceExpr(Expr *Base, QualType BaseType,
                               SourceLocation OpLoc, bool IsArrow,
                               CXXScopeSpec &SS,
                               SourceLocation TemplateKWLoc,
                               NamedDecl *FirstQualifierInScope,
                               const DeclarationNameInfo &NameInfo,
                               const TemplateArgumentListInfo *TemplateArgs) {

  if (BaseType->isDependentType() ||
      (SS.isSet() && isDependentScopeSpecifier(SS)))
    return ActOnDependentMemberExpr(Base, BaseType,
                                    IsArrow, OpLoc,
                                    SS, TemplateKWLoc, FirstQualifierInScope,
                                    NameInfo, TemplateArgs);

  LookupResult R(*this, NameInfo, LookupMemberName);

  // Implicit member accesses.
  if (!Base) {
    QualType MeshTy = BaseType;
    if (IsArrow) MeshTy = MeshTy->getAs<PointerType>()->getPointeeType();
    if (LookupMemberExprInMesh(*this, R, SourceRange(),
                               MeshTy->getAs<MeshType>(),
                               OpLoc, SS))
      return ExprError();

  // Explicit member accesses.
  } else {
    ExprResult BaseResult = Base;
    ExprResult Result =
      LookupMemberExpr(*this, R, BaseResult, IsArrow, OpLoc,
                       SS, nullptr, TemplateArgs != nullptr);

    if (BaseResult.isInvalid())
      return ExprError();
    Base = BaseResult.get();

    if (Result.isInvalid()) {
      return ExprError();
    }

    if (Result.get())
      return Result;

    // LookupMemberExpr can modify Base, and thus change BaseType
    BaseType = Base->getType();
  }

  return BuildMeshMemberReferenceExpr(Base, BaseType,
                                  OpLoc, IsArrow, SS, TemplateKWLoc,
                                  FirstQualifierInScope, R, TemplateArgs);
}

ExprResult
Sema::BuildMeshMemberReferenceExpr(Expr *BaseExpr, QualType BaseExprType,
                                   SourceLocation OpLoc, bool IsArrow,
                                   const CXXScopeSpec &SS,
                                   SourceLocation TemplateKWLoc,
                                   NamedDecl *FirstQualifierInScope,
                                   LookupResult &R,
                                   const TemplateArgumentListInfo *TemplateArgs,
                                   bool SuppressQualifierCheck,
                                   ActOnMemberAccessExtraArgs *ExtraArgs) {

  QualType BaseType = BaseExprType;

  if (IsArrow) {
    assert(BaseType->isPointerType());
    BaseType = BaseType->castAs<PointerType>()->getPointeeType();
  }

  R.setBaseObjectType(BaseType);

  const DeclarationNameInfo &MemberNameInfo = R.getLookupNameInfo();
  DeclarationName MemberName = MemberNameInfo.getName();
  SourceLocation MemberLoc = MemberNameInfo.getLoc();

  if (R.isAmbiguous())
    return ExprError();

  if (R.empty()) {
    // Rederive where we looked up.
    DeclContext *DC = (SS.isSet()
                       ? computeDeclContext(SS, false)
                       : BaseType->getAs<RecordType>()->getDecl());

    if (ExtraArgs) {
      ExprResult RetryExpr;
      if (!IsArrow && BaseExpr) {
        SFINAETrap Trap(*this, true);
        ParsedType ObjectType;
        bool MayBePseudoDestructor = false;
        RetryExpr = ActOnStartCXXMemberReference(getCurScope(), BaseExpr,
                                                 OpLoc, tok::arrow, ObjectType,
                                                 MayBePseudoDestructor);
        if (RetryExpr.isUsable() && !Trap.hasErrorOccurred()) {
          CXXScopeSpec TempSS(SS);
          RetryExpr = ActOnMemberAccessExpr(
              ExtraArgs->S, RetryExpr.get(), OpLoc, tok::arrow, TempSS,
              TemplateKWLoc, ExtraArgs->Id, ExtraArgs->ObjCImpDecl);
        }
        if (Trap.hasErrorOccurred())
          RetryExpr = ExprError();
      }
      if (RetryExpr.isUsable()) {
        Diag(OpLoc, diag::err_no_member_overloaded_arrow)
          << MemberName << DC << FixItHint::CreateReplacement(OpLoc, "->");
        return RetryExpr;
      }
    }

    Diag(R.getNameLoc(), diag::err_no_member)
      << MemberName << DC
      << (BaseExpr ? BaseExpr->getSourceRange() : SourceRange());
    return ExprError();
  }

  // Diagnose lookups that find only declarations from a non-base
  // type.  This is possible for either qualified lookups (which may
  // have been qualified with an unrelated type) or implicit member
  // expressions (which were found with unqualified lookup and thus
  // may have come from an enclosing scope).  Note that it's okay for
  // lookup to find declarations from a non-base type as long as those
  // aren't the ones picked by overload resolution.
  if ((SS.isSet() || !BaseExpr ||
       (isa<CXXThisExpr>(BaseExpr) &&
        cast<CXXThisExpr>(BaseExpr)->isImplicit())) &&
      !SuppressQualifierCheck &&
      CheckQualifiedMemberReference(BaseExpr, BaseType, SS, R))
    return ExprError();

  // Construct an unresolved result if we in fact got an unresolved
  // result.
  if (R.isOverloadedResult() || R.isUnresolvableResult()) {
    // Suppress any lookup-related diagnostics; we'll do these when we
    // pick a member.
    R.suppressDiagnostics();
    UnresolvedMemberExpr *MemExpr
      = UnresolvedMemberExpr::Create(Context, R.isUnresolvableResult(),
                                     BaseExpr, BaseExprType,
                                     IsArrow, OpLoc,
                                     SS.getWithLocInContext(Context),
                                     TemplateKWLoc, MemberNameInfo,
                                     TemplateArgs, R.begin(), R.end());

    return MemExpr;
  }

  assert(R.isSingleResult());
  DeclAccessPair FoundDecl = R.begin().getPair();
  NamedDecl *MemberDecl = R.getFoundDecl();

  // FIXME: diagnose the presence of template arguments now.

  // If the decl being referenced had an error, return an error for this
  // sub-expr without emitting another error, in order to avoid cascading
  // error cases.
  if (MemberDecl->isInvalidDecl())
    return ExprError();

  // Handle the implicit-member-access case.
  if (!BaseExpr) {
    // If this is not an instance member, convert to a non-member access.
    if (!MemberDecl->isCXXInstanceMember())
      return BuildDeclarationNameExpr(SS, R.getLookupNameInfo(), MemberDecl);

    SourceLocation Loc = R.getNameLoc();
    if (SS.getRange().isValid())
      Loc = SS.getRange().getBegin();
    CheckCXXThisCapture(Loc);
    BaseExpr = new (Context) CXXThisExpr(Loc, BaseExprType,/*isImplicit=*/true);
  }

  bool ShouldCheckUse = true;
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MemberDecl)) {
    // Don't diagnose the use of a virtual member function unless it's
    // explicitly qualified.
    if (MD->isVirtual() && !SS.isSet())
      ShouldCheckUse = false;
  }

  // Check the use of this member.
  if (ShouldCheckUse && DiagnoseUseOfDecl(MemberDecl, MemberLoc))
    return ExprError();

  if (MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(MemberDecl))
    return BuildMeshFieldReferenceExpr(*this, BaseExpr, IsArrow,
                                       SS, MFD, FoundDecl, MemberNameInfo);

  if (FieldDecl *FD = dyn_cast<FieldDecl>(MemberDecl))
    return BuildFieldReferenceExpr(*this, BaseExpr, IsArrow,
                                   SourceLocation(),
                                   SS, FD, FoundDecl, MemberNameInfo);

  if (MSPropertyDecl *PD = dyn_cast<MSPropertyDecl>(MemberDecl))
    return BuildMSPropertyRefExpr(*this, BaseExpr, IsArrow, SS, PD,
                                  MemberNameInfo);

  if (IndirectFieldDecl *FD = dyn_cast<IndirectFieldDecl>(MemberDecl))
    // We may have found a field within an anonymous union or struct
    // (C++ [class.union]).
    return BuildAnonymousStructUnionMemberReference(SS, MemberLoc, FD,
                                                    FoundDecl, BaseExpr,
                                                    OpLoc);

  if (VarDecl *Var = dyn_cast<VarDecl>(MemberDecl)) {
    return BuildMemberExpr(*this, Context, BaseExpr, IsArrow,
                           SourceLocation(), SS,
                           TemplateKWLoc, Var, FoundDecl, MemberNameInfo,
                           Var->getType().getNonReferenceType(),
                           VK_LValue, OK_Ordinary);
  }

  if (CXXMethodDecl *MemberFn = dyn_cast<CXXMethodDecl>(MemberDecl)) {
    ExprValueKind valueKind;
    QualType type;
    if (MemberFn->isInstance()) {
      valueKind = VK_RValue;
      type = Context.BoundMemberTy;
    } else {
      valueKind = VK_LValue;
      type = MemberFn->getType();
    }

    return BuildMemberExpr(*this, Context, BaseExpr, IsArrow,
                           SourceLocation(), SS,
                           TemplateKWLoc, MemberFn, FoundDecl,
                           MemberNameInfo, type, valueKind,
                           OK_Ordinary);
  }
  assert(!isa<FunctionDecl>(MemberDecl) && "member function not C++ method?");

  if (EnumConstantDecl *Enum = dyn_cast<EnumConstantDecl>(MemberDecl)) {
    return BuildMemberExpr(*this, Context, BaseExpr, IsArrow,
                           SourceLocation(), SS,
                           TemplateKWLoc, Enum, FoundDecl, MemberNameInfo,
                           Enum->getType(), VK_RValue, OK_Ordinary);
  }

  // We found something that we didn't expect. Complain.
  if (isa<TypeDecl>(MemberDecl))
    Diag(MemberLoc, diag::err_typecheck_member_reference_type)
      << MemberName << BaseType << int(IsArrow);
  else {
    Diag(MemberLoc, diag::err_typecheck_member_reference_unknown)
      << MemberName << BaseType << int(IsArrow);
  }

  Diag(MemberDecl->getLocation(), diag::note_member_declared_here)
    << MemberName;
  R.suppressDiagnostics();
  return ExprError();
}

