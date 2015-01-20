
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/Scout/MeshDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>

using namespace clang;

static bool isFieldOrIndirectField(Decl::Kind K) {
  return FieldDecl::classofKind(K) || IndirectFieldDecl::classofKind(K);
}

//===----------------------------------------------------------------------===//
// MeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//

SourceLocation MeshDecl::getOuterLocStart() const {
  return getInnerLocStart();
}

SourceRange MeshDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(getOuterLocStart(), E);
}

MeshDecl* MeshDecl::getCanonicalDecl() {
  return getFirstDecl();
}

void MeshDecl::startDefinition() {
  IsBeingDefined = true;
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void MeshDecl::completeDefinition() {
  assert(!isCompleteDefinition() && "Cannot redefine mesh!");

  IsCompleteDefinition = true;
  IsBeingDefined = false;

  if (ASTMutationListener *L = getASTMutationListener())
    L->CompletedMeshDefinition(this);
}

MeshDecl *MeshDecl::getDefinition() const {
  if (isCompleteDefinition())
    return const_cast<MeshDecl *>(this);

  // If it's possible for us to have an out-of-date definition, check now.
  if (MayHaveOutOfDateDef) {
    if (IdentifierInfo *II = getIdentifier()) {
      if (II->isOutOfDate()) {
        updateOutOfDate(*II);
      }
    }
  }

  for (redecl_iterator R = redecls_begin(), REnd = redecls_end();
       R != REnd; ++R)
    if (R->isCompleteDefinition())
      return *R;

  return 0;
}

void MeshDecl::setQualifierInfo(NestedNameSpecifierLoc QualifierLoc) {
  if (QualifierLoc) {
    // Make sure the extended qualifier info is allocated.
    if (!hasExtInfo())
      TypedefNameDeclOrQualifier = new (getASTContext()) ExtInfo;
    // Set qualifier info.
    getExtInfo()->QualifierLoc = QualifierLoc;
  } else {
    // Here Qualifier == 0, i.e., we are removing the qualifier (if any).
    if (hasExtInfo()) {
      if (getExtInfo()->NumTemplParamLists == 0) {
        getASTContext().Deallocate(getExtInfo());
        TypedefNameDeclOrQualifier = (TypedefNameDecl*) 0;
      }
      else
        getExtInfo()->QualifierLoc = QualifierLoc;
    }
  }
}





void MeshDecl::LoadFieldsFromExternalStorage() const {
  ExternalASTSource *Source = getASTContext().getExternalSource();
  assert(hasExternalLexicalStorage() && Source && "No external storage?");

  // Notify that we have a RecordDecl doing some initialization.
  ExternalASTSource::Deserializing TheFields(Source);

  SmallVector<Decl*, 64> Decls;
  LoadedFieldsFromExternalStorage = true;
  switch (Source->FindExternalLexicalDecls(this, isFieldOrIndirectField,
                                           Decls)) {
  case ELR_Success:
    break;

  case ELR_AlreadyLoaded:
  case ELR_Failure:
    return;
  }

#ifndef NDEBUG
  // Check that all decls we got were FieldDecls.
  for (unsigned i=0, e=Decls.size(); i != e; ++i)
    assert(isa<MeshFieldDecl>(Decls[i]));
#endif

  if (Decls.empty())
    return;

  std::tie(FirstDecl, LastDecl) = BuildDeclChain(Decls,
                                                 /*FieldsAlreadyLoaded=*/false);
}

MeshDecl::field_iterator MeshDecl::field_begin() const {
  if (hasExternalLexicalStorage() && !LoadedFieldsFromExternalStorage)
    LoadFieldsFromExternalStorage();

  return field_iterator(decl_iterator(FirstDecl));
}


const char *MeshDecl::getKindName() const {

  if (isUniformMesh()) {
    return "uniform mesh";
  } else if (isRectilinearMesh()) {
    return "rectilinear mesh";
  } else if (isStructuredMesh()) {
    return "structured mesh";
  } else if (isUnstructuredMesh()) {
    return "unstructured mesh";
  } else {
    llvm_unreachable("unexpected/unknown mesh type!");
  }
}


//===----------------------------------------------------------------------===//
// UniformMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
UniformMeshDecl::UniformMeshDecl(const ASTContext &C,
                                 DeclContext     *DC,
                                 SourceLocation  StartLoc,
                                 SourceLocation  IdLoc,
                                 IdentifierInfo  *Id,
                                 UniformMeshDecl *PrevDecl)
  : MeshDecl(UniformMesh, TTK_UniformMesh, C, DC, IdLoc, Id, PrevDecl, StartLoc) { }

UniformMeshDecl *UniformMeshDecl::Create(const ASTContext &C,
                                         DeclContext *DC,
                                         SourceLocation StartLoc,
                                         SourceLocation IdLoc,
                                         IdentifierInfo *Id,
                                         UniformMeshDecl* PrevDecl) {

  UniformMeshDecl* M = new (C, DC) UniformMeshDecl(C, DC,
                                               StartLoc,
                                               IdLoc, Id,
                                               PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

UniformMeshDecl *UniformMeshDecl::CreateDeserialized(const ASTContext &C,
                                                     unsigned ID) {
  UniformMeshDecl *M = new (C, ID) UniformMeshDecl(C, 0, SourceLocation(),
                                                 SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}



//===----------------------------------------------------------------------===//
// RectilinearMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
RectilinearMeshDecl::RectilinearMeshDecl(const ASTContext &C,
                                         DeclContext     *DC,
                                         SourceLocation  StartLoc,
                                         SourceLocation  IdLoc,
                                         IdentifierInfo  *Id,
                                         RectilinearMeshDecl *PrevDecl)
  : MeshDecl(RectilinearMesh, TTK_RectilinearMesh, C, DC, IdLoc, Id, PrevDecl, StartLoc) { }

RectilinearMeshDecl *RectilinearMeshDecl::Create(const ASTContext &C,
                                                 DeclContext *DC,
                                                 SourceLocation StartLoc,
                                                 SourceLocation IdLoc,
                                                 IdentifierInfo *Id,
                                                 RectilinearMeshDecl* PrevDecl) {

  RectilinearMeshDecl* M = new (C, DC) RectilinearMeshDecl(C, DC,
                                                       StartLoc,
                                                       IdLoc, Id,
                                                       PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

RectilinearMeshDecl *RectilinearMeshDecl::CreateDeserialized(const ASTContext &C,
                                                             unsigned ID) {
  RectilinearMeshDecl *M = new (C, ID) RectilinearMeshDecl(C, 0, SourceLocation(),
                                                         SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}

//===----------------------------------------------------------------------===//
// StructuredMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
StructuredMeshDecl::StructuredMeshDecl(const ASTContext &C,
                                       DeclContext* DC,
                                       SourceLocation StartLoc,
                                       SourceLocation IdLoc,
                                       IdentifierInfo* Id,
                                       StructuredMeshDecl* PrevDecl)
  : MeshDecl(StructuredMesh, TTK_StructuredMesh, C, DC, IdLoc, Id, PrevDecl, StartLoc) { }

StructuredMeshDecl *StructuredMeshDecl::Create(const ASTContext &C,
                                               DeclContext *DC,
                                               SourceLocation StartLoc,
                                               SourceLocation IdLoc,
                                               IdentifierInfo *Id,
                                               StructuredMeshDecl* PrevDecl) {

  StructuredMeshDecl* M = new (C, DC) StructuredMeshDecl(C, DC,
                                                     StartLoc,
                                                     IdLoc, Id,
                                                     PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

StructuredMeshDecl *StructuredMeshDecl::CreateDeserialized(const ASTContext &C,
                                                           unsigned ID) {
  StructuredMeshDecl *M = new (C, ID) StructuredMeshDecl(C, 0, SourceLocation(),
                                                       SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}


//===----------------------------------------------------------------------===//
// UnstructuredMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
UnstructuredMeshDecl::UnstructuredMeshDecl(const ASTContext &C,
                                           DeclContext* DC,
                                           SourceLocation StartLoc,
                                           SourceLocation IdLoc,
                                           IdentifierInfo* Id,
                                           UnstructuredMeshDecl* PrevDecl)
  : MeshDecl(UnstructuredMesh, TTK_UnstructuredMesh, C, DC, IdLoc, Id, PrevDecl, StartLoc) { }


UnstructuredMeshDecl *UnstructuredMeshDecl::Create(const ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation StartLoc,
                                                   SourceLocation IdLoc,
                                                   IdentifierInfo *Id,
                                                   UnstructuredMeshDecl* PrevDecl) {

  UnstructuredMeshDecl* M = new (C, DC) UnstructuredMeshDecl(C, DC,
                                                         StartLoc,
                                                         IdLoc, Id,
                                                         PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

UnstructuredMeshDecl *UnstructuredMeshDecl::CreateDeserialized(const ASTContext &C,
                                                               unsigned ID) {
  UnstructuredMeshDecl *M = new (C, ID) UnstructuredMeshDecl(C, 0, SourceLocation(),
                                                           SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}

