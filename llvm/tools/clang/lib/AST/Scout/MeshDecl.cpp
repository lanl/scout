
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
#include "clang/AST/scout/MeshDecl.h"
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
MeshDecl::MeshDecl(Kind DK, TagKind TK, DeclContext* DC,
                   SourceLocation StartLoc, SourceLocation IdLoc,
                   IdentifierInfo* Id, MeshDecl* PrevDecl)
  : TagDecl(DK, TK, DC, IdLoc, Id, PrevDecl, StartLoc) {
HasVolatileMember                = false;
HasCellData                      = false;
HasVertexData                    = false;
HasFaceData                      = false;
HasEdgeData                      = false;
LoadedFieldsFromExternalStorage  = false;
assert(static_cast<Decl*>(this) && "Invalid Kind!");
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void MeshDecl::completeDefinition() {
  assert(!isCompleteDefinition() && "Cannot redefine record!");
  TagDecl::completeDefinition();
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
    assert(isa<FieldDecl>(Decls[i]) || isa<IndirectFieldDecl>(Decls[i]));
#endif

  if (Decls.empty())
    return;

  llvm::tie(FirstDecl, LastDecl) = BuildDeclChain(Decls,
                                                 /*FieldsAlreadyLoaded=*/false);
}

MeshDecl::field_iterator MeshDecl::field_begin() const {
  if (hasExternalLexicalStorage() && !LoadedFieldsFromExternalStorage)
    LoadFieldsFromExternalStorage();

  return field_iterator(decl_iterator(FirstDecl));
}


//===----------------------------------------------------------------------===//
// UniformMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
UniformMeshDecl::UniformMeshDecl(DeclContext     *DC,
                                 SourceLocation  StartLoc,
                                 SourceLocation  IdLoc,
                                 IdentifierInfo  *Id, 
                                 UniformMeshDecl *PrevDecl)
  : MeshDecl(UniformMesh, TTK_UniformMesh, DC, StartLoc,
             IdLoc, Id, PrevDecl) {

}

UniformMeshDecl *UniformMeshDecl::Create(const ASTContext &C, 
                                         DeclContext *DC,
                                         SourceLocation StartLoc, 
                                         SourceLocation IdLoc,
                                         IdentifierInfo *Id, 
                                         UniformMeshDecl* PrevDecl) {

  UniformMeshDecl* M = new (C) UniformMeshDecl(DC, 
                                               StartLoc, 
                                               IdLoc, Id,
                                               PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

UniformMeshDecl *UniformMeshDecl::CreateDeserialized(const ASTContext &C, 
                                                     unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UniformMeshDecl));
  UniformMeshDecl *M = new (Mem) UniformMeshDecl(0, SourceLocation(),
                                                 SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}



//===----------------------------------------------------------------------===//
// RectilinearMeshDecl Implementation
//===----------------------------------------------------------------------===//
//
//
RectilinearMeshDecl::RectilinearMeshDecl(DeclContext     *DC,
                                         SourceLocation  StartLoc,
                                         SourceLocation  IdLoc,
                                         IdentifierInfo  *Id, 
                                         RectilinearMeshDecl *PrevDecl)
  : MeshDecl(RectilinearMesh, TTK_RectilinearMesh, DC, StartLoc,
             IdLoc, Id, PrevDecl) {

}

RectilinearMeshDecl *RectilinearMeshDecl::Create(const ASTContext &C, 
                                                 DeclContext *DC,
                                                 SourceLocation StartLoc, 
                                                 SourceLocation IdLoc,
                                                 IdentifierInfo *Id, 
                                                 RectilinearMeshDecl* PrevDecl) {

  RectilinearMeshDecl* M = new (C) RectilinearMeshDecl(DC, 
                                                       StartLoc, 
                                                       IdLoc, Id,
                                                       PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

RectilinearMeshDecl *RectilinearMeshDecl::CreateDeserialized(const ASTContext &C, 
                                                             unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(RectilinearMeshDecl));
  RectilinearMeshDecl *M = new (Mem) RectilinearMeshDecl(0, SourceLocation(),
                                                         SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}

//===----------------------------------------------------------------------===//
// StructuredMeshDecl Implementation
//===----------------------------------------------------------------------===//
// 
//
StructuredMeshDecl::StructuredMeshDecl(DeclContext* DC,
                                       SourceLocation StartLoc,
                                       SourceLocation IdLoc,
                                       IdentifierInfo* Id, 
                                       StructuredMeshDecl* PrevDecl)
  : MeshDecl(StructuredMesh, TTK_StructuredMesh, DC, StartLoc,
             IdLoc, Id, PrevDecl) {

}

StructuredMeshDecl *StructuredMeshDecl::Create(const ASTContext &C, 
                                               DeclContext *DC,
                                               SourceLocation StartLoc, 
                                               SourceLocation IdLoc,
                                               IdentifierInfo *Id, 
                                               StructuredMeshDecl* PrevDecl) {

  StructuredMeshDecl* M = new (C) StructuredMeshDecl(DC, 
                                                     StartLoc, 
                                                     IdLoc, Id,
                                                     PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

StructuredMeshDecl *StructuredMeshDecl::CreateDeserialized(const ASTContext &C, 
                                                           unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(StructuredMeshDecl));
  StructuredMeshDecl *M = new (Mem) StructuredMeshDecl(0, SourceLocation(),
                                                       SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}


//===----------------------------------------------------------------------===//
// UnstructuredMeshDecl Implementation
//===----------------------------------------------------------------------===//
// 
//
UnstructuredMeshDecl::UnstructuredMeshDecl(DeclContext* DC,
                                           SourceLocation StartLoc,
                                           SourceLocation IdLoc,
                                           IdentifierInfo* Id, 
                                           UnstructuredMeshDecl* PrevDecl)
  : MeshDecl(UnstructuredMesh, TTK_UnstructuredMesh, DC, StartLoc,
             IdLoc, Id, PrevDecl) {

}

UnstructuredMeshDecl *UnstructuredMeshDecl::Create(const ASTContext &C, 
                                                   DeclContext *DC,
                                                   SourceLocation StartLoc, 
                                                   SourceLocation IdLoc,
                                                   IdentifierInfo *Id, 
                                                   UnstructuredMeshDecl* PrevDecl) {

  UnstructuredMeshDecl* M = new (C) UnstructuredMeshDecl(DC, 
                                                         StartLoc, 
                                                         IdLoc, Id,
                                                         PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}

UnstructuredMeshDecl *UnstructuredMeshDecl::CreateDeserialized(const ASTContext &C, 
                                                               unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UnstructuredMeshDecl));
  UnstructuredMeshDecl *M = new (Mem) UnstructuredMeshDecl(0, SourceLocation(),
                                                           SourceLocation(), 0, 0);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}

