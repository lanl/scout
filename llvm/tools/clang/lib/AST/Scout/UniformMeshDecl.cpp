
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
#include "clang/AST/scout/UniformMeshDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>

using namespace clang;


UniformMeshDecl* UniformMeshDecl::Create(ASTContext& C, Kind K, DeclContext* DC,
                                         SourceLocation StartLoc, SourceLocation IdLoc,
                                         IdentifierInfo* Id, UniformMeshDecl* PrevDecl){
  
  UniformMeshDecl* M =
  new (C) UniformMeshDecl(K, DC, StartLoc, IdLoc, Id, PrevDecl);
  
  RecordDecl* SR =
  RecordDecl::Create(C, TTK_Struct, DC, IdLoc,
                     IdLoc, &C.Idents.get(M->getName()));
  
  M->setStructRep(SR);
  
  C.getTypeDeclType(M);
  return M;
}


UniformMeshDecl* UniformMeshDecl::CreateFromStructRep(ASTContext& C,
                                                      Kind DK,
                                                      DeclContext* DC,
                                                      IdentifierInfo* Id,
                                                      RecordDecl* SR){
  
  UniformMeshDecl* M =
  new (C) UniformMeshDecl(DK, DC, SR->getLocStart(),
                          SR->getLocStart(), SR->getIdentifier(), 0);
  
  M->setStructRep(SR);
  
  C.getTypeDeclType(M);
  
  for(RecordDecl::mesh_field_iterator itr = SR->mesh_field_begin(),
        itrEnd = SR->mesh_field_end(); itr != itrEnd; ++itr){
    
    MeshFieldDecl* field = *itr;
    
    MeshFieldDecl* newField =
    MeshFieldDecl::Create(C, M, field->getLocation(),
                          field->getLocation(),
                          &C.Idents.get(field->getName()),
                          field->getType().getTypePtr()->getPointeeType(),
                          0,
                          0,
                          true,
                          ICIS_NoInit,
                          MeshFieldDecl::CellLoc);
    newField->setImplicit(false);
    M->addDecl(newField);
  }

  M->addImplicitFields(SR->getLocStart(), C);

  return M;
}

void UniformMeshDecl::addImplicitFields(SourceLocation Loc, const ASTContext &Context) {

  MeshFieldDecl *PositionFD =
  MeshFieldDecl::Create(Context, this, Loc, Loc,
                        &Context.Idents.get("position"), Context.Int4Ty, 0,
                        0, true, ICIS_NoInit, MeshFieldDecl::BuiltIn);
  PositionFD->setImplicit(true);
  addDecl(PositionFD);

  MeshFieldDecl *WidthFD =
  MeshFieldDecl::Create(Context, this, Loc, Loc,
                        &Context.Idents.get("width"), Context.IntTy, 0,
                        0, true, ICIS_NoInit, MeshFieldDecl::BuiltIn);
  WidthFD->setImplicit(true);
  addDecl(WidthFD);

  MeshFieldDecl *HeightFD =
  MeshFieldDecl::Create(Context, this, Loc, Loc,
                        &Context.Idents.get("height"), Context.IntTy, 0,
                        0, true, ICIS_NoInit, MeshFieldDecl::BuiltIn);
  HeightFD->setImplicit(true);
  addDecl(HeightFD);

  MeshFieldDecl *DepthFD =
  MeshFieldDecl::Create(Context, this, Loc, Loc,
                        &Context.Idents.get("depth"), Context.IntTy, 0,
                        0, true, ICIS_NoInit, MeshFieldDecl::BuiltIn);
  DepthFD->setImplicit(true);
  addDecl(DepthFD);

  // SC_TODO - what the heck is 'ptr' again?  What do we use it for?
  MeshFieldDecl *PtrFD =
  MeshFieldDecl::Create(Context, this, Loc, Loc,
                        &Context.Idents.get("ptr"), Context.VoidPtrTy, 0,
                        0, true, ICIS_NoInit, MeshFieldDecl::BuiltIn);
  PtrFD->setImplicit(true);
  addDecl(PtrFD);

}

