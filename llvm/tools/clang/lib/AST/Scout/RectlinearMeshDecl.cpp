
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
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include "clang/AST/scout/RectilinearMeshDecl.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// RectilinearMeshDecl Implementation
//===----------------------------------------------------------------------===//
// 
//
RectilinearMeshDecl::RectilinearMeshDecl(DeclContext* DC,
                                         SourceLocation StartLoc,
                                         SourceLocation IdLoc,
                                         IdentifierInfo* Id, 
                                         RectilinearMeshDecl* PrevDecl)
  : MeshDecl(RectilinearMesh, TTK_RectilinearMesh, DC, StartLoc,
             IdLoc, Id, PrevDecl) {

}

RectilinearMeshDecl *RectilinearMeshDecl::Create(const ASTContext &C, 
                                                 DeclContext *DC,
                                                 SourceLocation StartLoc, 
                                                 SourceLocation IdLoc,
                                                 IdentifierInfo *Id, 
                                                 RectilinearMeshDecl* PrevDecl) {

  RectilinearMeshDecl* M = new (C) RectilinearMeshDecl(DC, StartLoc, 
                                                       IdLoc, Id,
                                                       PrevDecl);
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(M, PrevDecl);
  return M;
}
