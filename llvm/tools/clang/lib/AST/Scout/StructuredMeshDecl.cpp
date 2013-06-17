
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
#include "clang/AST/scout/StructuredMeshDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>

using namespace clang;

StructuredMeshDecl*
StructuredMeshDecl::Create(ASTContext& C, Kind K, DeclContext* DC,
                           SourceLocation StartLoc, SourceLocation IdLoc,
                           IdentifierInfo* Id, StructuredMeshDecl* PrevDecl){
  
  StructuredMeshDecl* M =
  new (C) StructuredMeshDecl(K, DC, StartLoc, IdLoc, Id, PrevDecl);
  
  RecordDecl* SR =
  RecordDecl::Create(C, TTK_Struct, DC, IdLoc,
                     IdLoc, &C.Idents.get(M->getName()));
  
  M->setStructRep(SR);
  
  C.getTypeDeclType(M);
  return M;
}

void StructuredMeshDecl::addImplicitFields(SourceLocation Loc, const ASTContext &Context) {}
