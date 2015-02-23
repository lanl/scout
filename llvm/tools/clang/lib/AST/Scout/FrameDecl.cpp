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
#include "clang/AST/Scout/FrameDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>

using namespace clang;

FrameDecl *FrameDecl::Create(const ASTContext &C,
                             DeclContext *DC,
                             SourceLocation StartLoc,
                             SourceLocation IdLoc,
                             IdentifierInfo *Id,
                             FrameDecl* PrevDecl) {

  FrameDecl* F = new (C, DC) FrameDecl(C, DC,
                                       StartLoc,
                                       Id,
                                       PrevDecl, StartLoc);
  F->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  C.getTypeDeclType(F, PrevDecl);
  return F;
}

FrameDecl *FrameDecl::CreateDeserialized(const ASTContext &C,
                                         unsigned ID) {
  FrameDecl *M = new (C, ID) FrameDecl(C, 0, SourceLocation(), 0,
                                       0, SourceLocation());
  M->MayHaveOutOfDateDef = C.getLangOpts().Modules;
  return M;
}

SourceLocation FrameDecl::getOuterLocStart() const {
  return getInnerLocStart();
}

SourceRange FrameDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(getOuterLocStart(), E);
}

FrameDecl* FrameDecl::getCanonicalDecl() {
  return getFirstDecl();
}

void FrameDecl::startDefinition() {
  IsBeingDefined = true;
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void FrameDecl::completeDefinition() {
  assert(!isCompleteDefinition() && "Cannot redefine mesh!");

  IsCompleteDefinition = true;
  IsBeingDefined = false;

  if (ASTMutationListener *L = getASTMutationListener())
    L->CompletedFrameDefinition(this);
}

FrameDecl *FrameDecl::getDefinition() const {
  if (isCompleteDefinition())
    return const_cast<FrameDecl *>(this);

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

void FrameDecl::setQualifierInfo(NestedNameSpecifierLoc QualifierLoc) {
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
