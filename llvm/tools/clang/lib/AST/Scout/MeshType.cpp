#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace clang;

MeshType::MeshType(TypeClass TC, const MeshDecl *D, QualType can)
    : Type(TC, can, D->isDependentType(),
        /*InstantiationDependent=*/D->isDependentType(),
        /*VariablyModified=*/false,
        /*ContainsUnexpandedParameterPack=*/false),
      decl(const_cast<MeshDecl*>(D)) {}


static MeshDecl *getInterestingMeshDecl(MeshDecl *decl) {
  for (MeshDecl::redecl_iterator I = decl->redecls_begin(),
       E = decl->redecls_end();
       I != E; ++I) {
    if (I->isCompleteDefinition() || I->isBeingDefined())
      return *I;
  }
  // If there's no definition (not even in progress), return what we have.
  return decl;
}

MeshDecl *MeshType::getDecl() const {
  return getInterestingMeshDecl(decl);
}


bool MeshType::isBeingDefined() const {
  return getDecl()->isBeingDefined();
}

bool MeshType::hasCellData() const {
  return getDecl()->hasCellData();
}

bool MeshType::hasVertexData() const {
  return getDecl()->hasVertexData();
}

bool MeshType::hasEdgeData() const {
  return getDecl()->hasEdgeData();
}

bool MeshType::hasFaceData() const {
  return getDecl()->hasFaceData();
}


MeshFieldType::MeshFieldType(TypeClass TC, 
                             const MeshFieldDecl *D, 
                             QualType can)
  : Type(TC, can, D->isDependentType(), D->isDependentType(), false, false),
    Decl(const_cast<MeshFieldDecl*>(D)) 
{  }

bool MeshFieldType::isCellLocated() const {
  return getDecl()->isCellLocated();
}

bool MeshFieldType::isVertexLocated() const {
  return getDecl()->isCellLocated();
}

bool MeshFieldType::isEdgeLocated() const {
  return getDecl()->isEdgeLocated();
}

bool MeshFieldType::isFaceLocated() const {
  return getDecl()->isFaceLocated();
}

bool MeshFieldType::isBuiltInField() const {
  return getDecl()->isBuiltInField();
}

