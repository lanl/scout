//===----------------------------------------------------------------------===//
//
//  ndm - This file implements type-related functionality for Scout types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"
#include "clang/AST/DeclScout.h"

using namespace clang;

bool MeshType::isBeingDefined() const{
  return decl->isBeingDefined();
}

