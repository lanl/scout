#include "clang/AST/ASTContext.h"
#include "CXXABI.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Scout/MeshLayout.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Capacity.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace clang;

// ===== Mesh Declaration Types ===============================================

/// getMeshDeclType - Return the unique reference to the type for the 
/// specified mesh decl. 
QualType ASTContext::getMeshDeclType(const MeshDecl *Decl) const {
  assert(Decl);
  // FIXME: What is the design on getMeshDeclType when it requires 
  // casting away const?  mutable?
  return getTypeDeclType(const_cast<MeshDecl*>(Decl));
}

// SC_TODO -- not sure we need these specialized... 
QualType 
ASTContext::getUniformMeshDeclType(const UniformMeshDecl *Decl) const {
  assert (Decl != 0);
  return getTypeDeclType(const_cast<UniformMeshDecl*>(Decl));
}

QualType 
ASTContext::getStructuredMeshDeclType(const StructuredMeshDecl *Decl) const {
  assert (Decl != 0);
  return getTypeDeclType(const_cast<StructuredMeshDecl*>(Decl));
}

QualType 
ASTContext::getRectilinearMeshDeclType(const RectilinearMeshDecl *Decl) const {
  assert (Decl != 0);
  return getTypeDeclType(const_cast<RectilinearMeshDecl*>(Decl));
}

QualType 
ASTContext::getUnstructuredMeshDeclType(const UnstructuredMeshDecl *Decl) const {
  assert (Decl != 0);
  return getTypeDeclType(const_cast<UnstructuredMeshDecl*>(Decl));
}


// ===== Mesh Types ===========================================================

QualType 
ASTContext::getUniformMeshType(const UniformMeshDecl *Decl) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) 
    return QualType(Decl->TypeForDecl, 0);

  UniformMeshType *newType;
  newType = new (*this, TypeAlignment) UniformMeshType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType 
ASTContext::getStructuredMeshType(const StructuredMeshDecl *Decl) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) 
  	return QualType(Decl->TypeForDecl, 0);
  
  StructuredMeshType *newType;
  newType = new (*this, TypeAlignment) StructuredMeshType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType 
ASTContext::getRectilinearMeshType(const RectilinearMeshDecl *Decl) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) 
    return QualType(Decl->TypeForDecl, 0);
  
  RectilinearMeshType *newType;
  newType = new (*this, TypeAlignment) RectilinearMeshType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType 
ASTContext::getUnstructuredMeshType(const UnstructuredMeshDecl *Decl) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) 
  	return QualType(Decl->TypeForDecl, 0);
  
  UnstructuredMeshType *newType;
newType = new (*this, TypeAlignment) UnstructuredMeshType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}
