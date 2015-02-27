/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2013. Los Alamos National Security, LLC. This software was
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
 *
 */

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


// make sure that mesh types and specifier match
bool ASTContext::CompareMeshTypes(const Type *T1, const Type *T2) {

  if (T1->isUniformMeshType() && T2->isUniformMeshType()
      && T1->getAsUniformMeshDecl()->getName() ==
          T2->getAsUniformMeshDecl()->getName()) {
    return true;
  }

  if (T1->isRectilinearMeshType() && T2->isRectilinearMeshType()
      && T1->getAsRectilinearMeshDecl()->getName() ==
          T2->getAsRectilinearMeshDecl()->getName()) {
    return true;
  }

  if (T1->isStructuredMeshType() && T2->isStructuredMeshType()
      && T1->getAsStructuredMeshDecl()->getName() ==
          T2->getAsStructuredMeshDecl()->getName()) {
    return true;
  }

  if (T1->isUnstructuredMeshType() && T2->isUnstructuredMeshType()
      && T1->getAsUnstructuredMeshDecl()->getName() ==
          T2->getAsUnstructuredMeshDecl()->getName()) {
    return true;
  }

  return false;
}

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

QualType
ASTContext::getFrameDeclType(const FrameDecl *Decl) const {
  assert (Decl != 0);
  return getFrameType(const_cast<FrameDecl*>(Decl));
}

// ===== Mesh Types ===========================================================

QualType ASTContext::getUniformMeshType(const UniformMeshDecl *Decl) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) 
    return QualType(Decl->TypeForDecl, 0);

  UniformMeshType *newType;
  newType = new (*this, TypeAlignment) UniformMeshType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType ASTContext::getUniformMeshType(const UniformMeshDecl *Decl,
                        const MeshType::MeshDimensions &dims) const {
  assert(Decl != 0);

  if (Decl->TypeForDecl) {
    // There is a previous type stored -- however, we're not sure
    // that it represents an exact type match until we check the
    // dimensions of the mesh (note we are checking for duplicates
    // here but don't go all the way down to checking equivalence
    // of the expressions).
    //
    // SC_TODO - is it worth the effort of checking expressions for
    // equivalence here?
    const UniformMeshType *UMT = dyn_cast<UniformMeshType>(Decl->TypeForDecl);
    if (UMT->dimensions() == dims) {
      return QualType(Decl->TypeForDecl, 0);
    }
  }

  UniformMeshType *newType;
  newType = new (*this, TypeAlignment) UniformMeshType(Decl);
  newType->setDimensions(dims);    
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

MeshFieldDecl *ASTContext::getInstantiatedFromUnnamedFieldDecl(MeshFieldDecl *Field) {
  llvm::DenseMap<MeshFieldDecl *, MeshFieldDecl *>::iterator Pos;
  Pos = InstantiatedFromUnnamedMeshFieldDecl.find(Field);
  if (Pos == InstantiatedFromUnnamedMeshFieldDecl.end())
    return 0;

  return Pos->second;
}

void ASTContext::setInstantiatedFromUnnamedFieldDecl(MeshFieldDecl *Inst,
                                                     MeshFieldDecl *Tmpl) {
  assert(!Inst->getDeclName() && "Instantiated mesh field decl is not unnamed");
  assert(!Tmpl->getDeclName() && "Template mesh field decl is not unnamed");
  assert(!InstantiatedFromUnnamedMeshFieldDecl[Inst] &&
         "Already noted what unnamed field was instantiated from");

  InstantiatedFromUnnamedMeshFieldDecl[Inst] = Tmpl;
}


// ===== Render Target Types ====================================================

QualType ASTContext::getWindowType(const llvm::SmallVector<Expr*,2> &dims) const {
  assert(dims[0] != 0 && dims[1] != 0);  
  WindowType *newType;
  newType = new (*this, TypeAlignment) WindowType(dims[0], dims[1]);
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType ASTContext::getImageType(const llvm::SmallVector<Expr*,2> &dims) const {
  assert(dims[0] != 0 && dims[1] != 0);
  ImageType *newType;
  newType = new (*this, TypeAlignment) ImageType(dims[0], dims[1]);
  Types.push_back(newType);
  return QualType(newType, 0);
}

// ===== Query Type ====================================================
QualType ASTContext::getQueryType() const {
  QueryType *newType;
  newType = new (*this, TypeAlignment) QueryType();
  Types.push_back(newType);
  return QualType(newType, 0);
}

// ===== Frame Type ====================================================
QualType ASTContext::getFrameType(const FrameDecl *Decl) const {
  assert(Decl != 0);
  
  if (Decl->TypeForDecl)
    return QualType(Decl->TypeForDecl, 0);
  
  FrameType *newType;
  newType = new (*this, TypeAlignment) FrameType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

// ===== FrameVar Type ====================================================
QualType ASTContext::getFrameVarType(const Type *ElementType) const {
  FrameVarType *newType;
  newType = new (*this, TypeAlignment) FrameVarType(ElementType);
  Types.push_back(newType);
  return QualType(newType, 0);
}

