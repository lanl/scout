/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
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
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 

#ifndef __SC_CLANG_UNIFORM_MESH_TYPE_H__
#define __SC_CLANG_UNIFORM_MESH_TYPE_H__

// NOTE: This file is expected to be included into the clang AST/Type.h
// file in order to work properly.  This makes our lives somewhat easier
// when we merge with the trunk...

/// UniformMeshType - This type is used to capture the details about
/// Scout's uniform mesh type.  In this case we've borrowed the basic
/// constructs behind C/C++ structs where the various data types
/// stored on the mesh are MemberExprs.  NOTE! Unlike C/C++ we should
/// not make any assumptions on the storage/ordering of data across
/// the various members stored per mesh location (nor across various
/// the various location types).  In other words, we would like to
/// experiment with choosing to layout data in contiguous linear
/// segments or as interleaved segments across a single location
/// (e.g. cells) or across multiple locations (e.g. interleaved cells
/// and face values).  So, we treat them as member expressions (for now)
/// only out of convenience...
///
///  Syntax:
///
///        uniform mesh UMeshTypeName {
///           [cells|vertex|edge|face]:  
///             member expr;
///             member expr;
///             ...
///           [cells|vertex|edge|face]:
///             member expr;
///             ...
///           ...
///        };
///
///  UMeshTypeName  my_mesh[xdim,ydim,zdim];  (can be 1, 2 or 3 dimensional)
///
class UniformMeshType : public Type {

 public:
  typedef llvm::SmallVector<Expr*, 3>  UniformMeshDimVec;

  UniformMeshType(const UniformMeshDecl *D)
  : Type(UniformMesh, QualType(), false, false, false, false),
    decl(const_cast<UniformMeshDecl*>(D)),
    instanceType(IT) {
    cellDataMember   = 0;
    vertexDataMember = 0;
    edgeDataMember   = 0;
    faceDataMember   = 0;
  }

  UniformMeshDecl* getDecl() const {
    return decl;
  }

  /// Return the dimensions of the mesh.  Note that
  /// only support meshes of rank 1, 2, or 3.  Other
  /// sizes will result in a compilation error. 
  const UniformMeshDimVec& dimensions() const {
    return dims;
  }

  /// Set the dimensions of the mesh.  Note that we
  /// only support meshes of 
  void setDimensions(const UniformMeshDimVec& dv) {
    dims = dv;
  }

  UniformMeshDimVec::size_type rankOf() const {
    return dims.size();
  }
  
  void isBeingDefined() const;

  bool isSugared() const {
    return false;
  }

  QualType desugar() const {
    return QualType(this, 0);
  }

  static bool classof(const UniformMeshType *T) {
    return true;
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == UniformMesh;
  }

  MemberExpr* getCellDataMember() {
    return cellDataMember;
  }

  bool hasCellData() const {
    return cellDataMember != 0;
  }

  MemberExpr* getVertexDataMember() {
    return vertexDataMember;
  }

  bool hasVertexData() const {
    return vertexDataMember != 0;
  }

  MemberExpr* getEdgeDataMember() {
    return edgeDataMember;
  }

  bool hasEdgeData() const {
    return edgeDataMember != 0;
  }  

  MemberExpr* getFaceDataMember() {
    return faceDataMember;
  };

  bool hasFaceData() const {
    return faceDataMember != 0;
  }    
  
 private:
  UniformMeshDecl    *decl;
  UniformMeshDimVec   dims;

  // The following member expressions contain the details
  // of the types stored within the various mesh (topology)
  // locations...
  MemberExpr         *cellDataMember;
  MemberExpr         *vertexDataMember;
  MemberExpr         *edgeDataMember;
  MemberExpr         *faceDataMember;
};

#endif
