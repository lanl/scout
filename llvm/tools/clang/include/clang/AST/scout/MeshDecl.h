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

#ifndef __SC_CLANG_MESH_DECL_H__
#define __SC_CLANG_MESH_DECL_H__


#include "clang/AST/APValue.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Linkage.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

// ===== Scout =========================================================================================
// Mesh
// A mesh declaration is similar to a TagDecl/RecordDecl but different
// enough that a new subclass of TypeDecl was created. It encapsulates
// a mesh definition such as:
//    
//   uniform mesh MyMesh{
//     cells:
//          float a;
//   }
//
// SC_TODO - from looking at the new LLVM IR we're generating this looks
// to have horrible alignment details -- need to check on this… 
class MeshDecl : public TypeDecl, public DeclContext{
  
 public:
  typedef std::vector<const MeshFieldDecl*> MeshFieldVec;

 private:
  bool IsDefinition : 1;
  bool IsBeingDefined : 1;
  SourceLocation RBraceLoc;
  RecordDecl* StructRep;
  MeshFieldVec CellFields;
  MeshFieldVec VertexFields;
  MeshFieldVec FaceFields;
  MeshFieldVec EdgeFields;

 protected:
  
  MeshDecl(Kind DK, DeclContext* DC,
           SourceLocation L, SourceLocation StartL,
           IdentifierInfo* Id, MeshDecl* PrevDecl)
  : TypeDecl(DK, DC, L, Id, StartL),
    DeclContext(DK),
    StructRep(0){
    IsDefinition = false;
    IsBeingDefined = false;
  }  
  
 public:
  typedef const MeshFieldDecl* const* const_mesh_field_iterator;
  
  void completeDefinition(ASTContext& C);
  
  RecordDecl* getStructRep() {
    return StructRep;
  }
  
  void setStructRep(RecordDecl* SR) {
    StructRep = SR;
  }
  
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }
  
  SourceLocation getInnerLocStart() const { return getLocStart(); }
  
  SourceLocation getOuterLocStart() const { return getLocStart(); }
  
  virtual SourceRange getSourceRange() const;
  
  bool isThisDeclarationADefinition() const {
    return isDefinition();
  }
  
  bool isDefinition() const {
    return IsDefinition;
  }
  
  bool isBeingDefined() const {
    return IsBeingDefined;
  }
  
  void startDefinition();
  
  MeshDecl* getDefinition() const;
  
  typedef specific_decl_iterator<MeshFieldDecl> mesh_field_iterator;
  
  mesh_field_iterator mesh_field_begin() const;
  
  mesh_field_iterator mesh_field_end() const{
    return mesh_field_iterator(decl_iterator());
  }
  
  bool mesh_field_empty() const{
    return mesh_field_begin() == mesh_field_end();
  }
  
  NestedNameSpecifierLoc getQualifierLoc() const {
    return NestedNameSpecifierLoc();
  }
  
  static bool classof(const Decl* D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K >= firstMesh && K <= lastMesh; }
  
  static DeclContext* castToDeclContext(const MeshDecl* D){
    return static_cast<DeclContext*>(const_cast<MeshDecl*>(D));
  }

  static MeshDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<MeshDecl *>(const_cast<DeclContext*>(DC));
  }
  
  bool canConvertTo(ASTContext& C, MeshDecl* MD);
  
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
  
  void addCellField(const MeshFieldDecl *field) {
    assert(field->meshLocation() == MeshFieldDecl::CellLoc &&
           "expected a cell field");
    CellFields.push_back(field);
  }
  
  const_mesh_field_iterator cell_begin() const {
    return CellFields.data();
  }
  
  const_mesh_field_iterator cell_end() const {
    return CellFields.data() + CellFields.size();
  }
  
  bool cell_empty() const {
    return CellFields.empty();
  }

  void addVertexField(const MeshFieldDecl *field) {
    assert(field->meshLocation() == MeshFieldDecl::VertexLoc &&
           "expected a vertex field");
    VertexFields.push_back(field);
  }

  const_mesh_field_iterator vertex_begin() const {
    return VertexFields.data();
  }
  
  const_mesh_field_iterator vertex_end() const {
    return VertexFields.data() + VertexFields.size();
  }
  
  bool vertex_empty() const {
    return VertexFields.empty();
  }
  
  void addFaceField(const MeshFieldDecl *field) {
    assert(field->meshLocation() == MeshFieldDecl::FaceLoc &&
           "expected a face field");
    FaceFields.push_back(field);
  }

  const_mesh_field_iterator face_begin() const {
    return FaceFields.data();
  }
  
  const_mesh_field_iterator face_end() const {
    return FaceFields.data() + FaceFields.size();
  }
  
  bool face_empty() const {
    return FaceFields.empty();
  }
  
  void addEdgeField(const MeshFieldDecl *field) {
    assert(field->meshLocation() == MeshFieldDecl::EdgeLoc &&
           "expected an edge field");
    EdgeFields.push_back(field);
  }
  
  const_mesh_field_iterator edge_begin() const {
    return EdgeFields.data();
  }
  
  const_mesh_field_iterator edge_end() const {
    return EdgeFields.data() + EdgeFields.size();
  }
  
  bool edge_empty() const {
    return EdgeFields.empty();
  }

  virtual void addImplicitFields(SourceLocation Loc, const ASTContext &C){}  

};

} // end namespace clang

#endif
