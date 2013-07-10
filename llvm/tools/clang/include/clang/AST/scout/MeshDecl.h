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
 * Notes: See the various mesh types in AST/Types.h for some 
 * -----  more details on Scout's mesh types.  It is important 
 *        to keep a connection between the various Decls in this
 *        file and those types. 
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

  class MeshDecl;

  // MeshFieldDecl - An instance of this class if create by
  // Sema::ActOnMeshField to represent a member of a Scout 
  // mesh.  We build directly on the features of FieldDecl's.
  //
  // NOTE:  Can't figure a way to extract MeshFieldDecl from 
  // this file, since it depends on FieldDecl, but RecordDecl 
  // depends on MeshFieldDecl being defined, so you'd get a 
  // circular dependency between Decl.h and MeshFieldDecl.h
  // if you did that. 
  class MeshFieldDecl : public FieldDecl {
    
    // FIXME: This can be packed into the bitfields in Decl.
    bool Mutable       : 1;

    // Each field can be placed at various locations within the 
    // topology of the mesh.  We use the following bitfields to 
    // track this location within the field's decl. 
    bool CellLocated   : 1;
    bool VertexLocated : 1;
    bool EdgeLocated   : 1;
    bool FaceLocated   : 1;
    bool BuiltInField  : 1;

    // The field index cache value below is borrowed from 
    // FieldDecl -- we could have made some changes to 
    // FieldDecl so we could inherit this but we've 
    // opted to have a smaller impact on the base Clang 
    // source instead.  Just be aware of this if you decide
    // to downcast a MeshFieldDecl to a FieldDecl.  
    mutable unsigned CachedFieldIndex : 26;

  protected:
  
    MeshFieldDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
                  SourceLocation IdLoc, IdentifierInfo *Id,
                  QualType T, TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
                  InClassInitStyle InitStyle)
        : FieldDecl(DK, DC, StartLoc, IdLoc, Id, T, TInfo,
                    BW, Mutable, InitStyle), CachedFieldIndex(0)
    { }

  public:

    static MeshFieldDecl *Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation StartLoc, SourceLocation IdLoc,
                                 IdentifierInfo *Id, QualType T,
                                 TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
                                 InClassInitStyle InitStyle);
  
    static MeshFieldDecl *CreateDeserialized(ASTContext &C, unsigned ID);  
  
    // Return the mesh that contains this field. 
    const MeshDecl *getParentMesh() const {
      return cast<MeshDecl>(getDeclContext());
    }

    // Return the index position of this field within the parent mesh. 
    unsigned getMeshFieldIndex() const;

    // \brief Determine if the field is located at the cells of the mesh.
    bool isCellLocated() const {
      return CellLocated;
    }

    // \brief Set the field to be located at the cells of the mesh. 
    void setCellLocated(bool flag = true) {
      CellLocated = flag;
    }

    // \brief Determine if the field is located at the vertices of the mesh.
    bool isVertexLocated() const {
      return VertexLocated;
    }
    // \brief Set the field to be located at the vertices of the mesh. 
    void setVertexLocated(bool flag = true) {
      VertexLocated = flag;
    }

    // \brief Determine if the field is located at the edges of the mesh.
    bool isEdgeLocated() const {
      return EdgeLocated;
    }
    // \brief Set the field to be located at the edges of the mesh. 
    void setEdgeLocated(bool flag = true) {
      EdgeLocated = flag;
    }

    // \brief Determine if the field is located at the faces of the mesh.
    bool isFaceLocated() const {
      return FaceLocated;
    }
    // \brief Set the field to be located at the faces of the mesh.
    void setFaceLocated(bool flag = true) {
      FaceLocated = flag;
    }

    // \brief Determine if the field represents a built-in value. 
    bool isBuiltInField() const {
      return BuiltInField;
    }
    // \brief Set the field to represent a built-in value. 
    void setBuiltInField(bool flag = true) {
      BuiltInField = flag;
    }

    bool isValidLocation() const {
      return (isCellLocated()    || 
              isVertexLocated()  ||
              isEdgeLocated()    || 
              isFaceLocated()    ||
              isBuiltInField());
    }

    // FIXME - This implementation will currently keep us from 
    // being able to support templated fields. 
    bool isDependentType() const { return false; }

    static bool classof(const Decl *D) { return classofKind(D->getKind()); }
    static bool classofKind(Kind K) { return K == MeshField; }

    friend class ASTDeclReader;
    friend class ASTDeclWriter;
  };


  // Mesh - This is a base class for all of our mesh types; thus it 
  // provides some basic functionality for all mesh decls.  Given the
  // similarity of our mesh types to C/C++ structs we follow a similar
  // path and implement the mesh decls on top of tag decls (this is a 
  // change from our first implementations).  
  //    
  // Note: Make sure you follow similar inheritance paths with both the
  // Decls and Types -- i.e. MeshType should inherit from TagType.  If 
  // you don't do this it is likely you'll see some fairly opaque error
  // messages stemming from the generated '.inc' files...
  // 
  // SC_TODO - from looking at the new LLVM IR we're generating this looks
  // to have horrible alignment details.  We will likely have to implement
  // our own alignment details like RecordDecl... 
  class MeshDecl : public TagDecl {
  
   private:
    // FIXME: This can be packed into the bitfields in Decl.
    /// HasCellData - This is true if the mesh has at least one member 
    /// stored at the cells of the mesh. 
    bool HasCellData   : 1;
    /// HasVertexData - This is true if the mesh has at least one member
    /// stored at the vertices of the mesh.
    bool HasVertexData : 1;
    /// HasFaceData - This is true if the mesh has at least one member 
    /// stored at the faces of the mesh. 
    bool HasFaceData   : 1;
    /// HasEdgeData - This is true if the mesh has at least one member 
    /// stored at the edges of the mesh. 
    bool HasEdgeData   : 1;

    /// HasVolatileMember - This is true if the mesh has at least one 
    /// member of 'volatile' type.
    bool HasVolatileMember : 1;

    friend class DeclContext;

   protected:
    MeshDecl(Kind DK, TagKind TK, DeclContext* DC,
             SourceLocation L, SourceLocation StartL,
             IdentifierInfo* Id, MeshDecl* PrevDecl);

    mutable bool LoadedFieldsFromExternalStorage : 1;

   public:
    bool hasVolatileMember() const { return HasVolatileMember; }
    void setHasVolatileMember (bool val) { HasVolatileMember = val; }

    // Return true if the mesh has one or more fields stored at the cells.
    bool hasCellData() const { return HasCellData; }
    // Flag the mesh as having on or more fields stored at the cells. 
    void setHasCellData(bool flag) { HasCellData = flag; }

    // Return true if the mesh has one or more fields stored at the vertices.
    bool hasVertexData() const { return HasVertexData; }
    // Flag the mesh as having on or more fields stored at the vertices.     
    void setHasVertexData(bool flag) { HasVertexData = flag; }
    
    // Return true if the mesh has one or more fields stored at the edges.
    bool hasEdgeData() const { return HasEdgeData; }
    // Flag the mesh as having on or more fields stored at the edges.     
    void setHasEdgeData(bool flag) { HasEdgeData = flag; }    

    // Return true if the mesh has one or more fields stored at the faces.
    bool hasFaceData() const { return HasFaceData; }
    // Flag the mesh as having on or more fields stored at the faces.     
    void setHasFaceData(bool flag) { HasFaceData = flag; }

    MeshDecl *getDefinition() const {
      return cast_or_null<MeshDecl>(TagDecl::getDefinition());
    }

    // Iterator access to mesh field members. The field iterator only
    // visits the non-static data members of this class, ignoring any
    // static data members.
    typedef specific_decl_iterator<MeshFieldDecl> field_iterator;

    field_iterator field_begin() const;

    field_iterator field_end() const {
      return field_iterator(decl_iterator());
    }

    // field_empty - Whether there are any fields (non-static data
    // members) in this record.
    bool field_empty() const {
      return field_begin() == field_end();
    }

    /// completeDefinition - Notes that the definition of this type is
    /// now complete.
    virtual void completeDefinition();

    static bool classof(const Decl *D) { return classofKind(D->getKind()); }
    static bool classofKind(Kind K) {
      return K >= firstMesh && K <= lastMesh;
    }

   protected:
    /// \brief Deserialize just the fields.
    virtual void LoadFieldsFromExternalStorage() const;
  };


} // end namespace clang

#endif
