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


  // ----- MeshFieldDecl
  /// An instance of this class is created by Sema::ActOnField to represent
  // a single member of a scout mesh.  Similar to records in C/C++ but
  // we use our own type for flexibility from a DSL point of view...
  class MeshFieldDecl : public DeclaratorDecl, public Mergeable<MeshFieldDecl> {

    // FIXME: This can be packed into the bitfields in Decl.
    bool Mutable : 1;

    // Each field can be placed at various locations within the
    // topology of the mesh.  We use the following bitfields to
    // track this location within the field's decl.
    bool CellLocated   : 1;
    bool VertexLocated : 1;
    bool EdgeLocated   : 1;
    bool FaceLocated   : 1;
    bool BuiltInField  : 1;

    mutable unsigned CachedFieldIndex : 26;

    /// \brief An InClassInitStyle value, and either a bit width expression (if
    /// the InClassInitStyle value is ICIS_NoInit), or a pointer to the in-class
    /// initializer for this field (otherwise).
    ///
    /// We can safely combine these two because in-class initializers are not
    /// permitted for bit-fields.
    ///
    /// If the InClassInitStyle is not ICIS_NoInit and the initializer is null,
    /// then this field has an in-class initializer which has not yet been parsed
    /// and attached.
    llvm::PointerIntPair<Expr *, 2, unsigned> InitializerOrBitWidth;
  protected:
    MeshFieldDecl(Kind DK, DeclContext *DC, SourceLocation StartLoc,
                  SourceLocation IdLoc, IdentifierInfo *Id,
                  QualType T, TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
                  InClassInitStyle InitStyle)
      : DeclaratorDecl(DK, DC, IdLoc, Id, T, TInfo, StartLoc),
        Mutable(Mutable), CachedFieldIndex(0),
        InitializerOrBitWidth(BW, InitStyle) {
      assert((!BW || InitStyle == ICIS_NoInit) && "got initializer for bitfield");

      CellLocated     = false;
      VertexLocated   = false;
      EdgeLocated     = false;
      FaceLocated     = false;
      BuiltInField    = false;
    }

  public:
    static MeshFieldDecl *Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation StartLoc, SourceLocation IdLoc,
                                 IdentifierInfo *Id, QualType T,
                                 TypeSourceInfo *TInfo, Expr *BW, bool Mutable,
                                 InClassInitStyle InitStyle);

    static MeshFieldDecl *CreateDeserialized(ASTContext &C, unsigned ID);

    /// getFieldIndex - Returns the index of this field within its record,
    /// as appropriate for passing to ASTRecordLayout::getFieldOffset.
    unsigned getFieldIndex() const;

    /// isMutable - Determines whether this field is mutable (C++ only).
    bool isMutable() const { return Mutable; }

    /// isBitfield - Determines whether this field is a bitfield.
    bool isBitField() const {
      return getInClassInitStyle() == ICIS_NoInit &&
             InitializerOrBitWidth.getPointer();
    }

    /// @brief Determines whether this is an unnamed bitfield.
    bool isUnnamedBitfield() const { return isBitField() && !getDeclName(); }

    Expr *getBitWidth() const {
      return isBitField() ? InitializerOrBitWidth.getPointer() : 0;
    }
    unsigned getBitWidthValue(const ASTContext &Ctx) const;

    /// setBitWidth - Set the bit-field width for this member.
    // Note: used by some clients (i.e., do not remove it).
    void setBitWidth(Expr *Width);
    /// removeBitWidth - Remove the bit-field width from this member.
    // Note: used by some clients (i.e., do not remove it).
    void removeBitWidth() {
      assert(isBitField() && "no bitfield width to remove");
      InitializerOrBitWidth.setPointer(0);
    }

    /// getInClassInitStyle - Get the kind of (C++11) in-class initializer which
    /// this field has.
    InClassInitStyle getInClassInitStyle() const {
      return static_cast<InClassInitStyle>(InitializerOrBitWidth.getInt());
    }

    /// hasInClassInitializer - Determine whether this member has a C++11 in-class
    /// initializer.
    bool hasInClassInitializer() const {
      return getInClassInitStyle() != ICIS_NoInit;
    }

    /// getInClassInitializer - Get the C++11 in-class initializer for this
    /// member, or null if one has not been set. If a valid declaration has an
    /// in-class initializer, but this returns null, then we have not parsed and
    /// attached it yet.
    Expr *getInClassInitializer() const {
      return hasInClassInitializer() ? InitializerOrBitWidth.getPointer() : 0;
    }

    /// setInClassInitializer - Set the C++11 in-class initializer for this
    /// member.
    void setInClassInitializer(Expr *Init);

    /// removeInClassInitializer - Remove the C++11 in-class initializer from this
    /// member.
    void removeInClassInitializer() {
      assert(hasInClassInitializer() && "no initializer to remove");
      InitializerOrBitWidth.setPointer(0);
      InitializerOrBitWidth.setInt(ICIS_NoInit);
    }

    /// getParent - Returns the parent of this field declaration, which
    /// is the struct in which this method is defined.
    const MeshDecl *getParent() const {
      return cast<MeshDecl>(getDeclContext());
    }

    MeshDecl *getParent() {
      return cast<MeshDecl>(getDeclContext());
    }

    SourceRange getSourceRange() const LLVM_READONLY;

    // Implement isa/cast/dyncast/etc.
    static bool classof(const Decl *D) { return classofKind(D->getKind()); }
    static bool classofKind(Kind K) { return K == MeshField; }

    friend class ASTDeclReader;
    friend class ASTDeclWriter;

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

    bool isBuiltInField() const {
      return BuiltInField;
    }

    void setBuiltInField(bool flag = true) {
      BuiltInField = flag;
    }

    bool isValidLocation() const {
      return CellLocated   ||
             VertexLocated ||
             EdgeLocated   ||
             FaceLocated   ||
             BuiltInField;
    }
  };

  // ----- MeshDecl
  // This is the base class for all of our mesh decls.  Given the syntax
  // similarities we have with structs in C/C++ we follow a similar
  // path as that used in TagDecls and RecordDecls -- however, we do not
  // inherit from this "family" as we want to enforce separation/uniqueness
  // in our mesh types (the details relate to the nuances of tree walkers
  // layout, debugging and flexible code generation choices at points down
  // the road -- and also match our push for domain-specific customizations
  // that differ from the "generic" handling in C/C++).
  //
  // Based on past experience, it is important (for our sanity) to follow a
  // similar path within types, decls, typelocs, etc. within the scout
  // implementation -- primarily we have found it cleans up the code and
  // appears to support a more flexible path...
  class MeshDecl
    : public TypeDecl, public DeclContext, public Redeclarable<MeshDecl> {

   public:
    typedef MeshTypeKind MeshKind;       // This is really ugly.

   private:
    // FIXME: This can be packed into the bitfields in Decl.
    /// MeshDeclKind - The MeshKind enum.
    unsigned MeshDeclKind : 3;

    /// IsCompleteDefinition - True if this is a definition ("uniform mesh foo
    /// {};"), false if it is a declaration ("uniform mesh foo;").  It is not
    /// a definition until the definition has been fully processed.
    bool IsCompleteDefinition : 1;

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

  protected:
    /// IsBeingDefined - True if this is currently being defined.
    bool IsBeingDefined : 1;

  private:
    /// IsEmbeddedInDeclarator - True if this mesh declaration is
    /// "embedded" (i.e., defined or declared for the very first time)
    /// in the syntax of a declarator.
    bool IsEmbeddedInDeclarator : 1;

    /// \brief True if this mesh is free standing, e.g. "uniform mesh foo;".
    bool IsFreeStanding : 1;

    /// HasVolatileMember - This is true if the mesh has at least one
    /// member of 'volatile' type.
    bool HasVolatileMember : 1;

  protected:
    /// \brief Indicates whether it is possible for declarations of this kind
    /// to have an out-of-date definition.
    ///
    /// This option is only enabled when modules are enabled.
    bool MayHaveOutOfDateDef : 1;

    /// Has the full definition of this type been required by a use somewhere in
    /// the TU.
    bool IsCompleteDefinitionRequired : 1;

    mutable bool LoadedFieldsFromExternalStorage : 1;

  private:
    SourceLocation RBraceLoc;

    // A mesh representing syntactic qualifier info,
    // to be used for the (uncommon) case of out-of-line declarations.
    typedef QualifierInfo ExtInfo;

    /// TypedefNameDeclOrQualifier - If the (out-of-line) mesh declaration name
    /// is qualified, it points to the qualifier info (nns and range);
    /// otherwise, if the mesh declaration is anonymous and it is part of
    /// a typedef or alias, it points to the TypedefNameDecl (used for mangling);
    /// otherwise, it is a null (TypedefNameDecl) pointer.
    llvm::PointerUnion<TypedefNameDecl*, ExtInfo*> TypedefNameDeclOrQualifier;

    bool hasExtInfo() const { return TypedefNameDeclOrQualifier.is<ExtInfo*>(); }
    ExtInfo *getExtInfo() { return TypedefNameDeclOrQualifier.get<ExtInfo*>(); }
    const ExtInfo *getExtInfo() const {
      return TypedefNameDeclOrQualifier.get<ExtInfo*>();
    }

  protected:
    MeshDecl(Kind            DK,
             MeshKind        TK,
             const ASTContext &ASTC,
             DeclContext    *DC,
             SourceLocation  L,
             IdentifierInfo *Id,
             MeshDecl       *PrevDecl,
             SourceLocation StartL)
      : TypeDecl(DK, DC, L, Id, StartL), DeclContext(DK), Redeclarable(ASTC),
        TypedefNameDeclOrQualifier((TypedefNameDecl*) 0) {
      MeshDeclKind                    = TK;
      IsCompleteDefinition            = false;
      IsCompleteDefinitionRequired    = false;
      IsBeingDefined                  = false;
      IsEmbeddedInDeclarator          = false;
      IsFreeStanding                  = false;
      HasVolatileMember               = false;
      HasCellData                     = false;
      HasVertexData                   = false;
      HasFaceData                     = false;
      HasEdgeData                     = false;
      LoadedFieldsFromExternalStorage = false;
      setPreviousDecl(PrevDecl);
    }

    typedef Redeclarable<MeshDecl> redeclarable_base;
    
    virtual MeshDecl *getPreviousDeclImpl() {
      return getPreviousDecl();
    }

      

    /// @brief Completes the definition of this mesh declaration.
    ///
    /// This is a helper function for derived classes.
    void completeDefinition();

  public:
    typedef redeclarable_base::redecl_iterator redecl_iterator;
    using   redeclarable_base::redecls_begin;
    using   redeclarable_base::redecls_end;
    using   redeclarable_base::getPreviousDecl;
    using   redeclarable_base::getMostRecentDecl;
    using   redeclarable_base::isFirstDecl;

    SourceLocation getRBraceLoc() const { return RBraceLoc; }
    void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

    /// getInnerLocStart - Return SourceLocation representing start of source
    /// range ignoring outer template declarations.
    SourceLocation getInnerLocStart() const { return getLocStart(); }

    /// getOuterLocStart - Return SourceLocation representing start of source
    /// range taking into account any outer template declarations.
    SourceLocation getOuterLocStart() const;
    virtual SourceRange getSourceRange() const LLVM_READONLY;
      
    virtual MeshDecl *getNextRedeclaration() const { return RedeclLink.getNext(this); }
      
    virtual MeshDecl *getMostRecentDeclImpl() {
        return getMostRecentDecl();
      }

    virtual MeshDecl* getCanonicalDecl();
    const MeshDecl* getCanonicalDecl() const {
      return const_cast<MeshDecl*>(this)->getCanonicalDecl();
    }

    /// isThisDeclarationADefinition() - Return true if this declaration
    /// is a completion definition of the type.  Provided for consistency.
    bool isThisDeclarationADefinition() const {
      return isCompleteDefinition();
    }

    /// isCompleteDefinition - Return true if this decl has its body
    /// fully specified.
    bool isCompleteDefinition() const {
      return IsCompleteDefinition;
    }

    /// \brief Return true if this complete decl is
    /// required to be complete for some existing use.
    bool isCompleteDefinitionRequired() const {
      return IsCompleteDefinitionRequired;
    }

    /// isBeingDefined - Return true if this decl is currently being defined.
    bool isBeingDefined() const {
      return IsBeingDefined;
    }

    bool isEmbeddedInDeclarator() const {
      return IsEmbeddedInDeclarator;
    }

    void setEmbeddedInDeclarator(bool isInDeclarator) {
      IsEmbeddedInDeclarator = isInDeclarator;
    }

    bool isFreeStanding() const { return IsFreeStanding; }
    void setFreeStanding(bool isFreeStanding = true) {
      IsFreeStanding = isFreeStanding;
    }

    bool hasVolatileMember() const { return HasVolatileMember; }
    void setHasVolatileMember (bool hasVolatileMember = true) {
      HasVolatileMember = hasVolatileMember; }

    /// \brief Whether this declaration declares a type that is
    /// dependent, i.e., a type that somehow depends on template
    /// parameters.
    bool isDependentType() const { return isDependentContext(); }

    /// @brief Starts the definition of this mesh declaration.
    ///
    /// This method should be invoked at the beginning of the definition
    /// of this mesh declaration. It will set the mesh type into a state
    /// where it is in the process of being defined.
    void startDefinition();

    /// getDefinition - Returns the MeshDecl that actually defines this
    ///  mesh.  When determining whether or not a mesh has a definition,
    /// one should use this method as opposed to 'isDefinition'.
    /// 'isDefinition' indicates whether or not a specific MeshDecl
    /// is defining declaration, not whether or not the mesh type is
    /// defined. This method returns NULL if there is no MeshDecl that
    /// defines the mesh.
    MeshDecl *getDefinition() const;

    void setCompleteDefinition(bool V) { IsCompleteDefinition = V; }

    // FIXME: Return StringRef;
    const char *getKindName() const;

    MeshKind getMeshKind() const {
      return MeshKind(MeshDeclKind);
    }

    void setMeshKind(MeshKind TK) { MeshDeclKind = TK; }

    bool isUniformMesh() const { return getMeshKind() == TTK_UniformMesh;  }
    bool isStructuredMesh() const { return getMeshKind() == TTK_StructuredMesh; }
    bool isRectilinearMesh() const{ return getMeshKind() == TTK_RectilinearMesh; }
    bool isUnstructuredMesh() const { return getMeshKind() == TTK_UnstructuredMesh; }
    bool isMesh() const {
      return isUniformMesh()     ||
             isStructuredMesh()  ||
             isRectilinearMesh() ||
             isUnstructuredMesh();
    }

    /// True if the mesh has one or more fields stored at the cells.
    bool hasCellData() const { return HasCellData; }

    /// Flag the mesh as having on or more fields stored at the cells.
    void setHasCellData(bool flag = true) { HasCellData = flag; }

    /// True if the mesh has one or more fields stored at the vertices.
    bool hasVertexData() const { return HasVertexData; }

    /// Flag the mesh as having on or more fields stored at the vertices.
    void setHasVertexData(bool flag = true) { HasVertexData = flag; }

    /// True if the mesh has one or more fields stored at the edges.
    bool hasEdgeData() const { return HasEdgeData; }

    /// Flag the mesh as having on or more fields stored at the edges.
    void setHasEdgeData(bool flag = true) { HasEdgeData = flag; }

    /// True if the mesh has one or more fields stored at the faces.
    bool hasFaceData() const { return HasFaceData; }

    /// Flag the mesh as having on or more fields stored at the faces.
    void setHasFaceData(bool flag = true) { HasFaceData = flag; }

    bool hasValidFieldData() const {
      if (fields() > 0) {
        if (hasCellData()   ||
            hasVertexData() ||
            hasEdgeData()   ||
            hasFaceData()) {
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
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
    // members) in this mesh.
    bool field_empty() const {
      return field_begin() == field_end();
    }

    unsigned int fields() const {
      unsigned int nfields = 0;
      for(field_iterator itr = field_begin(); itr != field_end(); ++itr) nfields++;
      return nfields;
    }

    /// Is this mesh type named, either directly or via being defined in
    /// a typedef of this type?
    ///
    /// SC_TODO - we need to define what we do in Scout's extensions below.
    ///
    /// C++11 [basic.link]p8:
    ///   A type is said to have linkage if and only if:
    ///     - it is a class or enumeration type that is named (or has a
    ///       name for linkage purposes) and the name has linkage; ...
    /// C++11 [dcl.typedef]p9:
    ///   If the typedef declaration defines an unnamed class (or enum),
    ///   the first typedef-name declared by the declaration to be that
    ///   class type (or enum type) is used to denote the class type (or
    ///   enum type) for linkage purposes only.
    ///
    /// C does not have an analogous rule, but the same concept is
    /// nonetheless useful in some places.
    bool hasNameForLinkage() const {
      return (getDeclName() || getTypedefNameForAnonDecl());
    }

    TypedefNameDecl *getTypedefNameForAnonDecl() const {
      return hasExtInfo() ? 0 :
             TypedefNameDeclOrQualifier.get<TypedefNameDecl*>();
    }

    void setTypedefNameForAnonDecl(TypedefNameDecl *TDD);
    /// \brief Retrieve the nested-name-specifier that qualifies the name of this
    /// declaration, if it was present in the source.
    NestedNameSpecifier *getQualifier() const {
      return hasExtInfo() ? getExtInfo()->QualifierLoc.getNestedNameSpecifier() : 0;
    }

    /// \brief Retrieve the nested-name-specifier (with source-location
    /// information) that qualifies the name of this declaration, if it was
    /// present in the source.
    NestedNameSpecifierLoc getQualifierLoc() const {
      return hasExtInfo() ? getExtInfo()->QualifierLoc : NestedNameSpecifierLoc();
    }

    void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

    // Implement isa/cast/dyncast/etc.
    static bool classof(const Decl *D) { return classofKind(D->getKind()); }
    static bool classofKind(Kind K) { return K >= firstMesh && K <= lastMesh; }

    static DeclContext *castToDeclContext(const MeshDecl *D) {
      return static_cast<DeclContext *>(const_cast<MeshDecl*>(D));
    }
    static MeshDecl *castFromDeclContext(const DeclContext *DC) {
      return static_cast<MeshDecl *>(const_cast<DeclContext*>(DC));
    }

    friend class ASTDeclReader;
    friend class ASTDeclWriter;

   protected:
    /// \brief De-serialize just the fields.
    virtual void LoadFieldsFromExternalStorage() const;
  };
} // end namespace clang

#endif
