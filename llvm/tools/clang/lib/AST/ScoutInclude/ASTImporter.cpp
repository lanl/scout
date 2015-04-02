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

// Note - this file is included by the ASTImporter source file
// one directory up (importer class is all contained in a single
// file there...).
//

// ===== Scout -- Visit Mesh Types  ===========================================

bool ASTNodeImporter::ImportDefinition(MeshDecl *From, MeshDecl *To,
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything)
      ImportDeclContext(From, /*ForceImport=*/true);

    return false;
  }

  To->startDefinition();

  if (shouldForceImportDeclContext(Kind))
    ImportDeclContext(From, /*ForceImport=*/true);

  if(UniformMeshDecl* MD = dyn_cast<UniformMeshDecl>(To)){
    MD->completeDefinition();
  }
  else if(ALEMeshDecl* MD = dyn_cast<ALEMeshDecl>(To)){
    MD->completeDefinition();
  }
  else if(StructuredMeshDecl* MD = dyn_cast<StructuredMeshDecl>(To)){
    MD->completeDefinition();
  }
  else if(RectilinearMeshDecl* MD = dyn_cast<RectilinearMeshDecl>(To)){
    MD->completeDefinition();
  }
  else if(UnstructuredMeshDecl* MD = dyn_cast<UnstructuredMeshDecl>(To)){
    MD->completeDefinition();
  }

  return false;
}

bool ASTNodeImporter::ImportDefinition(FrameDecl *From, FrameDecl *To,
                                       ImportDefinitionKind Kind) {
  if (To->getDefinition() || To->isBeingDefined()) {
    if (Kind == IDK_Everything)
      ImportDeclContext(From, /*ForceImport=*/true);
    
    return false;
  }
  
  To->startDefinition();
  To->completeDefinition();
  
  return false;
}

QualType
ASTNodeImporter::VisitUniformMeshType(const UniformMeshType *T) {
  assert(T != 0);

  UniformMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<UniformMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getUniformMeshType(ToDecl, T->dimensions());
}

QualType
ASTNodeImporter::VisitALEMeshType(const ALEMeshType *T) {
  assert(T != 0);
  
  ALEMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<ALEMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getALEMeshType(ToDecl, T->dimensions());
}

QualType
ASTNodeImporter::VisitStructuredMeshType(const StructuredMeshType *T) {
  assert(T != 0);

  StructuredMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<StructuredMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getStructuredMeshDeclType(ToDecl);
}

QualType
ASTNodeImporter::VisitRectilinearMeshType(const RectilinearMeshType *T) {
  assert(T != 0);

  RectilinearMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<RectilinearMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getRectilinearMeshDeclType(ToDecl);
}

QualType
ASTNodeImporter::VisitUnstructuredMeshType(const UnstructuredMeshType *T) {
  assert(T != 0);

  UnstructuredMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<UnstructuredMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getUnstructuredMeshDeclType(ToDecl);
}

QualType
ASTNodeImporter::VisitWindowType(const WindowType *T) {
  llvm::SmallVector<Expr*, 2> dims;
  dims.push_back(T->getWidthExpr());
  dims.push_back(T->getHeightExpr());
  return Importer.getToContext().getWindowType(dims);
}

QualType
ASTNodeImporter::VisitImageType(const ImageType *T) {
  llvm::SmallVector<Expr*, 2> dims;
  dims.push_back(T->getWidthExpr());
  dims.push_back(T->getHeightExpr());
  return Importer.getToContext().getImageType(dims);
}

QualType
ASTNodeImporter::VisitFrameType(const FrameType *T) {
  assert(T != 0);
  
  FrameDecl *ToDecl;
  ToDecl = dyn_cast_or_null<FrameDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getFrameType(ToDecl);
}

// ===== Scout -- Mesh Types Import ===========================================

Decl *ASTNodeImporter::VisitUniformMeshDecl(UniformMeshDecl *D) {
  MeshDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  UniformMeshDecl *D2 =
      UniformMeshDecl::Create(Importer.getToContext(),
                              DC,
                              StartLoc, Loc,
                              Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  D2->setHasCellData(D->hasCellData());
  D2->setHasVertexData(D->hasVertexData());
  D2->setHasEdgeData(D->hasEdgeData());
  D2->setHasFaceData(D->hasFaceData());

  Importer.Imported(D, D2);

  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;

  return D2;
}

Decl *ASTNodeImporter::VisitALEMeshDecl(ALEMeshDecl *D) {
  MeshDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  ALEMeshDecl *D2 =
  ALEMeshDecl::Create(Importer.getToContext(),
                          DC,
                          StartLoc, Loc,
                          Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);
  
  D2->setHasCellData(D->hasCellData());
  D2->setHasVertexData(D->hasVertexData());
  D2->setHasEdgeData(D->hasEdgeData());
  D2->setHasFaceData(D->hasFaceData());
  
  Importer.Imported(D, D2);
  
  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;
  
  return D2;
}


Decl *ASTNodeImporter::VisitStructuredMeshDecl(StructuredMeshDecl *D) {
  MeshDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  StructuredMeshDecl *D2 =
      StructuredMeshDecl::Create(Importer.getToContext(),
                                 DC,
                                 StartLoc, Loc,
                                 Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  D2->setHasCellData(D->hasCellData());
  D2->setHasVertexData(D->hasVertexData());
  D2->setHasEdgeData(D->hasEdgeData());
  D2->setHasFaceData(D->hasFaceData());

  Importer.Imported(D, D2);

  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;

  return D2;
}

Decl *ASTNodeImporter::VisitRectilinearMeshDecl(RectilinearMeshDecl *D) {
  MeshDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  RectilinearMeshDecl *D2 =
      RectilinearMeshDecl::Create(Importer.getToContext(),
                                  DC,
                                  StartLoc, Loc,
                                  Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  D2->setHasCellData(D->hasCellData());
  D2->setHasVertexData(D->hasVertexData());
  D2->setHasEdgeData(D->hasEdgeData());
  D2->setHasFaceData(D->hasFaceData());

  Importer.Imported(D, D2);

  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;

  return D2;
}

Decl *ASTNodeImporter::VisitUnstructuredMeshDecl(UnstructuredMeshDecl *D) {
  MeshDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;

    return Importer.Imported(D, ImportedDef);
  }

  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  UnstructuredMeshDecl *D2 =
      UnstructuredMeshDecl::Create(Importer.getToContext(),
                                   DC,
                                   StartLoc, Loc,
                                   Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);

  D2->setHasCellData(D->hasCellData());
  D2->setHasVertexData(D->hasVertexData());
  D2->setHasEdgeData(D->hasEdgeData());
  D2->setHasFaceData(D->hasFaceData());

  Importer.Imported(D, D2);

  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;

  return D2;
}

Decl *ASTNodeImporter::VisitMeshFieldDecl(MeshFieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Determine whether we've already imported this field.
  SmallVector<NamedDecl *, 2> FoundDecls;
  DC->localUncachedLookup(Name, FoundDecls);
  for (unsigned I = 0, N = FoundDecls.size(); I != N; ++I) {
    if (FieldDecl *FoundField = dyn_cast<FieldDecl>(FoundDecls[I])) {
      // For anonymous fields, match up by index.
      if (!Name && getFieldIndex(D) != getFieldIndex(FoundField))
        continue;

      if (Importer.IsStructurallyEquivalent(D->getType(),
                                            FoundField->getType())) {
        Importer.Imported(D, FoundField);
        return FoundField;
      }

      Importer.ToDiag(Loc, diag::err_odr_field_type_inconsistent)
        << Name << D->getType() << FoundField->getType();
      Importer.ToDiag(FoundField->getLocation(), diag::note_odr_value_here)
        << FoundField->getType();
      return 0;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;

  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return 0;

  MeshFieldDecl *ToField =
      MeshFieldDecl::Create(Importer.getToContext(), DC,
                            Importer.Import(D->getInnerLocStart()),
                            Loc, Name.getAsIdentifierInfo(),
                            T, TInfo, BitWidth, D->isMutable(),
                            D->getInClassInitStyle());

  if(D->isCellLocated()){
    ToField->setCellLocated(true);
  }
  else if(D->isVertexLocated()){
    ToField->setVertexLocated(true);
  }
  else if(D->isEdgeLocated()){
    ToField->setEdgeLocated(true);
  }
  else if(D->isFaceLocated()){
    ToField->setFaceLocated(true);
  }

  ToField->setAccess(D->getAccess());
  ToField->setLexicalDeclContext(LexicalDC);
  if (ToField->hasInClassInitializer())
    ToField->setInClassInitializer(D->getInClassInitializer());
  ToField->setImplicit(D->isImplicit());
  Importer.Imported(D, ToField);
  LexicalDC->addDeclInternal(ToField);
  return ToField;
}

Decl *ASTNodeImporter::VisitFrameDecl(FrameDecl *D) {
  FrameDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  FrameDecl *D2 =
  FrameDecl::Create(Importer.getToContext(),
                    DC,
                    StartLoc, Loc,
                    Name.getAsIdentifierInfo());
  D2->setAccess(D->getAccess());
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDeclInternal(D2);
  
  Importer.Imported(D, D2);
  
  if (D->isCompleteDefinition() && ImportDefinition(D, D2, IDK_Default))
    return 0;
  
  return D2;
}
