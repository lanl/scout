// Note - this file is included by the ASTImporter source file 
// one directory up (importer class is all contained in a single
// file there...). 
//  

// ===== Scout -- Visit Mesh Types  ===========================================

QualType 
ASTNodeImporter::VisitUniformMeshType(const UniformMeshType *T) {
  assert(T != 0);

  UniformMeshDecl *ToDecl;
  ToDecl = dyn_cast_or_null<UniformMeshDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getUniformMeshDeclType(ToDecl);
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


// ===== Scout -- Mesh Types Import ===========================================
// 
// SC_TODO - we need to implement these... 

Decl *ASTNodeImporter::VisitUniformMeshDecl(UniformMeshDecl *D) {
  assert(D != 0);
  return 0;
}

Decl *ASTNodeImporter::VisitStructuredMeshDecl(StructuredMeshDecl *D) {
  assert(D != 0);
  return 0;
}

Decl *ASTNodeImporter::VisitRectilinearMeshDecl(RectilinearMeshDecl *D) {
  assert(D != 0);
  return 0;
}

Decl *ASTNodeImporter::VisitUnstructuredMeshDecl(UnstructuredMeshDecl *D) {
  assert(D != 0);
  return 0;
}

