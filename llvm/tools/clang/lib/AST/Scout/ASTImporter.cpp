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

