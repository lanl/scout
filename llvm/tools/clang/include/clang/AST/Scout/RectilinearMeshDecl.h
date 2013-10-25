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

#ifndef __SC_CLANG_RECTLINEAR_MESH_DECL_H__
#define __SC_CLANG_RECTLINEAR_MESH_DECL_H__


#include "clang/AST/Scout/MeshDecl.h"

namespace clang {

// ----- RectilinearMeshDecl
//
class RectilinearMeshDecl : public MeshDecl {

protected:
  RectilinearMeshDecl(DeclContext* DC,
                      SourceLocation L,
                      SourceLocation StartL,
                      IdentifierInfo* Id,
                      RectilinearMeshDecl* PrevDecl);

public:
  static RectilinearMeshDecl *Create(const ASTContext &C,
                                     DeclContext *DC,
                                     SourceLocation StartLoc,
                                     SourceLocation IdLoc,
                                     IdentifierInfo *Id,
                                     RectilinearMeshDecl* PrevDecl = 0);

  static RectilinearMeshDecl *CreateDeserialized(const ASTContext &C,
                                                 unsigned ID);

  const RectilinearMeshDecl *getPreviousDecl() const {
    return cast_or_null<RectilinearMeshDecl>(MeshDecl::getPreviousDecl());
  }

  RectilinearMeshDecl *getPreviousDecl() {
    return cast_or_null<RectilinearMeshDecl>(MeshDecl::getPreviousDecl());
  }

  const RectilinearMeshDecl *getMostRecentDecl() const {
    return cast<RectilinearMeshDecl>(MeshDecl::getMostRecentDecl());
  }

  RectilinearMeshDecl *getMostRecentDecl() {
    return cast<RectilinearMeshDecl>(MeshDecl::getMostRecentDecl());
  }

  void completeDefinition() {
    assert(!isCompleteDefinition() && "Cannot redefine rectilinear mesh!");
    MeshDecl::completeDefinition();
  }

  static bool classof(const Decl* D) { return classofKind(D->getKind()); }
  static bool classof(const RectilinearMeshDecl* D) { return true; }
  static bool classofKind(Kind K) { return K == RectilinearMesh; }
};


} // end namespace clang

#endif
