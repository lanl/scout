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

#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"

using namespace clang;
using namespace sema;

// compare mesh references to make sure they are compatible
bool Sema::CompareMeshRefTypes(SourceLocation &Loc,
    QualType &QT1, QualType &QT2, Sema::ReferenceCompareResult &Ref) {
  const Type *T1 = QT1.getTypePtr();
  const Type *T2 = QT2.getTypePtr();

  return CompareMeshTypes(Loc, T1, T2, Ref);
}

// compare mesh pointers to make sure they are compatible
bool Sema::CompareMeshPtrTypes(SourceLocation &Loc, QualType &QT1, QualType &QT2) {
  const Type *T1 = QT1.getTypePtr()->getPointeeType().getTypePtr();
  const Type *T2 = QT2.getTypePtr()->getPointeeType().getTypePtr();

  Sema::ReferenceCompareResult Ref;
  return CompareMeshTypes(Loc, T1, T2, Ref);
}

// helper used by CompareMeshRefTypes() and CompareMeshPtrTypes()
bool Sema::CompareMeshTypes(SourceLocation &Loc,
    const Type *T1, const Type *T2, Sema::ReferenceCompareResult &Ref) {

  if(T1->isMeshType() && T2->isMeshType()) {
#if 0 //skip dimension compare for now
        const MeshType* MT1 = dyn_cast<MeshType>(T1);
        const MeshType* MT2 = dyn_cast<MeshType>(T2);
        if(MT1->rankOf() != MT2->rankOf()) {
          Diag(Loc, diag::err_mesh_param_dimensionality_mismatch);
          Ref = Ref_Incompatible;
          return false;
        }
#endif
        //check that kinds match
        if ((T1->isUniformMeshType() && T2->isUniformMeshType())  ||
            (T1->isRectilinearMeshType() && T2->isRectilinearMeshType()) ||
            (T1->isStructuredMeshType() && T2->isStructuredMeshType()) ||
            (T1->isUnstructuredMeshType() && T2->isUnstructuredMeshType())) {
          Ref = Ref_Compatible;
          llvm::errs() << "Sema mesh ref/ptr compare ok\n";
          return true;
        }
    }
    return false;
}
