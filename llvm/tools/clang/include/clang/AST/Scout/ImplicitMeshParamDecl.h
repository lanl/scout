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
 * Notes: Based on ImplicitParamDecl, but with pointer to VarDecl of underlying mesh
 * so we can get to the underlying mesh from the implicit mesh
 * #####
 */

#ifndef __SC_CLANG_IMPLICIT_MESH_PARAM_DECL_H__
#define __SC_CLANG_IMPLICIT_MESH_PARAM_DECL_H__

#include "clang/AST/Decl.h"
#include "clang/AST/Scout/MeshDecl.h"

namespace clang {

  class ImplicitMeshParamDecl : public ImplicitParamDecl {
  public:
    enum MeshElementType {
      Undefined    = -1,
      Cells        =  1,
      Vertices     =  2,
      Edges        =  3,
      Faces        =  4
    };

    static ImplicitMeshParamDecl *Create(ASTContext &C, DeclContext *DC,
        MeshElementType ET, SourceLocation IdLoc, IdentifierInfo *Id,
        QualType Type, VarDecl *VD) {
      return new (C, DC) ImplicitMeshParamDecl(C, DC, ET, IdLoc, Id, Type, VD);
    }

    ImplicitMeshParamDecl(ASTContext &C,
                          DeclContext *DC,
                          MeshElementType ET,
                          SourceLocation IdLoc,
                          IdentifierInfo *Id,
                          QualType Type,
                          VarDecl *VD)
          : ImplicitParamDecl(C, DC, IdLoc, Id, Type, ImplicitMeshParam) {
      BVD = VD;
      ElementType = ET;
    }

    const VarDecl *getBaseVarDecl() const {
      return BVD;
    }

    const VarDecl* getMeshVarDecl() const {
      const VarDecl* VD = BVD;
      for(;;){
        if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(VD)){
          VD = IP->getBaseVarDecl();
        }
        else{
          return VD;
        }
      }
    }

    MeshElementType getElementType() const{
      return ElementType;
    }

    // Implement isa/cast/dyncast/etc.
    static bool classof(const Decl *D) { return classofKind(D->getKind()); }
    static bool classofKind(Kind K) { return K == ImplicitMeshParam; }

    friend class ASTDeclReader;
    friend class ASTDeclWriter;
  private:
    // Parent of this implicit mesh param decl VarDecl
    // could be a MeshDecl or another ImplicitMeshParamDecl
    VarDecl *BVD;
    MeshElementType ElementType;
  };

}
#endif
