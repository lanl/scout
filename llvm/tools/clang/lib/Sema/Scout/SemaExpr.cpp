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

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"

using namespace clang;
using namespace sema;


// Check for undeclared identifiers to see if they can be qualified as
// member reference expr's by enclosing forall / renderall loop
// variables.  ER is modified by this call
bool Sema::ScoutMemberReferenceExpr(DeclarationName &Name,
                                    SourceLocation &NameLoc,
                                    DeclarationNameInfo &NameInfo,
                                    CXXScopeSpec &SS,
                                    const TemplateArgumentListInfo *&TemplateArgs,
                                    ExprResult &ER) {

  for(ScoutLoopStack::iterator sitr = SCLStack.begin(), sitrEnd = SCLStack.end();
      sitr != sitrEnd; ++sitr) {

    VarDecl* vd = *sitr;
    ImplicitMeshParamDecl* ip = dyn_cast<ImplicitMeshParamDecl>(vd);
    assert(ip && "Expected an implicit mesh param decl");
    ImplicitMeshParamDecl::MeshElementType et = ip->getElementType();

    const MeshType* mt = dyn_cast<MeshType>(vd->getType().getCanonicalType());
    MeshDecl* md = mt->getDecl();

    for(MeshDecl::field_iterator fitr = md->field_begin(),
        fitrEnd = md->field_end();
        fitr != fitrEnd; ++fitr) {

      MeshFieldDecl* fd = *fitr;

      if(fd->isCellLocated()){
        if(et != ImplicitMeshParamDecl::Cells){
          continue;
        }
      }
      else if(fd->isVertexLocated()){
        if(et != ImplicitMeshParamDecl::Vertices){
          continue;
        }
      }
      else if(fd->isEdgeLocated()){
        if(et != ImplicitMeshParamDecl::Edges){
          continue;
        }
      }
      else if(fd->isFaceLocated()){
        if(et != ImplicitMeshParamDecl::Faces){
          continue;
        }
      }

      bool valid = fd->isValidLocation();

      if (valid && Name.getAsString() == fd->getName()) {
        Expr* baseExpr;
        baseExpr = BuildDeclRefExpr(vd, QualType(mt, 0),
                                    VK_LValue, NameLoc).get();

        ER = Owned(BuildMeshMemberReferenceExpr(baseExpr, QualType(mt, 0),
                                                NameLoc, false, SS,
                                                SourceLocation(), 0,
                                                NameInfo,
                                                TemplateArgs));
        return true;
      }
    }
  }
  return false;
}
