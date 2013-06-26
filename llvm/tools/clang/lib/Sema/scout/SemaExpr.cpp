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
using namespace clang;
using namespace sema;

// Check for undeclared identifiers to see if they can be qualified
// as member reference expr's by enclosing forall / renderall
// loop variables.
// ER is modified by this call
bool Sema::ScoutMemberReferenceExpr(DeclarationName &Name,
    SourceLocation &NameLoc,
    DeclarationNameInfo &NameInfo,
    CXXScopeSpec &SS,
    const TemplateArgumentListInfo *&TemplateArgs,
    ExprResult &ER
) {
  for(ScoutLoopStack::iterator sitr = SCLStack.begin(),
      sitrEnd = SCLStack.end();
      sitr != sitrEnd; ++sitr) {

    VarDecl* vd = *sitr;

    const MeshType* mt = dyn_cast<MeshType>(vd->getType().getCanonicalType());
    MeshDecl* md = mt->getDecl();

    for(MeshDecl::mesh_field_iterator fitr = md->mesh_field_begin(),
        fitrEnd = md->mesh_field_end();
        fitr != fitrEnd; ++fitr) {

      MeshFieldDecl* fd = *fitr;

      bool valid;

      if (mt->getInstanceType() == MeshType::MeshInstance) {
        valid = true;
      } else {

        switch(fd->meshLocation()) {

        case MeshFieldDecl::VertexLoc:
          valid = mt->getInstanceType() == MeshType::VerticesInstance;
          break;

        case MeshFieldDecl::CellLoc:
          valid = mt->getInstanceType() == MeshType::CellsInstance;
          break;

        case MeshFieldDecl::FaceLoc:
          valid = mt->getInstanceType() == MeshType::FacesInstance;
          break;

        case MeshFieldDecl::EdgeLoc:
          valid = mt->getInstanceType() == MeshType::EdgesInstance;
          break;

        case MeshFieldDecl::BuiltIn:
          valid = true;
          break;

        default:
          assert(false && "invalid field type while attempting "
              "to look up unqualified forall/renderall variable");
        }
      }

      if (valid && Name.getAsString() == fd->getName()) {
        Expr* baseExpr =
            BuildDeclRefExpr(vd, QualType(mt, 0), VK_LValue, NameLoc).get();

        ER = Owned(BuildMemberReferenceExpr(baseExpr, QualType(mt, 0),
            NameLoc, false,
            SS, SourceLocation(),
            0, NameInfo,
            TemplateArgs));
        return true;

      }
    }
  }
  return false;
}


// If this is a mesh member in the case of assigning it to a pointer
// to allocated mesh values, make it think we have a pointer
// type as the mesh member.
// this is what is used by the externmeshalloc.sc test, which is currently broken.
// SC_TODO: hasExternalFormalLinkage() is not returning true even if mesh member is extern.
void Sema::ScoutMeshExternAlloc(Expr *LHSExpr, QualType &LHSType) {
  if (isa<MemberExpr>(LHSExpr)) {

    Expr *Base = LHSExpr->IgnoreParenImpCasts();
    MemberExpr *ME = dyn_cast<MemberExpr>(Base);
    Expr *BaseExpr = ME->getBase()->IgnoreParenImpCasts();

    if (BaseExpr->getStmtClass() == Expr::DeclRefExprClass) {
      const NamedDecl *ND = cast< DeclRefExpr >(BaseExpr)->getDecl();

      if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
        if (isa<MeshType>(VD->getType().getCanonicalType().getNonReferenceType())) {
          if (!isa<ImplicitParamDecl>(VD) ) {
            const MeshType *MT = cast<MeshType>(VD->getType().getCanonicalType());
            MeshDecl* MD = MT->getDecl();
            MeshDecl::mesh_field_iterator itr_end = MD->mesh_field_end();
            llvm::StringRef memberName = ME->getMemberDecl()->getName();
            for(MeshDecl::mesh_field_iterator itr = MD->mesh_field_begin(); itr != itr_end; ++itr) {
              if (dyn_cast<NamedDecl>(*itr)->getName() == memberName) {
                if ((*itr)->hasExternalFormalLinkage()) {
                  LHSType = Context.getPointerType(LHSType);
                } else {
                  Diags.getDiagnosticLevel(diag::err_typecheck_expression_not_modifiable_lvalue,
                                           LHSExpr->getLocStart());
                  // error
                }
              }
            } // end for
          }
        }
      }
    }
  }
}
