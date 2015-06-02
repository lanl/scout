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

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Support/raw_ostream.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"

#include "clang/AST/Scout/MeshDecl.h"

using namespace lldb_private;
using namespace clang;

bool
ClangASTImporter::CompleteMeshDecl (clang::MeshDecl *decl)
{
    ClangASTMetrics::RegisterDeclCompletion();

    DeclOrigin decl_origin = GetDeclOrigin(decl);

    if (!decl_origin.Valid())
        return false;

    if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
        return false;

    MinionSP minion_sp (GetMinion(&decl->getASTContext(), decl_origin.ctx));

    if (minion_sp)
        minion_sp->ImportDefinitionTo(decl, decl_origin.decl);

    return true;
}

bool
ClangASTImporter::CompleteMeshDeclWithOrigin(clang::MeshDecl *decl, clang::MeshDecl *origin_decl)
{
    ClangASTMetrics::RegisterDeclCompletion();

    clang::ASTContext *origin_ast_ctx = &origin_decl->getASTContext();

    if (!ClangASTContext::GetCompleteDecl(origin_ast_ctx, origin_decl))
        return false;

    MinionSP minion_sp (GetMinion(&decl->getASTContext(), origin_ast_ctx));

    if (minion_sp)
        minion_sp->ImportDefinitionTo(decl, origin_decl);

    ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());

    OriginMap &origins = context_md->m_origins;

    origins[decl] = DeclOrigin(origin_ast_ctx, origin_decl);

    return true;
}

bool
ClangASTImporter::CompleteFrameDecl (clang::FrameDecl *decl)
{
  ClangASTMetrics::RegisterDeclCompletion();
  
  DeclOrigin decl_origin = GetDeclOrigin(decl);
  
  if (!decl_origin.Valid())
    return false;
  
  if (!ClangASTContext::GetCompleteDecl(decl_origin.ctx, decl_origin.decl))
    return false;
  
  MinionSP minion_sp (GetMinion(&decl->getASTContext(), decl_origin.ctx));
  
  if (minion_sp)
    minion_sp->ImportDefinitionTo(decl, decl_origin.decl);
  
  for(auto itr = decl->decls_begin(), itrEnd = decl->decls_end();
      itr != itrEnd; ++itr){
    
    VarDecl* vd = dyn_cast<VarDecl>(*itr);
    if(!vd){
      continue;
    }
  
    decl->resetVar(vd->getName().str(), vd);
  }
  
  return true;
}

bool
ClangASTImporter::CompleteFrameDeclWithOrigin(clang::FrameDecl *decl, clang::FrameDecl *origin_decl)
{
  ClangASTMetrics::RegisterDeclCompletion();
  
  clang::ASTContext *origin_ast_ctx = &origin_decl->getASTContext();
  
  if (!ClangASTContext::GetCompleteDecl(origin_ast_ctx, origin_decl))
    return false;
  
  MinionSP minion_sp (GetMinion(&decl->getASTContext(), origin_ast_ctx));
  
  if (minion_sp)
    minion_sp->ImportDefinitionTo(decl, origin_decl);
  
  ASTContextMetadataSP context_md = GetContextMetadata(&decl->getASTContext());
  
  OriginMap &origins = context_md->m_origins;
  
  origins[decl] = DeclOrigin(origin_ast_ctx, origin_decl);
  
  return true;
}
