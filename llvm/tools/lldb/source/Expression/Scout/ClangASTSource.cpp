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

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace clang;
using namespace lldb_private;

void
ClangASTSource::CompleteType (MeshDecl *mesh_decl)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    static unsigned int invocation_id = 0;
    unsigned int current_id = invocation_id++;

    if (log)
    {
        log->Printf("    CompleteMeshDecl[%u] on (ASTContext*)%p Completing (MeshDecl*)%p named %s",
                    current_id,
                    (void *)m_ast_context,
                    (void *)mesh_decl,
                    mesh_decl->getName().str().c_str());

        log->Printf("      CTD[%u] Before:", current_id);
        ASTDumper dumper((Decl*)mesh_decl);
        dumper.ToLog(log, "      [CTD] ");
    }

    if (!m_ast_importer->CompleteMeshDecl (mesh_decl))
    {
        // We couldn't complete the type.  Maybe there's a definition
        // somewhere else that can be completed.

        if (log)
            log->Printf("      CTD[%u] Type could not be completed in the module in which it was first found.", current_id);

        bool found = false;

        DeclContext *decl_ctx = mesh_decl->getDeclContext();

        if (const NamespaceDecl *namespace_context = dyn_cast<NamespaceDecl>(decl_ctx))
        {
            ClangASTImporter::NamespaceMapSP namespace_map = m_ast_importer->GetNamespaceMap(namespace_context);

            if (log && log->GetVerbose())
                log->Printf("      CTD[%u] Inspecting namespace map %p (%d entries)",
                            current_id,
                            (void *)namespace_map.get(),
                            (int)namespace_map->size());

            if (!namespace_map)
                return;

            for (ClangASTImporter::NamespaceMap::iterator i = namespace_map->begin(), e = namespace_map->end();
                 i != e && !found;
                 ++i)
            {
                if (log)
                    log->Printf("      CTD[%u] Searching namespace %s in module %s",
                                current_id,
                                i->second.GetNamespaceDecl()->getNameAsString().c_str(),
                                i->first->GetFileSpec().GetFilename().GetCString());

                TypeList types;

                SymbolContext null_sc;
                ConstString name(mesh_decl->getName().str().c_str());

                i->first->FindTypesInNamespace(null_sc, name, &i->second, UINT32_MAX, types);

                for (uint32_t ti = 0, te = types.GetSize();
                     ti != te && !found;
                     ++ti)
                {
                    lldb::TypeSP type = types.GetTypeAtIndex(ti);

                    if (!type)
                        continue;

                    ClangASTType clang_type (type->GetClangFullType());

                    if (!clang_type)
                        continue;

                    const MeshType *mesh_type = clang_type.GetQualType()->getAs<MeshType>();

                    if (!mesh_type)
                        continue;

                    MeshDecl *candidate_mesh_decl = const_cast<MeshDecl*>(mesh_type->getDecl());

                    if (m_ast_importer->CompleteMeshDeclWithOrigin (mesh_decl, candidate_mesh_decl))
                        found = true;
                }
            }
        }
        else
        {
            TypeList types;

            SymbolContext null_sc;
            ConstString name(mesh_decl->getName().str().c_str());
            ClangNamespaceDecl namespace_decl;

            const ModuleList &module_list = m_target->GetImages();

            bool exact_match = false;
            module_list.FindTypes (null_sc, name, exact_match, UINT32_MAX, types);

            for (uint32_t ti = 0, te = types.GetSize();
                 ti != te && !found;
                 ++ti)
            {
                lldb::TypeSP type = types.GetTypeAtIndex(ti);

                if (!type)
                    continue;

                ClangASTType clang_type (type->GetClangFullType());

                if (!clang_type)
                    continue;

                const MeshType *mesh_type = clang_type.GetQualType()->getAs<MeshType>();

                if (!mesh_type)
                    continue;

                MeshDecl *candidate_mesh_decl = const_cast<MeshDecl*>(mesh_type->getDecl());

                if (m_ast_importer->CompleteMeshDeclWithOrigin (mesh_decl, candidate_mesh_decl))
                    found = true;
            }
        }
    }

    if (log)
    {
        log->Printf("      [CTD] After:");
        ASTDumper dumper((Decl*)mesh_decl);
        dumper.ToLog(log, "      [CTD] ");
    }
}

void
ClangASTSource::CompleteType (FrameDecl *frame_decl)
{
  Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
  
  static unsigned int invocation_id = 0;
  unsigned int current_id = invocation_id++;
  
  if (log)
  {
    log->Printf("    CompleteFrameDecl[%u] on (ASTContext*)%p Completing (MeshDecl*)%p named %s",
                current_id,
                (void *)m_ast_context,
                (void *)frame_decl,
                frame_decl->getName().str().c_str());
    
    log->Printf("      CTD[%u] Before:", current_id);
    ASTDumper dumper((Decl*)frame_decl);
    dumper.ToLog(log, "      [CTD] ");
  }
  
  if (!m_ast_importer->CompleteFrameDecl (frame_decl))
  {
    // We couldn't complete the type.  Maybe there's a definition
    // somewhere else that can be completed.
    
    if (log)
      log->Printf("      CTD[%u] Type could not be completed in the module in which it was first found.", current_id);
    
    bool found = false;
    
    DeclContext *decl_ctx = frame_decl->getDeclContext();
    
    if (const NamespaceDecl *namespace_context = dyn_cast<NamespaceDecl>(decl_ctx))
    {
      ClangASTImporter::NamespaceMapSP namespace_map = m_ast_importer->GetNamespaceMap(namespace_context);
      
      if (log && log->GetVerbose())
        log->Printf("      CTD[%u] Inspecting namespace map %p (%d entries)",
                    current_id,
                    (void *)namespace_map.get(),
                    (int)namespace_map->size());
      
      if (!namespace_map)
        return;
      
      for (ClangASTImporter::NamespaceMap::iterator i = namespace_map->begin(), e = namespace_map->end();
           i != e && !found;
           ++i)
      {
        if (log)
          log->Printf("      CTD[%u] Searching namespace %s in module %s",
                      current_id,
                      i->second.GetNamespaceDecl()->getNameAsString().c_str(),
                      i->first->GetFileSpec().GetFilename().GetCString());
        
        TypeList types;
        
        SymbolContext null_sc;
        ConstString name(frame_decl->getName().str().c_str());
        
        i->first->FindTypesInNamespace(null_sc, name, &i->second, UINT32_MAX, types);
        
        for (uint32_t ti = 0, te = types.GetSize();
             ti != te && !found;
             ++ti)
        {
          lldb::TypeSP type = types.GetTypeAtIndex(ti);
          
          if (!type)
            continue;
          
          ClangASTType clang_type (type->GetClangFullType());
          
          if (!clang_type)
            continue;
          
          const FrameType *frame_type = clang_type.GetQualType()->getAs<FrameType>();
          
          if (!frame_type)
            continue;
          
          FrameDecl *candidate_frame_decl = const_cast<FrameDecl*>(frame_type->getDecl());
          
          if (m_ast_importer->CompleteFrameDeclWithOrigin (frame_decl, candidate_frame_decl))
            found = true;
        }
      }
    }
    else
    {
      TypeList types;
      
      SymbolContext null_sc;
      ConstString name(frame_decl->getName().str().c_str());
      ClangNamespaceDecl namespace_decl;
      
      const ModuleList &module_list = m_target->GetImages();
      
      bool exact_match = false;
      module_list.FindTypes (null_sc, name, exact_match, UINT32_MAX, types);
      
      for (uint32_t ti = 0, te = types.GetSize();
           ti != te && !found;
           ++ti)
      {
        lldb::TypeSP type = types.GetTypeAtIndex(ti);
        
        if (!type)
          continue;
        
        ClangASTType clang_type (type->GetClangFullType());
        
        if (!clang_type)
          continue;
        
        const FrameType *frame_type = clang_type.GetQualType()->getAs<FrameType>();
        
        if (!frame_type)
          continue;
        
        FrameDecl *candidate_frame_decl = const_cast<FrameDecl*>(frame_type->getDecl());
        
        if (m_ast_importer->CompleteFrameDeclWithOrigin (frame_decl, candidate_frame_decl))
          found = true;
      }
    }
  }
  
  if (log)
  {
    log->Printf("      [CTD] After:");
    ASTDumper dumper((Decl*)frame_decl);
    dumper.ToLog(log, "      [CTD] ");
  }
}
