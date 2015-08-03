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

#include "lldb/Symbol/ClangASTType.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace clang;
using namespace llvm;

bool
ClangASTType::StartScoutDeclarationDefinition ()
{
    if (IsValid())
    {
        QualType qual_type (GetQualType());
        const clang::Type *t = qual_type.getTypePtr();
        if (t)
        {
            const MeshType *mesh_type = dyn_cast<MeshType>(t);
            
            if (mesh_type)
            {
                MeshDecl *mesh_decl = mesh_type->getDecl();
                if (mesh_decl)
                {
                    mesh_decl->startDefinition();
                    return true;
                }
            }
          
            const FrameType *frame_type = dyn_cast<FrameType>(t);
          
            if (frame_type)
            {
              FrameDecl *frame_decl = frame_type->getDecl();
              if (frame_decl)
              {
                frame_decl->startDefinition();
                return true;
              }
            }
        }
    }
    return false;
}

bool
ClangASTType::CompleteScoutDeclarationDefinition ()
{
    if (IsValid())
    {
        QualType qual_type (GetQualType());
        
        if(UniformMeshDecl *mesh_decl = qual_type->getAsUniformMeshDecl()){
          mesh_decl->completeDefinition();
          return true;
        }
        else if(ALEMeshDecl *mesh_decl = qual_type->getAsALEMeshDecl()){
          mesh_decl->completeDefinition();
          return true;
        }
        else if(StructuredMeshDecl *mesh_decl = qual_type->getAsStructuredMeshDecl()){
          mesh_decl->completeDefinition();
          return true;
        }
        else if(RectilinearMeshDecl *mesh_decl = qual_type->getAsRectilinearMeshDecl()){
          mesh_decl->completeDefinition();
          return true;
        }
        else if(UnstructuredMeshDecl *mesh_decl = qual_type->getAsUnstructuredMeshDecl()){
          mesh_decl->completeDefinition();
          return true;
        }
        else if(FrameDecl *frame_decl = qual_type->getAsFrameDecl()){
          frame_decl->completeDefinition();
          return true;
        }
    }
    return false;
}

clang::MeshFieldDecl *
ClangASTType::AddFieldToMeshType (const char *name,
                                  const ClangASTType &field_clang_type,
                                  AccessType access,
                                  uint32_t bitfield_bit_size,
                                  uint32_t field_flags)
{  
  if (!IsValid() || !field_clang_type.IsValid())
        return NULL;

    MeshFieldDecl *field = NULL;

    clang::Expr *bit_width = NULL;
    if (bitfield_bit_size != 0)
    {
        APInt bitfield_bit_size_apint(m_ast->getTypeSize(m_ast->IntTy), bitfield_bit_size);
        bit_width = new (*m_ast)IntegerLiteral (*m_ast, bitfield_bit_size_apint, m_ast->IntTy, SourceLocation());
    }

    MeshDecl *mesh_decl = GetAsMeshDecl ();
    assert(mesh_decl && "Expected a MeshDecl");

    field = MeshFieldDecl::Create (*m_ast,
                                   mesh_decl,
                                   SourceLocation(),
                                   SourceLocation(),
                                   name ? &m_ast->Idents.get(name) : NULL,  // Identifier
                                   field_clang_type.GetQualType(),          // Field type
                                   NULL,            // TInfo *
                                   bit_width,       // BitWidth
                                   false,           // Mutable
                                   ICIS_NoInit);    // HasInit

    if (field)
    {
      field->setAccess (ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));

      mesh_decl->addDecl(field);

#ifdef LLDB_CONFIGURATION_DEBUG
      VerifyDecl(field);
#endif

      if(field_flags & llvm::DIScoutDerivedType::FlagMeshFieldCellLocated){
        field->setCellLocated(true);
        mesh_decl->setHasCellData(true);
      }
      else if(field_flags & llvm::DIScoutDerivedType::FlagMeshFieldVertexLocated){
        field->setVertexLocated(true);
        mesh_decl->setHasVertexData(true);
      }
      else if(field_flags & llvm::DIScoutDerivedType::FlagMeshFieldEdgeLocated){
        field->setEdgeLocated(true);
        mesh_decl->setHasEdgeData(true);
      }
      else if(field_flags & llvm::DIScoutDerivedType::FlagMeshFieldFaceLocated){
        field->setFaceLocated(true);
        mesh_decl->setHasFaceData(true);
      }
    }

    return field;
}

clang::MeshDecl *
ClangASTType::GetAsMeshDecl () const
{
    const MeshType *mesh_type = dyn_cast<MeshType>(GetCanonicalQualType());
    if (mesh_type)
        return mesh_type->getDecl();
    return NULL;
}

clang::FrameDecl *
ClangASTType::GetAsFrameDecl () const
{
  const FrameType *frame_type = dyn_cast<FrameType>(GetCanonicalQualType());
  if (frame_type)
    return frame_type->getDecl();
  return NULL;
}

clang::VarDecl*
ClangASTType::AddFieldToFrameType (const char *name,
                                   const ClangASTType &field_clang_type,
                                   uint32_t varId)
{
  if (!IsValid() || !field_clang_type.IsValid())
    return NULL;
  
  FrameDecl *frame_decl = GetAsFrameDecl ();
  assert(frame_decl && "Expected a FrameDecl");
  
  VarDecl* VD =
  VarDecl::Create(*m_ast, frame_decl, SourceLocation(), SourceLocation(),
                  &m_ast->Idents.get(name),
                  field_clang_type.GetQualType(),
                  m_ast->getTrivialTypeSourceInfo(field_clang_type.GetQualType()),
                  SC_Static);

  frame_decl->addVar(name, VD, varId);
  frame_decl->addDecl(VD);
  
  return VD;
}
