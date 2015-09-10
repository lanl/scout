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

#include "lldb/Symbol/ClangASTContext.h"

// C Includes
// C++ Includes
#include <mutex> // std::once
#include <string>

// Other libraries and framework includes

// Clang headers like to use NDEBUG inside of them to enable/disable debug
// related features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/LangStandard.h"

#include "clang/AST/Scout/UniformMeshDecl.h"
#include "clang/AST/Scout/ALEMeshDecl.h"
#include "clang/AST/Scout/StructuredMeshDecl.h"
#include "clang/AST/Scout/RectilinearMeshDecl.h"
#include "clang/AST/Scout/UnstructuredMeshDecl.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

#include "llvm/Support/Signals.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ThreadSafeDenseMap.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

#include <stdio.h>

#include <mutex>

//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace clang;

namespace{
  
  enum {
    FlagMeshFieldCellLocated     = 1 << 0,
    FlagMeshFieldVertexLocated   = 1 << 1,
    FlagMeshFieldEdgeLocated     = 1 << 2,
    FlagMeshFieldFaceLocated     = 1 << 3
  };
  
} // namespace

CompilerType
ClangASTContext::CreateUniformMeshType (DeclContext *decl_ctx,
                                        AccessType access_type,
                                        const char *name,
                                        unsigned dimX,
                                        unsigned dimY,
                                        unsigned dimZ,
                                        LanguageType language,
                                        ClangASTMetadata *metadata)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);

    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();

    UniformMeshDecl *decl = UniformMeshDecl::Create (*ast,
                                                     decl_ctx,
                                                     SourceLocation(),
                                                     SourceLocation(),
                                                     name && name[0] ? &ast->Idents.get(name) : NULL);

    if (decl)
    {
        if (metadata)
            SetMetadata(ast, decl, *metadata);

        if (access_type != eAccessNone)
            decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));

        if (decl_ctx)
            decl_ctx->addDecl (decl);

        MeshType::MeshDimensions dims;
        for(size_t i = 0; i < 3; ++i){
          unsigned dim;

          switch(i){
          case 0:
            dim = dimX;
            break;
          case 1:
            dim = dimY;
            break;
          case 2:
            dim = dimZ;
            break;
          }

          if(dim == 0){
            break;
          }

          dims.push_back(IntegerLiteral::Create(*ast, APInt(32, dim), ast->UnsignedIntTy, SourceLocation()));
        }

        return CompilerType(ast, ast->getUniformMeshType(decl, dims));
    }
    return CompilerType();
}

CompilerType
ClangASTContext::CreateALEMeshType (DeclContext *decl_ctx,
                                        AccessType access_type,
                                        const char *name,
                                        unsigned dimX,
                                        unsigned dimY,
                                        unsigned dimZ,
                                        LanguageType language,
                                        ClangASTMetadata *metadata)
{
  ASTContext *ast = getASTContext();
  assert (ast != NULL);
  
  if (decl_ctx == NULL)
    decl_ctx = ast->getTranslationUnitDecl();
  
  ALEMeshDecl *decl = ALEMeshDecl::Create (*ast,
                                                   decl_ctx,
                                                   SourceLocation(),
                                                   SourceLocation(),
                                                   name && name[0] ? &ast->Idents.get(name) : NULL);
  
  if (decl)
  {
    if (metadata)
      SetMetadata(ast, decl, *metadata);
    
    if (access_type != eAccessNone)
      decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
    
    if (decl_ctx)
      decl_ctx->addDecl (decl);
    
    MeshType::MeshDimensions dims;
    for(size_t i = 0; i < 3; ++i){
      unsigned dim;
      
      switch(i){
        case 0:
          dim = dimX;
          break;
        case 1:
          dim = dimY;
          break;
        case 2:
          dim = dimZ;
          break;
      }
      
      if(dim == 0){
        break;
      }
      
      dims.push_back(IntegerLiteral::Create(*ast, APInt(32, dim), ast->UnsignedIntTy, SourceLocation()));
    }
    
    return CompilerType(ast, ast->getALEMeshType(decl, dims));
  }
  return CompilerType();
}

CompilerType
ClangASTContext::CreateStructuredMeshType (DeclContext *decl_ctx,
                                           AccessType access_type,
                                           const char *name,
                                           unsigned dimX,
                                           unsigned dimY,
                                           unsigned dimZ,
                                           LanguageType language,
                                           ClangASTMetadata *metadata)
{
  assert(false && "unimplemented");
}

CompilerType
ClangASTContext::CreateRectilinearMeshType (DeclContext *decl_ctx,
                                           AccessType access_type,
                                           const char *name,
                                           unsigned dimX,
                                           unsigned dimY,
                                           unsigned dimZ,
                                           LanguageType language,
                                           ClangASTMetadata *metadata)
{
  assert(false && "unimplemented");
}

CompilerType
ClangASTContext::CreateUnstructuredMeshType (DeclContext *decl_ctx,
                                             AccessType access_type,
                                             const char *name,
                                             unsigned dimX,
                                             unsigned dimY,
                                             unsigned dimZ,
                                             LanguageType language,
                                             ClangASTMetadata *metadata)
{
  assert(false && "unimplemented");
}

CompilerType
ClangASTContext::CreateFrameType (DeclContext *decl_ctx,
                                  AccessType access_type,
                                  const char *name,
                                  LanguageType language,
                                  ClangASTMetadata *metadata)
{
  ASTContext *ast = getASTContext();
  assert (ast != NULL);
  
  if (decl_ctx == NULL)
    decl_ctx = ast->getTranslationUnitDecl();
  
  FrameDecl *decl = FrameDecl::Create (*ast,
                                       decl_ctx,
                                       SourceLocation(),
                                       SourceLocation(),
                                       name && name[0] ? &ast->Idents.get(name) : NULL);
  
  if (decl)
  {
    if (metadata)
      SetMetadata(ast, decl, *metadata);
    
    if (access_type != eAccessNone)
      decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
    
    if (decl_ctx)
      decl_ctx->addDecl (decl);
    
    return CompilerType(ast, ast->getFrameType(decl));
  }
  return CompilerType();
}

bool
ClangASTContext::StartScoutDeclarationDefinition (const CompilerType &type)
{
    if (type)
    {
        QualType qual_type (GetQualType(type));
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
ClangASTContext::CompleteScoutDeclarationDefinition (const CompilerType &type)
{
    if (type)
    {
        QualType qual_type (GetQualType(type));
        
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
ClangASTContext::AddFieldToMeshType (const CompilerType& type,
                                     const char *name,
                                  const CompilerType &field_clang_type,
                                  AccessType access,
                                  uint32_t bitfield_bit_size,
                                  uint32_t field_flags)
{  
  if (!type.IsValid() || !field_clang_type.IsValid())
        return NULL;

  ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(type.GetTypeSystem());
  if (!ast)
    return nullptr;
  
  clang::ASTContext* clang_ast = ast->getASTContext();
  
    MeshFieldDecl *field = NULL;

    clang::Expr *bit_width = NULL;
    if (bitfield_bit_size != 0)
    {
        APInt bitfield_bit_size_apint(clang_ast->getTypeSize(clang_ast->IntTy), bitfield_bit_size);
        bit_width = new (*clang_ast)IntegerLiteral (*clang_ast, bitfield_bit_size_apint, clang_ast->IntTy, SourceLocation());
    }

    MeshDecl *mesh_decl = GetAsMeshDecl (type);
    assert(mesh_decl && "Expected a MeshDecl");

    field = MeshFieldDecl::Create (*clang_ast,
                                   mesh_decl,
                                   SourceLocation(),
                                   SourceLocation(),
                                   name ? &clang_ast->Idents.get(name) : NULL,  // Identifier
                                   GetQualType(field_clang_type),          // Field type
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

      if(field_flags & FlagMeshFieldCellLocated){
        field->setCellLocated(true);
        mesh_decl->setHasCellData(true);
      }
      else if(field_flags & FlagMeshFieldVertexLocated){
        field->setVertexLocated(true);
        mesh_decl->setHasVertexData(true);
      }
      else if(field_flags & FlagMeshFieldEdgeLocated){
        field->setEdgeLocated(true);
        mesh_decl->setHasEdgeData(true);
      }
      else if(field_flags & FlagMeshFieldFaceLocated){
        field->setFaceLocated(true);
        mesh_decl->setHasFaceData(true);
      }
    }

    return field;
}

clang::MeshDecl *
ClangASTContext::GetAsMeshDecl (const CompilerType& type)
{
  const clang::MeshType *mesh_type = dyn_cast<clang::MeshType>(GetCanonicalQualType(type));
    if (mesh_type)
        return mesh_type->getDecl();
    return nullptr;
}

clang::FrameDecl *
ClangASTContext::GetAsFrameDecl (const CompilerType& type)
{
  const clang::FrameType *frame_type = dyn_cast<clang::FrameType>(GetCanonicalQualType(type));
  if (frame_type)
    return frame_type->getDecl();
  return nullptr;
}

clang::VarDecl*
ClangASTContext::AddFieldToFrameType (const CompilerType& type,
                                      const char *name,
                                      const CompilerType &field_clang_type,
                                      uint32_t varId)
{
  if (!type.IsValid() || !field_clang_type.IsValid())
    return NULL;
  
  ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(type.GetTypeSystem());
  if (!ast)
    return nullptr;
  
  clang::ASTContext* clang_ast = ast->getASTContext();
  
  FrameDecl *frame_decl = GetAsFrameDecl (type);
  assert(frame_decl && "Expected a FrameDecl");
  
  VarDecl* VD =
  VarDecl::Create(*clang_ast, frame_decl, SourceLocation(), SourceLocation(),
                  &clang_ast->Idents.get(name),
                  GetQualType(field_clang_type),
                  clang_ast->getTrivialTypeSourceInfo(GetQualType(field_clang_type)),
                  SC_Static);

  frame_decl->addVar(name, VD, varId);
  frame_decl->addDecl(VD);
  
  return VD;
}
