//===-- ClangASTType.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

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
ClangASTType::StartMeshDeclarationDefinition ()
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
        }
    }
    return false;
}

bool
ClangASTType::CompleteMeshDeclarationDefinition ()
{
    if (IsValid())
    {
        QualType qual_type (GetQualType());
        
        if(UniformMeshDecl *mesh_decl = qual_type->getAsUniformMeshDecl()){
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
    }
    return false;
}

clang::MeshFieldDecl *
ClangASTType::AddFieldToMeshType (const char *name,
                                  const ClangASTType &field_clang_type,
                                  AccessType access,
                                  uint32_t bitfield_bit_size)
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
