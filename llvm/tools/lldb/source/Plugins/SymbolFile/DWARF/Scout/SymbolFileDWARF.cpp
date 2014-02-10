//===-- SymbolFileDWARF.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

// Other libraries and framework includes
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DeclSpec.h"

#include "llvm/Support/Casting.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Value.h"

#include "lldb/Host/Host.h"

#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/ClangExternalASTSourceCallbacks.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDebugLine.h"
#include "DWARFDebugPubnames.h"
#include "DWARFDebugRanges.h"
#include "DWARFDeclContext.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "DWARFLocationList.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARFDebugMap.h"

#include <map>

#include "llvm/Scout/DebugInfo.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#define DIE_IS_BEING_PARSED ((lldb_private::Type*)1)

using namespace lldb;
using namespace lldb_private;

struct BitfieldInfo
{
    uint64_t bit_size;
    uint64_t bit_offset;

    BitfieldInfo () :
        bit_size (LLDB_INVALID_ADDRESS),
        bit_offset (LLDB_INVALID_ADDRESS)
    {
    }

    void
    Clear()
    {
        bit_size = LLDB_INVALID_ADDRESS;
        bit_offset = LLDB_INVALID_ADDRESS;
    }

    bool IsValid ()
    {
        return (bit_size != LLDB_INVALID_ADDRESS) &&
               (bit_offset != LLDB_INVALID_ADDRESS);
    }
};

size_t
SymbolFileDWARF::ParseMeshChildMembers
(
    const SymbolContext& sc,
    DWARFCompileUnit* dwarf_cu,
    const DWARFDebugInfoEntry *parent_die,
    ClangASTType &class_clang_type,
    AccessType& default_accessibility,
    MeshLayoutInfo &layout_info
)
{
  if (parent_die == NULL)
      return 0;

  size_t count = 0;
  const DWARFDebugInfoEntry *die;
  const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize());
  uint32_t member_idx = 0;
  BitfieldInfo last_field_info;
  ModuleSP module = GetObjectFile()->GetModule();

  for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
  {
      dw_tag_t tag = die->Tag();

      switch (tag)
      {
      case DW_TAG_member:
        DWARFDebugInfoEntry::Attributes attributes;
        const size_t num_attributes = die->GetAttributes (this,
                                                          dwarf_cu,
                                                          fixed_form_sizes,
                                                          attributes);

        if (num_attributes > 0)
        {
            Declaration decl;
            //DWARFExpression location;
            const char *name = NULL;
            const char *prop_name = NULL;
            const char *prop_getter_name = NULL;
            const char *prop_setter_name = NULL;
            uint32_t prop_attributes = 0;


            lldb::user_id_t encoding_uid = LLDB_INVALID_UID;
            AccessType accessibility = eAccessNone;
            uint32_t member_byte_offset = UINT32_MAX;
            uint32_t i;
            uint32_t fieldFlags;
            for (i=0; i<num_attributes; ++i)
            {
              const dw_attr_t attr = attributes.AttributeAtIndex(i);
              DWARFFormValue form_value;
              if (attributes.ExtractFormValueAtIndex(this, i, form_value))
              {
                switch (attr)
                {
                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                case DW_AT_name:        name = form_value.AsCString(&get_debug_str_data()); break;
                case DW_AT_type:        encoding_uid = form_value.Reference(dwarf_cu); break;
                case DW_AT_data_member_location:
                  if (form_value.BlockData())
                  {
                    Value initialValue(0);
                    Value memberOffset(0);
                    const DWARFDataExtractor& debug_info_data = get_debug_info_data();
                    uint32_t block_length = form_value.Unsigned();
                    uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                    if (DWARFExpression::Evaluate(NULL, // ExecutionContext *
                        NULL, // ClangExpressionVariableList *
                        NULL, // ClangExpressionDeclMap *
                        NULL, // RegisterContext *
                        module,
                        debug_info_data,
                        block_offset,
                        block_length,
                        eRegisterKindDWARF,
                        &initialValue,
                        memberOffset,
                        NULL))
                    {
                      member_byte_offset = memberOffset.ResolveValue(NULL).UInt();
                    }
                  }
                  else
                  {
                    // With DWARF 3 and later, if the value is an integer constant,
                    // this form value is the offset in bytes from the beginning
                    // of the containing entity.
                    member_byte_offset = form_value.Unsigned();
                  }
                  break;
                case DW_AT_SCOUT_mesh_field_flags:
                  fieldFlags = form_value.Unsigned();
                  break;
                }
              }
            }

            Type *member_type = ResolveTypeUID(encoding_uid);
            clang::MeshFieldDecl *field_decl = NULL;
            if (tag == DW_TAG_member)
            {
                if (member_type)
                {
                  if (accessibility == eAccessNone)
                      accessibility = default_accessibility;

                  uint64_t field_bit_offset = (member_byte_offset == UINT32_MAX ? 0 : (member_byte_offset * 8));

                  last_field_info.Clear();

                  ClangASTType member_clang_type = member_type->GetClangLayoutType();

                  field_decl = class_clang_type.AddFieldToMeshType (name,
                                                                    member_clang_type,
                                                                    accessibility,
                                                                    0);

                  if(fieldFlags & llvm::DIScoutDerivedType::FlagMeshFieldCellLocated){
                    field_decl->setCellLocated(true);
                  }
                  else if(fieldFlags & llvm::DIScoutDerivedType::FlagMeshFieldVertexLocated){
                    field_decl->setVertexLocated(true);
                  }
                  else if(fieldFlags & llvm::DIScoutDerivedType::FlagMeshFieldEdgeLocated){
                    field_decl->setEdgeLocated(true);
                  }
                  else if(fieldFlags & llvm::DIScoutDerivedType::FlagMeshFieldFaceLocated){
                    field_decl->setFaceLocated(true);
                  }

                  GetClangASTContext().SetMetadataAsUserID (field_decl, MakeUserID(die->GetOffset()));

                  layout_info.field_offsets.insert(std::make_pair(field_decl, field_bit_offset));
                }
            }
        }
      }
  }

  return count;
}
