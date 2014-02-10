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

#include "DwarfUnit.h"
#include "DwarfAccelTable.h"
#include "DwarfDebug.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/DIBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

// The following functions were copied from LLVM DwarfUnit.cpp

/// If this type is derived from a base type then return base type size.
static uint64_t getBaseTypeSize(DwarfDebug *DD, DIDerivedType Ty) {
  unsigned Tag = Ty.getTag();

  if (Tag != dwarf::DW_TAG_member && Tag != dwarf::DW_TAG_typedef &&
      Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type)
    return Ty.getSizeInBits();

  DIType BaseType = DD->resolve(Ty.getTypeDerivedFrom());

  // If this type is not derived from any type then take conservative approach.
  if (!BaseType.isValid())
    return Ty.getSizeInBits();

  // If this is a derived type, go ahead and get the base type, unless it's a
  // reference then it's just the size of the field. Pointer types have no need
  // of this since they're a different type of qualification on the type.
  if (BaseType.getTag() == dwarf::DW_TAG_reference_type ||
      BaseType.getTag() == dwarf::DW_TAG_rvalue_reference_type)
    return Ty.getSizeInBits();

  if (BaseType.isDerivedType())
    return getBaseTypeSize(DD, DIDerivedType(BaseType));

  return BaseType.getSizeInBits();
}

void DwarfUnit::constructMeshMemberDIE(DIE &Buffer, DIScoutDerivedType DT) {
  // This method is modeled after constructMemberDIE - which is called for
  // constructing the DWARF info for the members of a struct/class but
  // is simpler because we do not have classes which have virtual members,
  // access/visibility, etc.

  DIE *MemberDie = createAndAddDIE(DT.getTag(), Buffer);
  StringRef Name = DT.getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, Name);

  addType(MemberDie, resolve(DT.getTypeDerivedFrom()));

  addSourceLine(MemberDie, DT);

  uint64_t OffsetInBytes = DT.getOffsetInBits() >> 3;

  if (DD->getDwarfVersion() <= 2) {
    DIEBlock *MemLocationDie = new (DIEValueAllocator) DIEBlock();
    addUInt(MemLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(MemLocationDie, dwarf::DW_FORM_udata, OffsetInBytes);
    addBlock(MemberDie, dwarf::DW_AT_data_member_location, MemLocationDie);
  } else
    addUInt(MemberDie, dwarf::DW_AT_data_member_location, None,
        OffsetInBytes);

  addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
      dwarf::DW_ACCESS_public);

  addUInt(MemberDie, dwarf::DW_AT_SCOUT_mesh_field_flags, None, DT.getScoutFlags());
}
