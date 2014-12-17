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

#include "llvm/IR/DIBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

/// createMeshMemberType - Create debugging information entry for a scout
/// mesh member field member.

// Note: the layout of the metadata must be kept in sync. with the
// layout of DIDerivedType as there may be new fields which are added
// to the DIDerivedType, and we need to maintain the ability that
// DIScoutDerivedType is a proper subclass of DIDerivedType

/// getNonCompileUnitScope - If N is compile unit return NULL otherwise return
/// N.

// -------------- The following functions were copied from LLVM
// DIBuilder.cpp and must be kept in sync in the event of changes.

static MDNode *getNonCompileUnitScope(MDNode *N) {
  if (DIDescriptor(N).isCompileUnit())
    return NULL;
  return N;
}

namespace {
  class HeaderBuilder {
    SmallVector<char, 256> Chars;
    
  public:
    explicit HeaderBuilder(Twine T) { T.toVector(Chars); }
    HeaderBuilder(const HeaderBuilder &X) : Chars(X.Chars) {}
    HeaderBuilder(HeaderBuilder &&X) : Chars(std::move(X.Chars)) {}
    
    template <class Twineable> HeaderBuilder &concat(Twineable &&X) {
      Chars.push_back(0);
      Twine(X).toVector(Chars);
      return *this;
    }
    
    MDString *get(LLVMContext &Context) const {
      return MDString::get(Context, StringRef(Chars.begin(), Chars.size()));
    }
    
    static HeaderBuilder get(unsigned Tag) {
      return HeaderBuilder("0x" + Twine::utohexstr(Tag));
    }
  };
}

// ----------------------------------------------------

DICompositeType DIBuilder::createUniformMeshType(DIDescriptor Context,
    StringRef Name, DIFile File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags, DIType DerivedFrom,
    DIArray Elements,
    unsigned dimX,
    unsigned dimY,
    unsigned dimZ,
    unsigned RunTimeLang,
    DIType VTableHolder,
    StringRef UniqueIdentifier
) {
  // TAG_structure_type is encoded in DICompositeType format.
  Metadata *Elts[] = {
    HeaderBuilder::get(dwarf::DW_TAG_SCOUT_uniform_mesh_type)
    .concat(Name)
    .concat(LineNumber)
    .concat(SizeInBits)
    .concat(AlignInBits)
    .concat(0)
    .concat(Flags)
    .concat(RunTimeLang)
    .get(VMContext),
    File.getFileNode(), DIScope(getNonCompileUnitScope(Context)).getRef(),
    DerivedFrom.getRef(), Elements, VTableHolder.getRef(), nullptr,
    UniqueIdentifier.empty() ? nullptr
    : MDString::get(VMContext, UniqueIdentifier),
    // These are the Scout-specific fields, we need to keep an eye on this when we merge
    // with LLVM/Clang in case new fields are added to DICompositeType, we need to add them
    // here and update the accessor methods on DIScoutCompositeType to reflect layout changes
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimX)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimY)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimZ))
  };
  DIScoutCompositeType R(MDNode::get(VMContext, Elts));
  assert(R.isCompositeType() &&
         "createUniformMeshType should return a DICompositeType");
  if (!UniqueIdentifier.empty()){
    retainType(R);
  }
  return R;
}

DICompositeType DIBuilder::createStructuredMeshType(DIDescriptor Context,
    StringRef Name, DIFile File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags, DIType DerivedFrom,
    DIArray Elements,
    unsigned dimX,
    unsigned dimY,
    unsigned dimZ,
    unsigned RunTimeLang,
    DIType VTableHolder,
    StringRef UniqueIdentifier) {
  // TAG_structure_type is encoded in DICompositeType format.
  Metadata *Elts[] = {
    HeaderBuilder::get(dwarf::DW_TAG_SCOUT_uniform_mesh_type)
    .concat(Name)
    .concat(LineNumber)
    .concat(SizeInBits)
    .concat(AlignInBits)
    .concat(0)
    .concat(Flags)
    .concat(RunTimeLang)
    .get(VMContext),
    File.getFileNode(), DIScope(getNonCompileUnitScope(Context)).getRef(),
    DerivedFrom.getRef(), Elements, VTableHolder.getRef(), nullptr,
    UniqueIdentifier.empty() ? nullptr
    : MDString::get(VMContext, UniqueIdentifier),
    // These are the Scout-specific fields, we need to keep an eye on this when we merge
    // with LLVM/Clang in case new fields are added to DICompositeType, we need to add them
    // here and update the accessor methods on DIScoutCompositeType to reflect layout changes
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimX)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimY)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimZ))
  };
  DIScoutCompositeType R(MDNode::get(VMContext, Elts));
  assert(R.isCompositeType() &&
         "createStructuredMeshType should return a DICompositeType");
  if (!UniqueIdentifier.empty()){
    retainType(R);
  }
  return R;
}

DICompositeType DIBuilder::createRectilinearMeshType(DIDescriptor Context,
    StringRef Name, DIFile File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags, DIType DerivedFrom,
    DIArray Elements,
    unsigned dimX,
    unsigned dimY,
    unsigned dimZ,
    unsigned RunTimeLang,
    DIType VTableHolder,
    StringRef UniqueIdentifier) {
 // TAG_structure_type is encoded in DICompositeType format.
  // TAG_structure_type is encoded in DICompositeType format.
  Metadata *Elts[] = {
    HeaderBuilder::get(dwarf::DW_TAG_SCOUT_uniform_mesh_type)
    .concat(Name)
    .concat(LineNumber)
    .concat(SizeInBits)
    .concat(AlignInBits)
    .concat(0)
    .concat(Flags)
    .concat(RunTimeLang)
    .get(VMContext),
    File.getFileNode(), DIScope(getNonCompileUnitScope(Context)).getRef(),
    DerivedFrom.getRef(), Elements, VTableHolder.getRef(), nullptr,
    UniqueIdentifier.empty() ? nullptr
    : MDString::get(VMContext, UniqueIdentifier),
    // These are the Scout-specific fields, we need to keep an eye on this when we merge
    // with LLVM/Clang in case new fields are added to DICompositeType, we need to add them
    // here and update the accessor methods on DIScoutCompositeType to reflect layout changes
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimX)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimY)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimZ))
  };
  DIScoutCompositeType R(MDNode::get(VMContext, Elts));
  assert(R.isCompositeType() &&
         "createRectilinearMeshType should return a DICompositeType");
  if (!UniqueIdentifier.empty()){
    retainType(R);
  }
  return R;
}

DICompositeType DIBuilder::createUnstructuredMeshType(DIDescriptor Context,
    StringRef Name, DIFile File,
    unsigned LineNumber,
    uint64_t SizeInBits,
    uint64_t AlignInBits,
    unsigned Flags, DIType DerivedFrom,
    DIArray Elements,
    unsigned dimX,
    unsigned dimY,
    unsigned dimZ,
    unsigned RunTimeLang,
    DIType VTableHolder,
    StringRef UniqueIdentifier) {
  // TAG_structure_type is encoded in DICompositeType format.
  Metadata *Elts[] = {
    HeaderBuilder::get(dwarf::DW_TAG_SCOUT_uniform_mesh_type)
    .concat(Name)
    .concat(LineNumber)
    .concat(SizeInBits)
    .concat(AlignInBits)
    .concat(0)
    .concat(Flags)
    .concat(RunTimeLang)
    .get(VMContext),
    File.getFileNode(), DIScope(getNonCompileUnitScope(Context)).getRef(),
    DerivedFrom.getRef(), Elements, VTableHolder.getRef(), nullptr,
    UniqueIdentifier.empty() ? nullptr
    : MDString::get(VMContext, UniqueIdentifier),
    // These are the Scout-specific fields, we need to keep an eye on this when we merge
    // with LLVM/Clang in case new fields are added to DICompositeType, we need to add them
    // here and update the accessor methods on DIScoutCompositeType to reflect layout changes
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimX)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimY)),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext), dimZ))
  };
  DIScoutCompositeType R(MDNode::get(VMContext, Elts));
  assert(R.isCompositeType() &&
         "createUnstructuredMeshType should return a DICompositeType");
  if (!UniqueIdentifier.empty()){
    retainType(R);
  }
  return R;
}

DIScoutDerivedType 
DIBuilder::createMeshMemberType(DIDescriptor Scope, StringRef Name,
                                DIFile File, unsigned LineNumber,
                                uint64_t SizeInBits,
                                uint64_t AlignInBits,
                                uint64_t OffsetInBits,
                                unsigned Flags,
                                unsigned ScoutFlags,
                                DIType Ty) {
  // TAG_member is encoded in DIScoutDerivedType format.
  Metadata *Elts[] = {HeaderBuilder::get(dwarf::DW_TAG_member)
    .concat(Name)
    .concat(LineNumber)
    .concat(SizeInBits)
    .concat(AlignInBits)
    .concat(OffsetInBits)
    .concat(Flags)
    .get(VMContext),
    File.getFileNode(),
    DIScope(getNonCompileUnitScope(Scope)).getRef(),
    Ty.getRef(),
    ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(VMContext),
                                             ScoutFlags))};
  return DIScoutDerivedType(MDNode::get(VMContext, Elts));
}
