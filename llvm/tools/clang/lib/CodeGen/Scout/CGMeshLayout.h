//===--- CGMeshLayout.h - LLVM Record Layout Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGMESHLAYOUT_H
#define CLANG_CODEGEN_CGMESHLAYOUT_H

#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Scout/MeshDecl.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DerivedTypes.h"

namespace llvm {
  class StructType;
}

namespace clang {
namespace CodeGen {



/// CGMeshLayout - This class handles mesh layout info while lowering AST types 
/// to LLVM types.
///
/// These layout objects are only created on demand as IR generation requires.
class CGRMeshLayout {
  friend class CodeGenTypes;

  CGMeshLayout(const CGMeshLayout &) LLVM_DELETED_FUNCTION;
  void operator=(const CGMeshLayout &) LLVM_DELETED_FUNCTION;

private:
  /// The LLVM type corresponding to this record layout; used when
  /// laying it out as a complete object.
  llvm::StructType *CompleteObjectType;

  /// The LLVM type for the non-virtual part of this record layout;
  /// used when laying it out as a base subobject.
  llvm::StructType *BaseSubobjectType;

  /// Map from (non-bit-field) struct field to the corresponding llvm struct
  /// type field no. This info is populated by record builder.
  llvm::DenseMap<const MeshFieldDecl *, unsigned> FieldInfo;

  /// Map from (bit-field) struct field to the corresponding llvm struct type
  /// field no. This info is populated by record builder.
  llvm::DenseMap<const MeshFieldDecl *, CGBitFieldInfo> BitFields;

  // FIXME: Maybe we could use a CXXBaseSpecifier as the key and use a single
  // map for both virtual and non virtual bases.
  llvm::DenseMap<const CXXRecordDecl *, unsigned> NonVirtualBases;

  /// Map from virtual bases to their field index in the complete object.
  llvm::DenseMap<const CXXRecordDecl *, unsigned> CompleteObjectVirtualBases;

  /// False if any direct or indirect subobject of this class, when
  /// considered as a complete object, requires a non-zero bitpattern
  /// when zero-initialized.
  bool IsZeroInitializable : 1;

  /// False if any direct or indirect subobject of this class, when
  /// considered as a base subobject, requires a non-zero bitpattern
  /// when zero-initialized.
  bool IsZeroInitializableAsBase : 1;

public:
  CGMeshLayout(llvm::StructType *CompleteObjectType,
                 llvm::StructType *BaseSubobjectType,
                 bool IsZeroInitializable,
                 bool IsZeroInitializableAsBase)
    : CompleteObjectType(CompleteObjectType),
      BaseSubobjectType(BaseSubobjectType),
      IsZeroInitializable(IsZeroInitializable),
      IsZeroInitializableAsBase(IsZeroInitializableAsBase) {}

  /// \brief Return the "complete object" LLVM type associated with
  /// this record.
  llvm::StructType *getLLVMType() const {
    return CompleteObjectType;
  }

  /// \brief Return the "base subobject" LLVM type associated with
  /// this record.
  llvm::StructType *getBaseSubobjectLLVMType() const {
    return BaseSubobjectType;
  }

  /// \brief Check whether this struct can be C++ zero-initialized
  /// with a zeroinitializer.
  bool isZeroInitializable() const {
    return IsZeroInitializable;
  }

  /// \brief Check whether this struct can be C++ zero-initialized
  /// with a zeroinitializer when considered as a base subobject.
  bool isZeroInitializableAsBase() const {
    return IsZeroInitializableAsBase;
  }

  /// \brief Return llvm::StructType element number that corresponds to the
  /// field FD.
  unsigned getLLVMFieldNo(const FieldDecl *FD) const {
    assert(FieldInfo.count(FD) && "Invalid field for record!");
    return FieldInfo.lookup(FD);
  }

  unsigned getNonVirtualBaseLLVMFieldNo(const CXXRecordDecl *RD) const {
    assert(NonVirtualBases.count(RD) && "Invalid non-virtual base!");
    return NonVirtualBases.lookup(RD);
  }

  /// \brief Return the LLVM field index corresponding to the given
  /// virtual base.  Only valid when operating on the complete object.
  unsigned getVirtualBaseIndex(const CXXRecordDecl *base) const {
    assert(CompleteObjectVirtualBases.count(base) && "Invalid virtual base!");
    return CompleteObjectVirtualBases.lookup(base);
  }

  /// \brief Return the BitFieldInfo that corresponds to the field FD.
  const CGBitFieldInfo &getBitFieldInfo(const FieldDecl *FD) const {
    assert(FD->isBitField() && "Invalid call for non bit-field decl!");
    llvm::DenseMap<const FieldDecl *, CGBitFieldInfo>::const_iterator
      it = BitFields.find(FD);
    assert(it != BitFields.end() && "Unable to find bitfield info");
    return it->second;
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
