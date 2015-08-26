//===--- CGMeshLayoutBuilder.cpp - CGMeshLayout builder  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builder implementation for CGMeshLayout objects.
//
//===----------------------------------------------------------------------===//

#include "../CGRecordLayout.h"
#include "CGMeshLayout.h"
#include "CGCXXABI.h"
#include "CodeGenTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Scout/MeshLayout.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace CodeGen;

namespace {

  class CGMeshLayoutBuilder {
   public:
    /// MeshFieldTypes - Holds the LLVM types that the mesh is created from.
    ///
    SmallVector<llvm::Type *, 16> MeshFieldTypes;

    /// FieldInfo - Holds a field and its corresponding LLVM field number.
    llvm::DenseMap<const MeshFieldDecl *, unsigned> Fields;

    /// BitFieldInfo - Holds location and size information about a bit field.
    llvm::DenseMap<const MeshFieldDecl *, CGBitFieldInfo> BitFields;

    /// Packed - Whether the resulting LLVM struct will be packed or not.
    bool Packed;

   private:
    CodeGenTypes &Types;

    /// LastLaidOutBaseInfo - Contains the offset and non-virtual size of the
    /// last base laid out. Used so that we can replace the last laid out base
    /// type with an i8 array if needed.
    struct LastLaidOutBaseInfo {
      CharUnits Offset;
      CharUnits NonVirtualSize;

      bool isValid() const { return !NonVirtualSize.isZero(); }
      void invalidate() { NonVirtualSize = CharUnits::Zero(); }

    } LastLaidOutBase;

    /// Alignment - Contains the alignment of the MeshDecl.
    CharUnits Alignment;

    /// NextFieldOffset - Holds the next field offset.
    CharUnits NextFieldOffset;

    /// LayoutField - try to layout all fields in the record decl.
    /// Returns false if the operation failed because the struct is not packed.
    bool LayoutFields(const MeshDecl *D);

    /// LayoutField - layout a single field. Returns false if the operation failed
    /// because the current struct is not packed.
    bool LayoutField(const MeshFieldDecl *D, uint64_t FieldOffset);

    /// LayoutBitField - layout a single bit field.
    void LayoutBitField(const MeshFieldDecl *D, uint64_t FieldOffset);

    /// AppendField - Appends a field with the given offset and type.
    void AppendField(CharUnits fieldOffset, llvm::Type *FieldTy);

    /// AppendPadding - Appends enough padding bytes so that the total
    /// struct size is a multiple of the field alignment.
    void AppendPadding(CharUnits fieldOffset, CharUnits fieldAlignment);

    /// ResizeLastBaseFieldIfNecessary - Fields and bases can be laid out in the
    /// tail padding of a previous base.  If this happens, the type of the previous
    /// base needs to be changed to an array of i8.  Returns true if the last
    /// laid out base was resized.
    bool ResizeLastBaseFieldIfNecessary(CharUnits offset);

    /// elements.
    llvm::Type *getByteArrayType(CharUnits NumBytes);

    /// AppendBytes - Append a given number of bytes to the record.
    void AppendBytes(CharUnits numBytes);

    /// AppendTailPadding - Append enough tail padding so that the type will have
    /// the passed size.
    void AppendTailPadding(CharUnits RecordSize);

    CharUnits getTypeAlignment(llvm::Type *Ty) const;

    /// getAlignmentAsLLVMStruct - Returns the maximum alignment of all the
    /// LLVM element types.
    CharUnits getAlignmentAsLLVMStruct() const;

   public:
    CGMeshLayoutBuilder(CodeGenTypes &Types)
        : Packed(false), Types(Types) { }

    /// Layout - Will layout a MeshDecl.
    void Layout(const MeshDecl *D);
  };

}

void CGMeshLayoutBuilder::Layout(const MeshDecl *D) {
  Alignment = Types.getContext().getASTMeshLayout(D).getAlignment();
  Packed = D->hasAttr<PackedAttr>();

  if (LayoutFields(D))
    return;

  // We weren't able to layout the struct. Try again with a packed struct
  Packed = true;
  NextFieldOffset = CharUnits::Zero();
  MeshFieldTypes.clear();
  Fields.clear();
  BitFields.clear();
  LayoutFields(D);
}

bool CGMeshLayoutBuilder::LayoutField(const MeshFieldDecl *D,
                                        uint64_t fieldOffset) {
  // If the field is packed, then we need a packed struct.
  if (!Packed && D->hasAttr<PackedAttr>())
    return false;

  assert(!D->isBitField() && "Bitfields should be laid out seperately.");

  assert(fieldOffset % Types.getTarget().getCharWidth() == 0
         && "field offset is not on a byte boundary!");
  CharUnits fieldOffsetInBytes
    = Types.getContext().toCharUnitsFromBits(fieldOffset);

  llvm::Type *Ty = Types.ConvertTypeForMem(D->getType());
  CharUnits typeAlignment = getTypeAlignment(Ty);

  // If the type alignment is larger then the struct alignment, we must use
  // a packed struct.
  if (typeAlignment > Alignment) {
    assert(!Packed && "Alignment is wrong even with packed struct!");
    return false;
  }

  if (!Packed) {
    if (const MeshType *RT = D->getType()->getAs<MeshType>()) {
      const MeshDecl *RD = cast<MeshDecl>(RT->getDecl());
      if (const MaxFieldAlignmentAttr *MFAA =
            RD->getAttr<MaxFieldAlignmentAttr>()) {
        if (MFAA->getAlignment() != Types.getContext().toBits(typeAlignment))
          return false;
      }
    }
  }

  // Round up the field offset to the alignment of the field type.
  CharUnits alignedNextFieldOffsetInBytes =
    NextFieldOffset.RoundUpToAlignment(typeAlignment);

  if (fieldOffsetInBytes < alignedNextFieldOffsetInBytes) {
    assert(!Packed && "Could not place field even with packed struct!");
    return false;
  }

  AppendPadding(fieldOffsetInBytes, typeAlignment);

  // Now append the field.
  Fields[D] = MeshFieldTypes.size();
  AppendField(fieldOffsetInBytes, Ty);

  return true;
}

bool CGMeshLayoutBuilder::LayoutFields(const MeshDecl *D) {
  assert(!Alignment.isZero() && "Did not set alignment!");

  const ASTMeshLayout &Layout = Types.getContext().getASTMeshLayout(D);

  unsigned FieldNo = 0;
  //const MeshFieldDecl *LastFD = 0;

  for (MeshDecl::field_iterator FI = D->field_begin(), FE = D->field_end();
       FI != FE; ++FI, ++FieldNo) {
    MeshFieldDecl *FD = *FI;

    if (!LayoutField(FD, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed &&
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

  // Append tail padding if necessary.
  AppendTailPadding(Layout.getSize());

  return true;
}

void CGMeshLayoutBuilder::AppendTailPadding(CharUnits RecordSize) {
  assert(NextFieldOffset <= RecordSize && "Size mismatch!");

  CharUnits AlignedNextFieldOffset =
    NextFieldOffset.RoundUpToAlignment(getAlignmentAsLLVMStruct());

  if (AlignedNextFieldOffset == RecordSize) {
    // We don't need any padding.
    return;
  }

  CharUnits NumPadBytes = RecordSize - NextFieldOffset;
  AppendBytes(NumPadBytes);
}

void CGMeshLayoutBuilder::AppendField(CharUnits fieldOffset,
                                        llvm::Type *fieldType) {
  CharUnits fieldSize =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(fieldType));

  MeshFieldTypes.push_back(fieldType);

  NextFieldOffset = fieldOffset + fieldSize;
}

void CGMeshLayoutBuilder::AppendPadding(CharUnits fieldOffset,
                                          CharUnits fieldAlignment) {
  assert(NextFieldOffset <= fieldOffset &&
         "Incorrect field layout!");

  // Do nothing if we're already at the right offset.
  if (fieldOffset == NextFieldOffset) return;

  // If we're not emitting a packed LLVM type, try to avoid adding
  // unnecessary padding fields.
  if (!Packed) {
    // Round up the field offset to the alignment of the field type.
    CharUnits alignedNextFieldOffset =
      NextFieldOffset.RoundUpToAlignment(fieldAlignment);
    assert(alignedNextFieldOffset <= fieldOffset);

    // If that's the right offset, we're done.
    if (alignedNextFieldOffset == fieldOffset) return;
  }

  // Otherwise we need explicit padding.
  CharUnits padding = fieldOffset - NextFieldOffset;
  AppendBytes(padding);
}

bool CGMeshLayoutBuilder::ResizeLastBaseFieldIfNecessary(CharUnits offset) {

  // Check if we have a base to resize.
  if (!LastLaidOutBase.isValid())
    return false;

  // This offset does not overlap with the tail padding.
  if (offset >= NextFieldOffset)
    return false;

  // Restore the field offset and append an i8 array instead.
  MeshFieldTypes.pop_back();
  NextFieldOffset = LastLaidOutBase.Offset;
  AppendBytes(LastLaidOutBase.NonVirtualSize);
  LastLaidOutBase.invalidate();
  
  return true;
}

llvm::Type *CGMeshLayoutBuilder::getByteArrayType(CharUnits numBytes) {
  assert(!numBytes.isZero() && "Empty byte arrays aren't allowed.");

  llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (numBytes > CharUnits::One())
    Ty = llvm::ArrayType::get(Ty, numBytes.getQuantity());

  return Ty;
}

void CGMeshLayoutBuilder::AppendBytes(CharUnits numBytes) {
  if (numBytes.isZero())
    return;

  // Append the padding field
  AppendField(NextFieldOffset, getByteArrayType(numBytes));
}

CharUnits CGMeshLayoutBuilder::getTypeAlignment(llvm::Type *Ty) const {
  if (Packed)
    return CharUnits::One();

  return CharUnits::fromQuantity(Types.getDataLayout().getABITypeAlignment(Ty));
}

CharUnits CGMeshLayoutBuilder::getAlignmentAsLLVMStruct() const {
  if (Packed)
    return CharUnits::One();

  CharUnits maxAlignment = CharUnits::One();
  for (size_t i = 0; i != MeshFieldTypes.size(); ++i)
    maxAlignment = std::max(maxAlignment, getTypeAlignment(MeshFieldTypes[i]));

  return maxAlignment;
}

CGMeshLayout *CodeGenTypes::ComputeMeshLayout(const MeshDecl *D,
                                              llvm::StructType *Ty) {
  CGMeshLayoutBuilder Builder(*this);

  Builder.Layout(D);

  Ty->setBody(Builder.MeshFieldTypes, Builder.Packed);

  CGMeshLayout *ML;
  ML = new CGMeshLayout(Ty);

  // Add all the field numbers.
  ML->MeshFieldInfo.swap(Builder.Fields);
  // Add bitfield info.
  ML->BitFields.swap(Builder.BitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpMeshLayouts) {
    llvm::errs() << "\n*** Dumping IRgen Mesh Layout\n";
    llvm::errs() << "Mesh: ";
    D->dump();
    llvm::errs() << "\nLayout: ";
    ML->dump();
  }

#ifndef NDEBUG
  // Verify that the computed LLVM struct size matches the AST layout size.
  const ASTMeshLayout &Layout = getContext().getASTMeshLayout(D);

  uint64_t TypeSizeInBits = getContext().toBits(Layout.getSize());
  assert(TypeSizeInBits == getDataLayout().getTypeAllocSizeInBits(Ty) &&
         "Type size mismatch!");

    // Verify that the LLVM and AST field offsets agree.
  llvm::StructType *ST =
    dyn_cast<llvm::StructType>(ML->getLLVMType());
  const llvm::StructLayout *SL = getDataLayout().getStructLayout(ST);

  const ASTMeshLayout &AST_ML = getContext().getASTMeshLayout(D);
  MeshDecl::field_iterator it = D->field_begin();
  for (unsigned i = 0, e = AST_ML.getFieldCount(); i != e; ++i, ++it) {
    const MeshFieldDecl *FD = *it;

    // For non-bit-fields, just check that the LLVM struct offset matches the
    // AST offset.
    if (!FD->isBitField()) {
      unsigned FieldNo = ML->getLLVMFieldNo(FD);
      assert(AST_ML.getFieldOffset(i) == SL->getElementOffsetInBits(FieldNo) &&
             "Invalid field offset!");
      continue;
    }

    // Ignore unnamed bit-fields.
    if (!FD->getDeclName()) {
      continue;
    }

    // Don't inspect zero-length bitfields.
    if (FD->getBitWidthValue(getContext()) == 0)
      continue;

    const CGBitFieldInfo &Info = ML->getBitFieldInfo(FD);
    llvm::Type *ElementTy = ST->getTypeAtIndex(ML->getLLVMFieldNo(FD));

    assert(Info.StorageSize ==
           getDataLayout().getTypeAllocSizeInBits(ElementTy) &&
           "Storage size does not match the element type size");
    assert(Info.Size > 0 && "Empty bitfield!");
    assert(static_cast<unsigned>(Info.Offset) + Info.Size <= Info.StorageSize &&
           "Bitfield outside of its allocated storage");
  }
#endif

  return ML;
}

void CGMeshLayout::print(raw_ostream &OS) const {
  OS << "<CGMeshLayout\n";
  OS << "  LLVMType:" << *CompleteObjectType << "\n";
  OS << "  BitFields:[\n";

  // Print bit-field infos in declaration order.
  std::vector<std::pair<unsigned, const CGBitFieldInfo*> > BFIs;
  for (llvm::DenseMap<const MeshFieldDecl*, CGBitFieldInfo>::const_iterator
         it = BitFields.begin(), ie = BitFields.end();
       it != ie; ++it) {
    const MeshDecl *MD = it->first->getParent();
    unsigned Index = 0;
    for (MeshDecl::field_iterator
           it2 = MD->field_begin(); *it2 != it->first; ++it2)
      ++Index;
    BFIs.push_back(std::make_pair(Index, &it->second));
  }
  llvm::array_pod_sort(BFIs.begin(), BFIs.end());
  for (unsigned i = 0, e = BFIs.size(); i != e; ++i) {
    OS.indent(4);
    BFIs[i].second->print(OS);
    OS << "\n";
  }

  OS << "]>\n";
}

void CGMeshLayout::dump() const {
  print(llvm::errs());
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                                        const MeshFieldDecl *MFD,
                                        uint64_t Offset, uint64_t Size,
                                        uint64_t StorageSize,
                                        CharUnits StorageOffset) {
  llvm::Type *Ty = Types.ConvertTypeForMem(MFD->getType());
  CharUnits TypeSizeInBytes =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(Ty));
  uint64_t TypeSizeInBits = Types.getContext().toBits(TypeSizeInBytes);

  bool IsSigned = MFD->getType()->isSignedIntegerOrEnumerationType();

  if (Size > TypeSizeInBits) {
    // We have a wide bit-field. The extra bits are only used for padding, so
    // if we have a bitfield of type T, with size N:
    //
    // T t : N;
    //
    // We can just assume that it's:
    //
    // T t : sizeof(T);
    //
    Size = TypeSizeInBits;
  }

  // Reverse the bit offsets for big endian machines. Because we represent
  // a bitfield as a single large integer load, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (Types.getDataLayout().isBigEndian()) {
    Offset = StorageSize - (Offset + Size);
  }

  return CGBitFieldInfo(Offset, Size, IsSigned, StorageSize, StorageOffset);
}
