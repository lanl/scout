//=== MeshLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/scout/MeshLayout.h"

namespace {

class MeshLayoutBuilder {

protected:
  // FIXME: Remove this and make the appropriate fields public.
  friend class clang::ASTContext;
  const ASTContext &Context;

  EmptySubobjectMap *EmptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t Size;

  /// Alignment - The current alignment of the record layout.
  CharUnits Alignment;

  /// \brief The alignment if attribute packed is not used.
  CharUnits UnpackedAlignment;

  SmallVector<uint64_t, 16> FieldOffsets;

  /// \brief Whether the external AST source has provided a layout for this
  /// record.
  unsigned ExternalLayout : 1;

  /// \brief Whether we need to infer alignment, even when we have an 
  /// externally-provided layout.
  unsigned InferAlignment : 1;
  
  /// Packed - Whether the record is packed or not.
  unsigned Packed : 1;

  unsigned IsMac68kAlign : 1;
  
  /// UnfilledBitsInLastByte - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last byte that can be used for
  /// an adjacent bitfield if necessary.
  unsigned char UnfilledBitsInLastByte;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  CharUnits MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  MeshFieldDecl *ZeroLengthBitfield;

  /// \brief Externally-provided size.
  uint64_t ExternalSize;
  
  /// \brief Externally-provided alignment.
  uint64_t ExternalAlign;
  
  /// \brief Externally-provided field offsets.
  llvm::DenseMap<const MeshFieldDecl *, uint64_t> ExternalFieldOffsets;

  MeshLayoutBuilder(const ASTContext &Context,
                      EmptySubobjectMap *EmptySubobjects)
    : Context(Context), EmptySubobjects(EmptySubobjects), Size(0), 
      Alignment(CharUnits::One()), UnpackedAlignment(CharUnits::One()),
      ExternalLayout(false), InferAlignment(false), 
      Packed(false), IsMac68kAlign(false),
      UnfilledBitsInLastByte(0), MaxFieldAlignment(CharUnits::Zero()), 
      DataSize(0),  
      ZeroLengthBitfield(0) { }

  /// Reset this MeshLayoutBuilder to a fresh state, using the given
  /// alignment as the initial alignment.  This is used for the
  /// correct layout of vb-table pointers in MSVC.
  void resetWithTargetAlignment(CharUnits TargetAlignment) {
    const ASTContext &Context = this->Context;
    EmptySubobjectMap *EmptySubobjects = this->EmptySubobjects;
    this->~MeshLayoutBuilder();
    new (this) MeshLayoutBuilder(Context, EmptySubobjects);
    Alignment = UnpackedAlignment = TargetAlignment;
  }

  void Layout(const MeshDecl *D);
  void LayoutFields(const MeshDecl *D);
  void LayoutField(const MeshFieldDecl *D);
  void LayoutWideBitField(uint64_t FieldSize, uint64_t TypeSize,
                          bool FieldPacked, const MeshFieldDecl *D);
  void LayoutBitField(const MeshFieldDecl *D);

  /// InitializeLayout - Initialize record layout for the given mesh decl.
  void InitializeLayout(const Decl *D);

  /// FinishLayout - Finalize record layout. Adjust record size based on the
  /// alignment.
  void FinishLayout(const NamedDecl *D);

  void UpdateAlignment(CharUnits NewAlignment, CharUnits UnpackedNewAlignment);
  void UpdateAlignment(CharUnits NewAlignment) {
    UpdateAlignment(NewAlignment, NewAlignment);
  }

  /// \brief Retrieve the externally-supplied field offset for the given
  /// field.
  ///
  /// \param Field The field whose offset is being queried.
  /// \param ComputedOffset The offset that we've computed for this field.
  uint64_t updateExternalFieldOffset(const MeshFieldDecl *Field, 
                                     uint64_t ComputedOffset);
  
  void CheckFieldPadding(uint64_t Offset, uint64_t UnpaddedOffset,
                         uint64_t UnpackedOffset, unsigned UnpackedAlign,
                         bool isPacked, const MeshFieldDecl *D);

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  CharUnits getSize() const { 
    assert(Size % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(Size); 
  }
  uint64_t getSizeInBits() const { return Size; }

  void setSize(CharUnits NewSize) { Size = Context.toBits(NewSize); }
  void setSize(uint64_t NewSize) { Size = NewSize; }

  CharUnits getAligment() const { return Alignment; }

  CharUnits getDataSize() const { 
    assert(DataSize % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(DataSize); 
  }
  uint64_t getDataSizeInBits() const { return DataSize; }

  void setDataSize(CharUnits NewSize) { DataSize = Context.toBits(NewSize); }
  void setDataSize(uint64_t NewSize) { DataSize = NewSize; }

  MeshLayoutBuilder(const MeshLayoutBuilder &) LLVM_DELETED_FUNCTION;
  void operator=(const MeshLayoutBuilder &) LLVM_DELETED_FUNCTION;
};

} // end anonymous namespace


void MeshLayoutBuilder::InitializeLayout(const Decl *D) {
  Packed = D->hasAttr<PackedAttr>();  

  // Honor the default struct packing maximum alignment flag.
  if (unsigned DefaultMaxFieldAlignment = Context.getLangOpts().PackStruct) {
    MaxFieldAlignment = CharUnits::fromQuantity(DefaultMaxFieldAlignment);
  }

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (D->hasAttr<AlignMac68kAttr>()) {
    IsMac68kAlign = true;
    MaxFieldAlignment = CharUnits::fromQuantity(2);
    Alignment = CharUnits::fromQuantity(2);
  } else {
    if (const MaxFieldAlignmentAttr *MFAA = D->getAttr<MaxFieldAlignmentAttr>())
      MaxFieldAlignment = Context.toCharUnitsFromBits(MFAA->getAlignment());

    if (unsigned MaxAlign = D->getMaxAlignment())
      UpdateAlignment(Context.toCharUnitsFromBits(MaxAlign));
  }
  
  // If there is an external AST source, ask it for the various offsets.
  if (const MeshDecl *MD = dyn_cast<MeshDecl>(D))
    if (ExternalASTSource *External = Context.getExternalSource()) {
      ExternalLayout = External->layoutMeshType(MD, 
                                                ExternalSize,
                                                ExternalAlign,
                                                ExternalFieldOffsets);
      
      // Update based on external alignment.
      if (ExternalLayout) {
        if (ExternalAlign > 0) {
          Alignment = Context.toCharUnitsFromBits(ExternalAlign);
        } else {
          // The external source didn't have alignment information; infer it.
          InferAlignment = true;
        }
      }
    }
}

void MeshLayoutBuilder::Layout(const MeshDecl *D) {
  InitializeLayout(D);
  LayoutFields(D);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout(D);
}

void MeshLayoutBuilder::LayoutFields(const MeshDecl *D) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  ZeroLengthBitfield = 0;
  for (MeshDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field) {
   
    if (!Context.getTargetInfo().useBitFieldTypeAlignment() &&
         Context.getTargetInfo().useZeroLengthBitfieldAlignment()) {             
      if (Field->isBitField() && Field->getBitWidthValue(Context) == 0)
        ZeroLengthBitfield = *Field;
    }
    LayoutField(*Field);
  }
}

void MeshLayoutBuilder::LayoutWideBitField(uint64_t FieldSize,
                                           uint64_t TypeSize,
                                           bool FieldPacked,
                                           const MeshFieldDecl *D) {
  assert(Context.getLangOpts().CPlusPlus &&
         "Can only have wide bit-fields in C++!");

  // Itanium C++ ABI 2.4:
  //   If sizeof(T)*8 < n, let T' be the largest integral POD type with
  //   sizeof(T')*8 <= n.

  QualType IntegralPODTypes[] = {
    Context.UnsignedCharTy, Context.UnsignedShortTy, Context.UnsignedIntTy,
    Context.UnsignedLongTy, Context.UnsignedLongLongTy
  };

  QualType Type;
  for (unsigned I = 0, E = llvm::array_lengthof(IntegralPODTypes);
       I != E; ++I) {
    uint64_t Size = Context.getTypeSize(IntegralPODTypes[I]);

    if (Size > FieldSize)
      break;

    Type = IntegralPODTypes[I];
  }
  assert(!Type.isNull() && "Did not find a type!");

  CharUnits TypeAlign = Context.getTypeAlignInChars(Type);

  // We're not going to use any of the unfilled bits in the last byte.
  UnfilledBitsInLastByte = 0;

  uint64_t FieldOffset;
  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;

  // The bitfield is allocated starting at the next offset aligned 
  // appropriately for T', with length n bits.
  FieldOffset = llvm::RoundUpToAlignment(getDataSizeInBits(), 
                                         Context.toBits(TypeAlign));

  uint64_t NewSizeInBits = FieldOffset + FieldSize;
  setDataSize(llvm::RoundUpToAlignment(NewSizeInBits, 
                                       Context.getTargetInfo().getCharAlign()));
  UnfilledBitsInLastByte = getDataSizeInBits() - NewSizeInBits;

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  CheckFieldPadding(FieldOffset, UnpaddedFieldOffset, FieldOffset,
                    Context.toBits(TypeAlign), FieldPacked, D);

  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(TypeAlign);
}

void MeshLayoutBuilder::LayoutBitField(const MeshFieldDecl *D) {
  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;
  uint64_t FieldOffset = UnpaddedFieldOffset;
  uint64_t FieldSize = D->getBitWidthValue(Context);

  std::pair<uint64_t, unsigned> FieldInfo = Context.getTypeInfo(D->getType());
  uint64_t TypeSize = FieldInfo.first;
  unsigned FieldAlign = FieldInfo.second;
  
  // This check is needed for 'long long' in -m32 mode.
  if (ZeroLengthBitfield) {
    std::pair<uint64_t, unsigned> FieldInfo;
    // The alignment of a zero-length bitfield affects the alignment
    // of the next member.  The alignment is the max of the zero 
    // length bitfield's alignment and a target specific fixed value.
    unsigned ZeroLengthBitfieldBoundary =
      Context.getTargetInfo().getZeroLengthBitfieldBoundary();
    if (ZeroLengthBitfieldBoundary > FieldAlign)
      FieldAlign = ZeroLengthBitfieldBoundary;
  }

  if (FieldSize > TypeSize) {
    LayoutWideBitField(FieldSize, TypeSize, FieldPacked, D);
    return;
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  unsigned UnpackedFieldAlign = FieldAlign;
  uint64_t UnpackedFieldOffset = FieldOffset;
  if (!Context.getTargetInfo().useBitFieldTypeAlignment() && !ZeroLengthBitfield)
    UnpackedFieldAlign = 1;

  if (FieldPacked || 
      (!Context.getTargetInfo().useBitFieldTypeAlignment() && !ZeroLengthBitfield))
    FieldAlign = 1;
  FieldAlign = std::max(FieldAlign, D->getMaxAlignment());
  UnpackedFieldAlign = std::max(UnpackedFieldAlign, D->getMaxAlignment());

  // The maximum field alignment overrides the aligned attribute.
  if (!MaxFieldAlignment.isZero() && FieldSize != 0) {
    unsigned MaxFieldAlignmentInBits = Context.toBits(MaxFieldAlignment);
    FieldAlign = std::min(FieldAlign, MaxFieldAlignmentInBits);
    UnpackedFieldAlign = std::min(UnpackedFieldAlign, MaxFieldAlignmentInBits);
  }

  // Check if we need to add padding to give the field the correct alignment.
  if (FieldSize == 0 || 
      (MaxFieldAlignment.isZero() &&
       (FieldOffset & (FieldAlign-1)) + FieldSize > TypeSize))
    FieldOffset = llvm::RoundUpToAlignment(FieldOffset, FieldAlign);

  if (FieldSize == 0 ||
      (MaxFieldAlignment.isZero() &&
       (UnpackedFieldOffset & (UnpackedFieldAlign-1)) + FieldSize > TypeSize))
    UnpackedFieldOffset = llvm::RoundUpToAlignment(UnpackedFieldOffset,
                                                   UnpackedFieldAlign);

  // Padding members don't affect overall alignment, unless zero length bitfield
  // alignment is enabled.
  if (!D->getIdentifier() && !Context.getTargetInfo().useZeroLengthBitfieldAlignment())
    FieldAlign = UnpackedFieldAlign = 1;

  ZeroLengthBitfield = 0;

  if (ExternalLayout)
    FieldOffset = updateExternalFieldOffset(D, FieldOffset);

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  if (!ExternalLayout)
    CheckFieldPadding(FieldOffset, UnpaddedFieldOffset, UnpackedFieldOffset,
                      UnpackedFieldAlign, FieldPacked, D);

  // Update DataSize to include the last byte containing (part of) the bitfield.
  uint64_t NewSizeInBits = FieldOffset + FieldSize;

  setDataSize(llvm::RoundUpToAlignment(NewSizeInBits, 
                                       Context.getTargetInfo().getCharAlign()));
  UnfilledBitsInLastByte = getDataSizeInBits() - NewSizeInBits;
  
  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(Context.toCharUnitsFromBits(FieldAlign), 
                  Context.toCharUnitsFromBits(UnpackedFieldAlign));
}

void MeshLayoutBuilder::LayoutField(const MeshFieldDecl *D) {  
  if (D->isBitField()) {
    LayoutBitField(D);
    return;
  }

  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;

  // Reset the unfilled bits.
  UnfilledBitsInLastByte = 0;

  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  CharUnits FieldOffset = getDataSize();
  CharUnits FieldSize;
  CharUnits FieldAlign;

  if (D->getType()->isIncompleteArrayType()) {
    // This is a flexible array member; we can't directly
    // query getTypeInfo about these, so we figure it out here.
    // Flexible array members don't have any size, but they
    // have to be aligned appropriately for their element type.
    FieldSize = CharUnits::Zero();
    const ArrayType* ATy = Context.getAsArrayType(D->getType());
    FieldAlign = Context.getTypeAlignInChars(ATy->getElementType());
  } else if (const ReferenceType *RT = D->getType()->getAs<ReferenceType>()) {
    unsigned AS = RT->getPointeeType().getAddressSpace();
    FieldSize = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(AS));
    FieldAlign = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerAlign(AS));
  } else {
    std::pair<CharUnits, CharUnits> FieldInfo = 
      Context.getTypeInfoInChars(D->getType());
    FieldSize = FieldInfo.first;
    FieldAlign = FieldInfo.second;

    if (ZeroLengthBitfield) {
      CharUnits ZeroLengthBitfieldBoundary = 
        Context.toCharUnitsFromBits(
          Context.getTargetInfo().getZeroLengthBitfieldBoundary());
      if (ZeroLengthBitfieldBoundary == CharUnits::Zero()) {
        // If a zero-length bitfield is inserted after a bitfield,
        // and the alignment of the zero-length bitfield is
        // greater than the member that follows it, `bar', `bar' 
        // will be aligned as the type of the zero-length bitfield.
        std::pair<CharUnits, CharUnits> FieldInfo = 
          Context.getTypeInfoInChars(ZeroLengthBitfield->getType());
        CharUnits ZeroLengthBitfieldAlignment = FieldInfo.second;        
        if (ZeroLengthBitfieldAlignment > FieldAlign)
          FieldAlign = ZeroLengthBitfieldAlignment;
      } else if (ZeroLengthBitfieldBoundary > FieldAlign) {
        // Align 'bar' based on a fixed alignment specified by the target.
        assert(Context.getTargetInfo().useZeroLengthBitfieldAlignment() &&
               "ZeroLengthBitfieldBoundary should only be used in conjunction"
               " with useZeroLengthBitfieldAlignment.");
        FieldAlign = ZeroLengthBitfieldBoundary;
      }
      ZeroLengthBitfield = 0;
    }
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  CharUnits UnpackedFieldAlign = FieldAlign;
  CharUnits UnpackedFieldOffset = FieldOffset;

  if (FieldPacked)
    FieldAlign = CharUnits::One();
  CharUnits MaxAlignmentInChars = 
    Context.toCharUnitsFromBits(D->getMaxAlignment());
  FieldAlign = std::max(FieldAlign, MaxAlignmentInChars);
  UnpackedFieldAlign = std::max(UnpackedFieldAlign, MaxAlignmentInChars);

  // The maximum field alignment overrides the aligned attribute.
  if (!MaxFieldAlignment.isZero()) {
    FieldAlign = std::min(FieldAlign, MaxFieldAlignment);
    UnpackedFieldAlign = std::min(UnpackedFieldAlign, MaxFieldAlignment);
  }

  // Round up the current record size to the field's alignment boundary.
  FieldOffset = FieldOffset.RoundUpToAlignment(FieldAlign);
  UnpackedFieldOffset = 
    UnpackedFieldOffset.RoundUpToAlignment(UnpackedFieldAlign);

  if (ExternalLayout) {
    FieldOffset = Context.toCharUnitsFromBits(
                    updateExternalFieldOffset(D, Context.toBits(FieldOffset)));
    
    if (EmptySubobjects) {
      // Record the fact that we're placing a field at this offset.
      bool Allowed = EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset);
      (void)Allowed;
      assert(Allowed && "Externally-placed field cannot be placed here");      
    }
  } else {
    if (EmptySubobjects) {
      // Check if we can place the field at this offset.
      while (!EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset)) {
        // We couldn't place the field at the offset. Try again at a new offset.
        FieldOffset += FieldAlign;
      }
    }
  }
  
  // Place this field at the current location.
  FieldOffsets.push_back(Context.toBits(FieldOffset));

  if (!ExternalLayout)
    CheckFieldPadding(Context.toBits(FieldOffset), UnpaddedFieldOffset, 
                      Context.toBits(UnpackedFieldOffset),
                      Context.toBits(UnpackedFieldAlign), FieldPacked, D);

  // Reserve space for this field.
  setDataSize(FieldOffset + FieldSize);

  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(FieldAlign, UnpackedFieldAlign);
}

void MeshLayoutBuilder::FinishLayout(const NamedDecl *D) {
  // In C++, records cannot be of size 0.
  if (Context.getLangOpts().CPlusPlus && getSizeInBits() == 0) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
      // Compatibility with gcc requires a class (pod or non-pod)
      // which is not empty but of size 0; such as having fields of
      // array of zero-length, remains of Size 0
      if (RD->isEmpty())
        setSize(CharUnits::One());
    }
    else
      setSize(CharUnits::One());
  }

  // Finally, round the size of the record up to the alignment of the
  // record itself.
  uint64_t UnpaddedSize = getSizeInBits() - UnfilledBitsInLastByte;
  uint64_t UnpackedSizeInBits =
  llvm::RoundUpToAlignment(getSizeInBits(),
                           Context.toBits(UnpackedAlignment));
  CharUnits UnpackedSize = Context.toCharUnitsFromBits(UnpackedSizeInBits);
  uint64_t RoundedSize
    = llvm::RoundUpToAlignment(getSizeInBits(), Context.toBits(Alignment));

  if (ExternalLayout) {
    // If we're inferring alignment, and the external size is smaller than
    // our size after we've rounded up to alignment, conservatively set the
    // alignment to 1.
    if (InferAlignment && ExternalSize < RoundedSize) {
      Alignment = CharUnits::One();
      InferAlignment = false;
    }
    setSize(ExternalSize);
    return;
  }

  // Set the size to the final size.
  setSize(RoundedSize);

  unsigned CharBitNum = Context.getTargetInfo().getCharWidth();
  if (const MeshDecl *RD = dyn_cast<MeshDecl>(D)) {
    // Warn if padding was introduced to the mesh.
    if (getSizeInBits() > UnpaddedSize) {
      unsigned PadSize = getSizeInBits() - UnpaddedSize;
      bool InBits = true;
      if (PadSize % CharBitNum == 0) {
        PadSize = PadSize / CharBitNum;
        InBits = false;
      }
      Diag(RD->getLocation(), diag::warn_padded_struct_size)
          << Context.getTypeDeclType(RD)
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1); // plural or not
    }

    // Warn if we packed it unnecessarily. If the alignment is 1 byte don't
    // bother since there won't be alignment issues.
    if (Packed && UnpackedAlignment > CharUnits::One() && 
        getSize() == UnpackedSize)
      Diag(D->getLocation(), diag::warn_unnecessary_packed)
          << Context.getTypeDeclType(RD);
  }
}

void MeshLayoutBuilder::UpdateAlignment(CharUnits NewAlignment,
                                          CharUnits UnpackedNewAlignment) {
  // The alignment is not modified when using 'mac68k' alignment or when
  // we have an externally-supplied layout that also provides overall alignment.
  if (IsMac68kAlign || (ExternalLayout && !InferAlignment))
    return;

  if (NewAlignment > Alignment) {
    assert(llvm::isPowerOf2_32(NewAlignment.getQuantity() && 
           "Alignment not a power of 2"));
    Alignment = NewAlignment;
  }

  if (UnpackedNewAlignment > UnpackedAlignment) {
    assert(llvm::isPowerOf2_32(UnpackedNewAlignment.getQuantity() &&
           "Alignment not a power of 2"));
    UnpackedAlignment = UnpackedNewAlignment;
  }
}

uint64_t
MeshLayoutBuilder::updateExternalFieldOffset(const MeshFieldDecl *Field, 
                                             uint64_t ComputedOffset) {
  assert(ExternalFieldOffsets.find(Field) != ExternalFieldOffsets.end() &&
         "Field does not have an external offset");
  
  uint64_t ExternalFieldOffset = ExternalFieldOffsets[Field];
  
  if (InferAlignment && ExternalFieldOffset < ComputedOffset) {
    // The externally-supplied field offset is before the field offset we
    // computed. Assume that the structure is packed.
    Alignment = CharUnits::One();
    InferAlignment = false;
  }
  
  // Use the externally-supplied field offset.
  return ExternalFieldOffset;
}

void MeshLayoutBuilder::CheckFieldPadding(uint64_t Offset,
                                          uint64_t UnpaddedOffset,
                                          uint64_t UnpackedOffset,
                                          unsigned UnpackedAlign,
                                          bool isPacked,
                                          const MeshFieldDecl *D) {
  
  // Don't warn about structs created without a SourceLocation.  This can
  // be done by clients of the AST, such as codegen.
  if (D->getLocation().isInvalid())
    return;
  
  unsigned CharBitNum = Context.getTargetInfo().getCharWidth();

  // Warn if padding was introduced to the mesh.
  if (Offset > UnpaddedOffset) {
    unsigned PadSize = Offset - UnpaddedOffset;
    bool InBits = true;
    if (PadSize % CharBitNum == 0) {
      PadSize = PadSize / CharBitNum;
      InBits = false;
    }
    if (D->getIdentifier())
      Diag(D->getLocation(), diag::warn_padded_struct_field)
          << getPaddingDiagFromTagKind(D->getParent()->getTagKind())
          << Context.getTypeDeclType(D->getParent())
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1) // plural or not
          << D->getIdentifier();
    else
      Diag(D->getLocation(), diag::warn_padded_struct_anon_field)
          << getPaddingDiagFromTagKind(D->getParent()->getTagKind())
          << Context.getTypeDeclType(D->getParent())
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1); // plural or not
  }

  // Warn if we packed it unnecessarily. If the alignment is 1 byte don't
  // bother since there won't be alignment issues.
  if (isPacked && UnpackedAlign > CharBitNum && Offset == UnpackedOffset)
    Diag(D->getLocation(), diag::warn_unnecessary_packed)
        << D->getIdentifier();
}



DiagnosticBuilder
MeshLayoutBuilder::Diag(SourceLocation Loc, unsigned DiagID) {
  return Context.getDiagnostics().Report(Loc, DiagID);
}

/// getASTMeshLayout - Get or compute information about the layout of the
/// specified mesh, which indicates its size and field
/// position information.
const ASTMeshLayout &
ASTContext::getASTMeshLayout(const MeshDecl *D) const {
  // These asserts test different things.  A mesh has a definition
  // as soon as we begin to parse the definition.  That definition is
  // not a complete definition (which is what isDefinition() tests)
  // until we *finish* parsing the definition.

  if (D->hasExternalLexicalStorage() && !D->getDefinition())
    getExternalSource()->CompleteType(const_cast<MeshDecl*>(D));
    
  D = D->getDefinition();
  assert(D && "Cannot get layout of forward declarations!");
  assert(D->isCompleteDefinition() && "Cannot layout type before complete!");

  // Look up this layout, if already laid out, return what we have.
  // Note that we can't save a reference to the entry because this function
  // is recursive.
  const ASTMeshLayout *Entry = ASTMeshLayouts[D];
  if (Entry) return *Entry;

  const ASTMeshLayout *NewEntry;

  
  MeshLayoutBuilder Builder(*this, /*EmptySubobjects=*/0);
  Builder.Layout(D);

  NewEntry = new (*this) ASTMeshLayout(*this, Builder.getSize(), 
                                       Builder.Alignment,
                                       Builder.getSize(),
                                       Builder.FieldOffsets.data(),
                                       Builder.FieldOffsets.size());
  ASTMeshLayouts[D] = NewEntry;

  if (getLangOpts().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping AST Mesh Layout\n";
    DumpMeshLayout(D, llvm::errs(), getLangOpts().DumpMeshLayoutsSimple);
  }

  return *NewEntry;
}

static uint64_t getMeshFieldOffset(const ASTContext &C, const MeshFieldDecl *FD) {
  const ASTMeshLayout &Layout = C.getASTMeshLayout(FD->getParentMesh());
  return Layout.getFieldOffset(FD->getFieldIndex());
}

uint64_t ASTContext::getMeshFieldOffset(const ValueDecl *VD) const {
  uint64_t OffsetInBits;
  if (const MeshFieldDecl *FD = dyn_cast<MeshFieldDecl>(VD)) {
    OffsetInBits = ::getMeshFieldOffset(*this, FD);
  } else {
    const IndirectFieldDecl *IFD = cast<IndirectFieldDecl>(VD);

    OffsetInBits = 0;
    for (IndirectFieldDecl::chain_iterator CI = IFD->chain_begin(),
                                           CE = IFD->chain_end();
         CI != CE; ++CI)
      OffsetInBits += ::getMeshFieldOffset(*this, cast<MeshFieldDecl>(*CI));
  }

  return OffsetInBits;
}

void ASTContext::DumpMeshLayout(const MeshDecl *MD,
                                raw_ostream &OS,
                                bool Simple) const {
  const ASTMeshLayout &Info = getASTMeshLayout(MD);



  OS << "Type: " << getTypeDeclType(MD).getAsString() << "\n";
  if (!Simple) {
    OS << "Mesh: ";
    MD->dump();
  }

  OS << "\nLayout: ";
  OS << "<ASTMeshLayout\n";
  OS << "  Size:" << toBits(Info.getSize()) << "\n";
  OS << "  DataSize:" << toBits(Info.getDataSize()) << "\n";
  OS << "  Alignment:" << toBits(Info.getAlignment()) << "\n";
  OS << "  FieldOffsets: [";
  for (unsigned i = 0, e = Info.getFieldCount(); i != e; ++i) {
    if (i) OS << ", ";
    OS << Info.getFieldOffset(i);
  }
  OS << "]>\n";
}
