//=== MeshLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecordLayout.h"
#include "clang/AST/scout/MeshLayout.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;

namespace {

/// BaseSubobjectInfo - Represents a single base subobject in a complete class.
/// For a class hierarchy like
///
/// class A { };
/// class B : A { };
/// class C : A, B { };
///
/// The BaseSubobjectInfo graph for C will have three BaseSubobjectInfo
/// instances, one for B and two for A.
///
/// If a base is virtual, it will only have one BaseSubobjectInfo allocated.
struct BaseSubobjectInfo {
  /// Class - The class for this base info.
  const CXXRecordDecl *Class;

  /// IsVirtual - Whether the BaseInfo represents a virtual base or not.
  bool IsVirtual;

  /// Bases - Information about the base subobjects.
  SmallVector<BaseSubobjectInfo*, 4> Bases;

  /// PrimaryVirtualBaseInfo - Holds the base info for the primary virtual base
  /// of this base info (if one exists).
  BaseSubobjectInfo *PrimaryVirtualBaseInfo;

  // FIXME: Document.
  const BaseSubobjectInfo *Derived;
};

/// EmptySubobjectMap - Keeps track of which empty subobjects exist at different
/// offsets while laying out a C++ class.
class EmptySubobjectMap {
  const ASTContext &Context;
  uint64_t CharWidth;
  
  /// Class - The class whose empty entries we're keeping track of.
  const CXXRecordDecl *Class;

  /// EmptyClassOffsets - A map from offsets to empty record decls.
  typedef SmallVector<const CXXRecordDecl *, 1> ClassVectorTy;
  typedef llvm::DenseMap<CharUnits, ClassVectorTy> EmptyClassOffsetsMapTy;
  EmptyClassOffsetsMapTy EmptyClassOffsets;
  
  /// MaxEmptyClassOffset - The highest offset known to contain an empty
  /// base subobject.
  CharUnits MaxEmptyClassOffset;
  
  /// ComputeEmptySubobjectSizes - Compute the size of the largest base or
  /// member subobject that is empty.
  void ComputeEmptySubobjectSizes();
  
  void AddSubobjectAtOffset(const CXXRecordDecl *RD, CharUnits Offset);
  
  void UpdateEmptyBaseSubobjects(const BaseSubobjectInfo *Info,
                                 CharUnits Offset, bool PlacingEmptyBase);
  
  void UpdateEmptyFieldSubobjects(const CXXRecordDecl *RD, 
                                  const CXXRecordDecl *Class,
                                  CharUnits Offset);
  void UpdateEmptyFieldSubobjects(const FieldDecl *FD, CharUnits Offset);
  
  /// AnyEmptySubobjectsBeyondOffset - Returns whether there are any empty
  /// subobjects beyond the given offset.
  bool AnyEmptySubobjectsBeyondOffset(CharUnits Offset) const {
    return Offset <= MaxEmptyClassOffset;
  }

  CharUnits 
  getFieldOffset(const ASTRecordLayout &Layout, unsigned FieldNo) const {
    uint64_t FieldOffset = Layout.getFieldOffset(FieldNo);
    assert(FieldOffset % CharWidth == 0 && 
           "Field offset not at char boundary!");

    return Context.toCharUnitsFromBits(FieldOffset);
  }

protected:
  bool CanPlaceSubobjectAtOffset(const CXXRecordDecl *RD,
                                 CharUnits Offset) const;

  bool CanPlaceBaseSubobjectAtOffset(const BaseSubobjectInfo *Info,
                                     CharUnits Offset);

  bool CanPlaceFieldSubobjectAtOffset(const CXXRecordDecl *RD, 
                                      const CXXRecordDecl *Class,
                                      CharUnits Offset) const;
  bool CanPlaceFieldSubobjectAtOffset(const FieldDecl *FD,
                                      CharUnits Offset) const;

public:
  /// This holds the size of the largest empty subobject (either a base
  /// or a member). Will be zero if the record being built doesn't contain
  /// any empty classes.
  CharUnits SizeOfLargestEmptySubobject;

  EmptySubobjectMap(const ASTContext &Context, const CXXRecordDecl *Class)
  : Context(Context), CharWidth(Context.getCharWidth()), Class(Class) {
      ComputeEmptySubobjectSizes();
  }

  /// CanPlaceBaseAtOffset - Return whether the given base class can be placed
  /// at the given offset.
  /// Returns false if placing the record will result in two components
  /// (direct or indirect) of the same type having the same offset.
  bool CanPlaceBaseAtOffset(const BaseSubobjectInfo *Info,
                            CharUnits Offset);

  /// CanPlaceFieldAtOffset - Return whether a field can be placed at the given
  /// offset.
  bool CanPlaceFieldAtOffset(const FieldDecl *FD, CharUnits Offset);
};

void EmptySubobjectMap::ComputeEmptySubobjectSizes() {
  // Check the bases.
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
       E = Class->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits EmptySize;
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(BaseDecl);
    if (BaseDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

    if (EmptySize > SizeOfLargestEmptySubobject)
      SizeOfLargestEmptySubobject = EmptySize;
  }

  // Check the fields.
  for (CXXRecordDecl::field_iterator I = Class->field_begin(),
       E = Class->field_end(); I != E; ++I) {

    const RecordType *RT =
      Context.getBaseElementType(I->getType())->getAs<RecordType>();

    // We only care about record types.
    if (!RT)
      continue;

    CharUnits EmptySize;
    const CXXRecordDecl *MemberDecl = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(MemberDecl);
    if (MemberDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

    if (EmptySize > SizeOfLargestEmptySubobject)
      SizeOfLargestEmptySubobject = EmptySize;
  }
}

bool
EmptySubobjectMap::CanPlaceSubobjectAtOffset(const CXXRecordDecl *RD, 
                                             CharUnits Offset) const {
  // We only need to check empty bases.
  if (!RD->isEmpty())
    return true;

  EmptyClassOffsetsMapTy::const_iterator I = EmptyClassOffsets.find(Offset);
  if (I == EmptyClassOffsets.end())
    return true;
  
  const ClassVectorTy& Classes = I->second;
  if (std::find(Classes.begin(), Classes.end(), RD) == Classes.end())
    return true;

  // There is already an empty class of the same type at this offset.
  return false;
}
  
void EmptySubobjectMap::AddSubobjectAtOffset(const CXXRecordDecl *RD, 
                                             CharUnits Offset) {
  // We only care about empty bases.
  if (!RD->isEmpty())
    return;

  // If we have empty structures inside a union, we can assign both
  // the same offset. Just avoid pushing them twice in the list.
  ClassVectorTy& Classes = EmptyClassOffsets[Offset];
  if (std::find(Classes.begin(), Classes.end(), RD) != Classes.end())
    return;
  
  Classes.push_back(RD);
  
  // Update the empty class offset.
  if (Offset > MaxEmptyClassOffset)
    MaxEmptyClassOffset = Offset;
}

bool
EmptySubobjectMap::CanPlaceBaseSubobjectAtOffset(const BaseSubobjectInfo *Info,
                                                 CharUnits Offset) {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;

  if (!CanPlaceSubobjectAtOffset(Info->Class, Offset))
    return false;

  // Traverse all non-virtual bases.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseSubobjectInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);

    if (!CanPlaceBaseSubobjectAtOffset(Base, BaseOffset))
      return false;
  }

  if (Info->PrimaryVirtualBaseInfo) {
    BaseSubobjectInfo *PrimaryVirtualBaseInfo = Info->PrimaryVirtualBaseInfo;

    if (Info == PrimaryVirtualBaseInfo->Derived) {
      if (!CanPlaceBaseSubobjectAtOffset(PrimaryVirtualBaseInfo, Offset))
        return false;
    }
  }
  
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = Info->Class->field_begin(), 
       E = Info->Class->field_end(); I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;
  
    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    if (!CanPlaceFieldSubobjectAtOffset(*I, FieldOffset))
      return false;
  }
  
  return true;
}

void EmptySubobjectMap::UpdateEmptyBaseSubobjects(const BaseSubobjectInfo *Info, 
                                                  CharUnits Offset,
                                                  bool PlacingEmptyBase) {
  if (!PlacingEmptyBase && Offset >= SizeOfLargestEmptySubobject) {
    // We know that the only empty subobjects that can conflict with empty
    // subobject of non-empty bases, are empty bases that can be placed at
    // offset zero. Because of this, we only need to keep track of empty base 
    // subobjects with offsets less than the size of the largest empty
    // subobject for our class.    
    return;
  }

  AddSubobjectAtOffset(Info->Class, Offset);

  // Traverse all non-virtual bases.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseSubobjectInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);
    UpdateEmptyBaseSubobjects(Base, BaseOffset, PlacingEmptyBase);
  }

  if (Info->PrimaryVirtualBaseInfo) {
    BaseSubobjectInfo *PrimaryVirtualBaseInfo = Info->PrimaryVirtualBaseInfo;
    
    if (Info == PrimaryVirtualBaseInfo->Derived)
      UpdateEmptyBaseSubobjects(PrimaryVirtualBaseInfo, Offset,
                                PlacingEmptyBase);
  }

  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = Info->Class->field_begin(), 
       E = Info->Class->field_end(); I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    UpdateEmptyFieldSubobjects(*I, FieldOffset);
  }
}

bool EmptySubobjectMap::CanPlaceBaseAtOffset(const BaseSubobjectInfo *Info,
                                             CharUnits Offset) {
  // If we know this class doesn't have any empty subobjects we don't need to
  // bother checking.
  if (SizeOfLargestEmptySubobject.isZero())
    return true;

  if (!CanPlaceBaseSubobjectAtOffset(Info, Offset))
    return false;

  // We are able to place the base at this offset. Make sure to update the
  // empty base subobject map.
  UpdateEmptyBaseSubobjects(Info, Offset, Info->Class->isEmpty());
  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldSubobjectAtOffset(const CXXRecordDecl *RD, 
                                                  const CXXRecordDecl *Class,
                                                  CharUnits Offset) const {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;

  if (!CanPlaceSubobjectAtOffset(RD, Offset))
    return false;
  
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Traverse all non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(BaseDecl);
    if (!CanPlaceFieldSubobjectAtOffset(BaseDecl, Class, BaseOffset))
      return false;
  }

  if (RD == Class) {
    // This is the most derived class, traverse virtual bases as well.
    for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *VBaseDecl =
        cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      CharUnits VBaseOffset = Offset + Layout.getVBaseClassOffset(VBaseDecl);
      if (!CanPlaceFieldSubobjectAtOffset(VBaseDecl, Class, VBaseOffset))
        return false;
    }
  }
    
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    
    if (!CanPlaceFieldSubobjectAtOffset(*I, FieldOffset))
      return false;
  }

  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldSubobjectAtOffset(const FieldDecl *FD,
                                                  CharUnits Offset) const {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;
  
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    return CanPlaceFieldSubobjectAtOffset(RD, RD, Offset);
  }

  // If we have an array type we need to look at every element.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return true;
  
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    CharUnits ElementOffset = Offset;
    for (uint64_t I = 0; I != NumElements; ++I) {
      // We don't have to keep looking past the maximum offset that's known to
      // contain an empty class.
      if (!AnyEmptySubobjectsBeyondOffset(ElementOffset))
        return true;
      
      if (!CanPlaceFieldSubobjectAtOffset(RD, RD, ElementOffset))
        return false;

      ElementOffset += Layout.getSize();
    }
  }

  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldAtOffset(const FieldDecl *FD, 
                                         CharUnits Offset) {
  if (!CanPlaceFieldSubobjectAtOffset(FD, Offset))
    return false;
  
  // We are able to place the member variable at this offset.
  // Make sure to update the empty base subobject map.
  UpdateEmptyFieldSubobjects(FD, Offset);
  return true;
}

void EmptySubobjectMap::UpdateEmptyFieldSubobjects(const CXXRecordDecl *RD, 
                                                   const CXXRecordDecl *Class,
                                                   CharUnits Offset) {
  // We know that the only empty subobjects that can conflict with empty
  // field subobjects are subobjects of empty bases that can be placed at offset
  // zero. Because of this, we only need to keep track of empty field 
  // subobjects with offsets less than the size of the largest empty
  // subobject for our class.
  if (Offset >= SizeOfLargestEmptySubobject)
    return;

  AddSubobjectAtOffset(RD, Offset);

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Traverse all non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(BaseDecl);
    UpdateEmptyFieldSubobjects(BaseDecl, Class, BaseOffset);
  }

  if (RD == Class) {
    // This is the most derived class, traverse virtual bases as well.
    for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *VBaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      CharUnits VBaseOffset = Offset + Layout.getVBaseClassOffset(VBaseDecl);
      UpdateEmptyFieldSubobjects(VBaseDecl, Class, VBaseOffset);
    }
  }
  
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);

    UpdateEmptyFieldSubobjects(*I, FieldOffset);
  }
}
  
void EmptySubobjectMap::UpdateEmptyFieldSubobjects(const FieldDecl *FD,
                                                   CharUnits Offset) {
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    UpdateEmptyFieldSubobjects(RD, RD, Offset);
    return;
  }

  // If we have an array type we need to update every element.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return;
    
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    CharUnits ElementOffset = Offset;
    
    for (uint64_t I = 0; I != NumElements; ++I) {
      // We know that the only empty subobjects that can conflict with empty
      // field subobjects are subobjects of empty bases that can be placed at 
      // offset zero. Because of this, we only need to keep track of empty field
      // subobjects with offsets less than the size of the largest empty
      // subobject for our class.
      if (ElementOffset >= SizeOfLargestEmptySubobject)
        return;

      UpdateEmptyFieldSubobjects(RD, RD, ElementOffset);
      ElementOffset += Layout.getSize();
    }
  }
}

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

/// \brief Get diagnostic %select index for tag kind for
/// field padding diagnostic message.
/// WARNING: Indexes apply to particular diagnostics only!
///
/// \returns diagnostic %select index.
static unsigned getPaddingDiagFromTagKind(TagTypeKind Tag) {
  switch (Tag) {
  case TTK_Struct: return 0;
  case TTK_Interface: return 1;
  case TTK_Class: return 2;
  default: llvm_unreachable("Invalid tag kind for field padding diagnostic!");
  }
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


static uint64_t getFieldOffset(const ASTContext &C, const MeshFieldDecl *FD) {
  const ASTMeshLayout &Layout = C.getASTMeshLayout(FD->getParentMesh());
  return Layout.getFieldOffset(FD->getFieldIndex());
}

uint64_t ASTContext::getFieldOffset(const ValueDecl *VD) const {
  uint64_t OffsetInBits;
  if (const MeshFieldDecl *FD = dyn_cast<MeshFieldDecl>(VD)) {
    OffsetInBits = ::getFieldOffset(*this, FD);
  } else {
    const IndirectFieldDecl *IFD = cast<IndirectFieldDecl>(VD);

    OffsetInBits = 0;
    for (IndirectFieldDecl::chain_iterator CI = IFD->chain_begin(),
                                           CE = IFD->chain_end();
         CI != CE; ++CI)
      OffsetInBits += ::getFieldOffset(*this, cast<MeshFieldDecl>(*CI));
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
