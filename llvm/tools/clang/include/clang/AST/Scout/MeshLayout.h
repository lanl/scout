//===--- MeshLayout.h - Layout information for meshes -*- Scout -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the MeshLayout interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MESHLAYOUT_H
#define LLVM_CLANG_AST_MESHLAYOUT_H

#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Scout/MeshDecl.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  class ASTContext;
  class MeshDecl;
  class MeshFieldDecl;

/// ASTMeshLayout -
/// This class contains layout information for one MeshDecl.
/// The decl represented must be a definition, not a forward declaration.
/// These objects are managed by ASTContext.
class ASTMeshLayout {

public:

  friend class ASTContext;

  ASTMeshLayout(const ASTContext &Ctx, CharUnits size, CharUnits alignment,
                CharUnits datasize, const uint64_t *fieldoffsets,
                unsigned fieldcount);

  ~ASTMeshLayout() {}

  void Destroy(ASTContext &Ctx);

  ASTMeshLayout(const ASTMeshLayout &) = delete;
  void operator=(const ASTMeshLayout &) = delete;

private:
    /// Size - Size of record in characters.
  CharUnits Size;

  /// DataSize - Size of record in characters without tail padding.
  CharUnits DataSize;

  // Alignment - Alignment of record in characters.
  CharUnits Alignment;

  /// FieldOffsets - Array of field offsets in bits.
  uint64_t *FieldOffsets;

  // FieldCount - Number of fields.
  unsigned FieldCount;

public:

  /// getAlignment - Get the mess alignment in characters.
  CharUnits getAlignment() const { return Alignment; }

  /// getSize - Get the mesh size in characters.
  CharUnits getSize() const { return Size; }

  /// getFieldCount - Get the number of fields in the layout.
  unsigned getFieldCount() const { return FieldCount; }

  /// getFieldOffset - Get the offset of the given field index, in
  /// bits.
  uint64_t getFieldOffset(unsigned FieldNo) const {
    assert (FieldNo < FieldCount && "Invalid Field No");
    return FieldOffsets[FieldNo];
  }

  /// getDataSize() - Get the mesh data size, which is the mesh size
  /// without tail padding, in characters.
  CharUnits getDataSize() const {
    return DataSize;
  }
};

}  // end namespace clang

#endif
