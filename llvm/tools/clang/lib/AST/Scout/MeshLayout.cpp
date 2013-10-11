//===-- RecordLayout.cpp - Layout information for a struct/union -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RecordLayout interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Scout/MeshLayout.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;

void ASTMeshLayout::Destroy(ASTContext &Ctx) {
  if (FieldOffsets)
    Ctx.Deallocate(FieldOffsets);
  this->~ASTMeshLayout();
  Ctx.Deallocate(this);
}

ASTMeshLayout::ASTMeshLayout(const ASTContext &Ctx, CharUnits size,
                             CharUnits alignment, CharUnits datasize,
                             const uint64_t *fieldoffsets,
                             unsigned fieldcount)
  : Size(size), DataSize(datasize), Alignment(alignment), FieldOffsets(0),
    FieldCount(fieldcount) {
  if (FieldCount > 0)  {
    FieldOffsets = new (Ctx) uint64_t[FieldCount];
    memcpy(FieldOffsets, fieldoffsets, FieldCount * sizeof(*FieldOffsets));
  }
}
