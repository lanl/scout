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

#ifndef LLVM_SCOUT_DEBUGINFO_H
#define LLVM_SCOUT_DEBUGINFO_H

#include "llvm/IR/DebugInfo.h"

#define RETURN_FROM_RAW(VALID, DEFAULT)                                        \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return VALID;                                                            \
    return DEFAULT;                                                            \
  } while (false)
#define RETURN_DESCRIPTOR_FROM_RAW(DESC, VALID)                                \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return DESC(dyn_cast_or_null<MDNode>(VALID));                            \
    return DESC(static_cast<const MDNode *>(nullptr));                         \
  } while (false)
#define RETURN_REF_FROM_RAW(REF, VALID)                                        \
  do {                                                                         \
    if (auto *N = getRaw())                                                    \
      return REF::get(VALID);                                                  \
    return REF::get(nullptr);                                                  \
  } while (false)

namespace llvm {

  // Note: the layout of the metadata must be kept in sync. with the
  // layout of DIDerivedType as there may be new fields which are added
  // to the DIDerivedType, and we need to maintain the ability that
  // DIScoutDerivedType is a proper subclass of DIDerivedType

  class DIScoutDerivedType : public DIType {
    MDDerivedTypeBase *getRaw() const {
      return dyn_cast_or_null<MDDerivedTypeBase>(get());
    }
    
  public:
    enum {
      FlagMeshFieldCellLocated     = 1 << 0,
      FlagMeshFieldVertexLocated   = 1 << 1,
      FlagMeshFieldEdgeLocated     = 1 << 2,
      FlagMeshFieldFaceLocated     = 1 << 3
    };
    
    unsigned getScoutFlags() const { assert(false);/*return getUnsignedField(4);*/ }
    
    bool isCellLocated() const {
      return (getScoutFlags() & FlagMeshFieldCellLocated) != 0;
    }
    
    bool isVertexLocated() const {
      return (getScoutFlags() & FlagMeshFieldVertexLocated) != 0;
    }
    
    bool isEdgeLocated() const {
      return (getScoutFlags() & FlagMeshFieldEdgeLocated) != 0;
    }
    
    bool isFaceLocated() const {
      return (getScoutFlags() & FlagMeshFieldFaceLocated) != 0;
    }
    
    explicit DIScoutDerivedType(const MDNode *N = nullptr) : DIType(N) {}
    DIScoutDerivedType(const MDDerivedTypeBase *N) : DIType(N) {}
    
    DITypeRef getTypeDerivedFrom() const {
      RETURN_REF_FROM_RAW(DITypeRef, N->getBaseType());
    }
    
    /// \brief Return property node, if this ivar is associated with one.
    MDNode *getObjCProperty() const {
      if (auto *N = dyn_cast_or_null<MDScoutDerivedType>(get()))
        return dyn_cast_or_null<MDNode>(N->getExtraData());
      return nullptr;
    }
    
    DITypeRef getClassType() const {
      assert(getTag() == dwarf::DW_TAG_ptr_to_member_type);
      if (auto *N = dyn_cast_or_null<MDScoutDerivedType>(get()))
        return DITypeRef::get(N->getExtraData());
      return DITypeRef::get(nullptr);
    }
    
    Constant *getConstant() const {
      assert((getTag() == dwarf::DW_TAG_member) && isStaticMember());
      if (auto *N = dyn_cast_or_null<MDScoutDerivedType>(get()))
        if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(N->getExtraData()))
          return C->getValue();
      
      return nullptr;
    }
    
    bool Verify() const;
  };

  class DIScoutCompositeType : public DIDerivedType {
    friend class DIBuilder;
    
    /// \brief Set the array of member DITypes.
    void setArraysHelper(MDNode *Elements, MDNode *TParams);
    
    MDCompositeTypeBase *getRaw() const {
      return dyn_cast_or_null<MDCompositeTypeBase>(get());
    }
    
  public:
    explicit DIScoutCompositeType(const MDNode *N = nullptr) : DIDerivedType(N) {}
    DIScoutCompositeType(const MDCompositeTypeBase *N) : DIDerivedType(N) {}
    
    DIArray getElements() const {
      assert(!isSubroutineType() && "no elements for DISubroutineType");
      RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getElements());
    }
    
  private:
    template <typename T>
    void setArrays(DITypedArray<T> Elements, DIArray TParams = DIArray()) {
      assert(
             (!TParams || DbgNode->getNumOperands() == 8) &&
             "If you're setting the template parameters this should include a slot "
             "for that!");
      setArraysHelper(Elements, TParams);
    }
    
  public:
    unsigned getRunTimeLang() const { RETURN_FROM_RAW(N->getRuntimeLang(), 0); }
    DITypeRef getContainingType() const {
      RETURN_REF_FROM_RAW(DITypeRef, N->getVTableHolder());
    }
    
  private:
    /// \brief Set the containing type.
    void setContainingType(DIScoutCompositeType ContainingType);
    
  public:
    DIArray getTemplateParams() const {
      RETURN_DESCRIPTOR_FROM_RAW(DIArray, N->getTemplateParams());
    }
    MDString *getIdentifier() const {
      RETURN_FROM_RAW(N->getRawIdentifier(), nullptr);
    }
    
    bool Verify() const;
    
    unsigned getDimension(int dim) const{
      assert(false && "unimplemented");
    }
  };

} // end namespace llvm

#undef RETURN_FROM_RAW
#undef RETURN_DESCRIPTOR_FROM_RAW
#undef RETURN_REF_FROM_RAW

#endif
