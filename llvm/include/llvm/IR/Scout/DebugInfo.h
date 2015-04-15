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
  public:
    DIScoutDerivedType() = default;
    DIScoutDerivedType(const MDScoutDerivedType *N) : DIType(N) {}
    
    MDScoutDerivedType *get() const {
      return cast_or_null<MDScoutDerivedType>(DIDescriptor::get());
    }
    operator MDScoutDerivedType *() const { return get(); }
    MDScoutDerivedType *operator->() const { return get(); }
    MDScoutDerivedType &operator*() const { return *get(); }
    
    DITypeRef getTypeDerivedFrom() const { return get()->getBaseType(); }
    
    /// \brief Return property node, if this ivar is associated with one.
    MDObjCProperty *getObjCProperty() const {
      return cast<MDDerivedType>(get())->getObjCProperty();
    }
    
    DITypeRef getClassType() const {
      return cast<MDDerivedType>(get())->getClassType();
    }
    
    Constant *getConstant() const {
      return cast<MDDerivedType>(get())->getConstant();
    }
    
    unsigned getScoutFlags() const { return get()->getScoutFlags(); }
    
    enum {
      FlagMeshFieldCellLocated     = 1 << 0,
      FlagMeshFieldVertexLocated   = 1 << 1,
      FlagMeshFieldEdgeLocated     = 1 << 2,
      FlagMeshFieldFaceLocated     = 1 << 3
    };
    
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
  };

  class DIScoutCompositeType : public DIDerivedType {
    friend class DIBuilder;
    
  public:
    DIScoutCompositeType() = default;
    DIScoutCompositeType(const MDScoutCompositeType *N) : DIDerivedType(N) {}
    
    MDScoutCompositeType *get() const {
      return cast_or_null<MDScoutCompositeType>(DIDescriptor::get());
    }
    operator MDScoutCompositeType *() const { return get(); }
    MDScoutCompositeType *operator->() const { return get(); }
    MDScoutCompositeType &operator*() const { return *get(); }
    
    DIArray getElements() const { return get()->getElements(); }
    
    unsigned getRunTimeLang() const { return get()->getRuntimeLang(); }
    DITypeRef getContainingType() const { return get()->getVTableHolder(); }
    
    DIArray getTemplateParams() const { return get()->getTemplateParams(); }
    MDString *getIdentifier() const { return get()->getRawIdentifier(); }
    
    unsigned getDimension(int dim) const{
      switch(dim){
        case 0:
          return get()->getDimX();
        case 1:
          return get()->getDimY();
        case 2:
          return get()->getDimZ();
        default:
          assert(false && "invalid dimension");
      }
    }
  };
  
} // end namespace llvm

#undef RETURN_FROM_RAW
#undef RETURN_DESCRIPTOR_FROM_RAW
#undef RETURN_REF_FROM_RAW

#endif
