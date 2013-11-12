#include "CGDebugInfo.h"
#include "CGBlocks.h"
#include "CGCXXABI.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "clang/AST/Scout/MeshLayout.h"

using namespace clang;
using namespace clang::CodeGen;


/// In C++ mode, types have linkage, so we can rely on the ODR and
/// on their mangled names, if they're external.
static SmallString<256>
getUniqueMeshTypeName(const MeshType *Ty, CodeGenModule &CGM,
                     llvm::DICompileUnit TheCU) {
  SmallString<256> FullName;
  // FIXME: ODR should apply to ObjC++ exactly the same wasy it does to C++.
  // For now, only apply ODR with C++.
  const MeshDecl *TD = Ty->getDecl();
  if (TheCU.getLanguage() != llvm::dwarf::DW_LANG_C_plus_plus ||
      !TD->isExternallyVisible())
    return FullName;
  // Microsoft Mangler does not have support for mangleCXXRTTIName yet.
  if (CGM.getTarget().getCXXABI().isMicrosoft())
    return FullName;

  // TODO: This is using the RTTI name. Is there a better way to get
  // a unique string for a type?
  llvm::raw_svector_ostream Out(FullName);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTIName(QualType(Ty, 0), Out);
  Out.flush();
  return FullName;
}

void CGDebugInfo::completeType(const MeshDecl *MD) {
  if (DebugKind > CodeGenOptions::LimitedDebugInfo ||
      !CGM.getLangOpts().CPlusPlus) {
    if(const UniformMeshDecl *UMD = dyn_cast<UniformMeshDecl>(MD))
      completeRequiredType(UMD);
    if (const RectilinearMeshDecl *RMD = dyn_cast<RectilinearMeshDecl>(MD))
      completeRequiredType(RMD);
    if (const StructuredMeshDecl *SMD = dyn_cast<StructuredMeshDecl>(MD))
      completeRequiredType(SMD);
    if (const UnstructuredMeshDecl *USMD = dyn_cast<UnstructuredMeshDecl>(MD))
      completeRequiredType(USMD);
  }
}

void CGDebugInfo::completeRequiredType(const UniformMeshDecl *MD) {
  QualType Ty = CGM.getContext().getUniformMeshType(MD);
  llvm::DIType T = getTypeOrNull(Ty);
  if (T && T.isForwardDecl())
    completeClassData(MD);
}

void CGDebugInfo::completeRequiredType(const RectilinearMeshDecl *MD) {
  QualType Ty = CGM.getContext().getRectilinearMeshType(MD);
  llvm::DIType T = getTypeOrNull(Ty);
  if (T && T.isForwardDecl())
    completeClassData(MD);
}

void CGDebugInfo::completeRequiredType(const StructuredMeshDecl *MD) {
  QualType Ty = CGM.getContext().getStructuredMeshType(MD);
  llvm::DIType T = getTypeOrNull(Ty);
  if (T && T.isForwardDecl())
    completeClassData(MD);
}


void CGDebugInfo::completeRequiredType(const UnstructuredMeshDecl *MD) {
  QualType Ty = CGM.getContext().getUnstructuredMeshType(MD);
  llvm::DIType T = getTypeOrNull(Ty);
  if (T && T.isForwardDecl())
    completeClassData(MD);
}

void CGDebugInfo::completeClassData(const UniformMeshDecl *MD) {
  if (DebugKind <= CodeGenOptions::DebugLineTablesOnly)
    return;
  QualType Ty = CGM.getContext().getUniformMeshType(MD);
  void* TyPtr = Ty.getAsOpaquePtr();
  if (CompletedTypeCache.count(TyPtr))
    return;
  llvm::DIType Res = CreateTypeDefinition(Ty->castAs<UniformMeshType>());
  assert(!Res.isForwardDecl());
  CompletedTypeCache[TyPtr] = Res;
  TypeCache[TyPtr] = Res;
}

void CGDebugInfo::completeClassData(const RectilinearMeshDecl *MD) {
  if (DebugKind <= CodeGenOptions::DebugLineTablesOnly)
    return;
  QualType Ty = CGM.getContext().getRectilinearMeshType(MD);
  void* TyPtr = Ty.getAsOpaquePtr();
  if (CompletedTypeCache.count(TyPtr))
    return;
  llvm::DIType Res = CreateTypeDefinition(Ty->castAs<RectilinearMeshType>());
  assert(!Res.isForwardDecl());
  CompletedTypeCache[TyPtr] = Res;
  TypeCache[TyPtr] = Res;
}

void CGDebugInfo::completeClassData(const StructuredMeshDecl *MD) {
  if (DebugKind <= CodeGenOptions::DebugLineTablesOnly)
    return;
  QualType Ty = CGM.getContext().getStructuredMeshType(MD);
  void* TyPtr = Ty.getAsOpaquePtr();
  if (CompletedTypeCache.count(TyPtr))
    return;
  llvm::DIType Res = CreateTypeDefinition(Ty->castAs<StructuredMeshType>());
  assert(!Res.isForwardDecl());
  CompletedTypeCache[TyPtr] = Res;
  TypeCache[TyPtr] = Res;
}

void CGDebugInfo::completeClassData(const UnstructuredMeshDecl *MD) {
  if (DebugKind <= CodeGenOptions::DebugLineTablesOnly)
    return;
  QualType Ty = CGM.getContext().getUnstructuredMeshType(MD);
  void* TyPtr = Ty.getAsOpaquePtr();
  if (CompletedTypeCache.count(TyPtr))
    return;
  llvm::DIType Res = CreateTypeDefinition(Ty->castAs<UnstructuredMeshType>());
  assert(!Res.isForwardDecl());
  CompletedTypeCache[TyPtr] = Res;
  TypeCache[TyPtr] = Res;
}



//===----------------------------------------------------------------------===//
// Uniform Mesh debug support
//===----------------------------------------------------------------------===//


// ----- CreateType
//
llvm::DIType CGDebugInfo::CreateType(const UniformMeshType *Ty) {

  UniformMeshDecl *MD = Ty->getDecl();

  // Always emit declarations for types that aren't required to be
  // complete when in limit-debug-info mode.  If the type is later
  // found to be required to be complete this declaration will be
  // upgraded to a definition by 'completeRequiredType'.
  llvm::DICompositeType T(getTypeOrNull(QualType(Ty, 0)));

  // If we're already emitted the type just use that, even if it is only a
  // declaration.  The completeType(), completeRequiredType(), and
  // completeClassData() callbacks will handle promoting the declaration
  // to a definition.
  if (T || (DebugKind <= CodeGenOptions::LimitedDebugInfo &&
    // Under -flimit-debug-info emit only a declaration unless the type is
    // required to be complete.
    !MD->isCompleteDefinitionRequired() && CGM.getLangOpts().CPlusPlus)) {

    llvm::DIDescriptor FDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));
    if (!T)
      T = getOrCreateMeshFwdDecl(Ty, FDContext);
    return T;
  }

  return CreateTypeDefinition(Ty);
}

// ----- CreateTypeDefinition
//
llvm::DIType CGDebugInfo::CreateTypeDefinition(const UniformMeshType *Ty) {
  UniformMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Meshes can be recursive.  To handle them, we first generate a
  // debug descriptor for the mesh as a forward declaration.  Then
  // (if it is a definition) we go through and get debug information
  // for all of its members.  Finally, we create a descriptor for the
  // complete type (which may refer to forward decl if it is recursive)
  // and replace all uses of the forward declaration with the final
  // definition.

  llvm::DICompositeType FwdDecl(getOrCreateLimitedType(Ty, DefUnit));
  assert(FwdDecl.isCompositeType() &&
    "The debug type of UniformMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the mesh on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[MD] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;
  // what about nested types?

  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);

  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements);

  RegionMap[MD] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}

// TODO: Currently used for context chains when limiting debug info.
llvm::DICompositeType
CGDebugInfo::CreateLimitedType(const UniformMeshType *Ty) {
  UniformMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  llvm::DIDescriptor MDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));

  // If we ended up creating the type during the context chain construction,
  // just return that.
  // FIXME: this could be dealt with better if the type was recorded as
  // completed before we started this (see the CompletedTypeCache usage in
  // CGDebugInfo::CreateTypeDefinition(const MeshType*) - that would need to
  // be pushed to before context creation, but after it was known to be
  // destined for completion (might still have an issue if this caller only
  // required a declaration but the context construction ended up creating a
  // definition)
  llvm::DICompositeType T(
      getTypeOrNull(CGM.getContext().getUniformMeshType(MD)));
  if (T && (!T.isForwardDecl() || !MD->getDefinition()))
      return T;

  // If this is just a forward declaration, construct an appropriately
  // marked node and just return it.
  if (!MD->getDefinition())
    return getOrCreateMeshFwdDecl(Ty, MDContext);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DICompositeType RealDecl;

  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);

  RealDecl = DBuilder.createStructType(MDContext, MDName, DefUnit, Line,
                                       Size, Align, 0, llvm::DIType(),
                                       llvm::DIArray(), 0, 0, FullName);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(RealDecl);
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl;

  return RealDecl;
}

// Creates a forward declaration for a uniform mesh the given context.
llvm::DICompositeType
CGDebugInfo::getOrCreateMeshFwdDecl(const UniformMeshType *Ty,
                                    llvm::DIDescriptor Ctx) {
  const UniformMeshDecl *MD = Ty->getDecl();
  if (llvm::DIType T = getTypeOrNull(CGM.getContext().getUniformMeshType(MD)))
    return llvm::DICompositeType(T);

  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;

  // Create the type.
  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);
  return DBuilder.createForwardDecl(Tag, MDName, Ctx, DefUnit, Line, 0, 0, 0,
                                    FullName);
}

/// getOrCreateLimitedType - Get the type from the cache or create a new
/// limited type if necessary.
llvm::DIType CGDebugInfo::getOrCreateLimitedType(const UniformMeshType *Ty,
                                                 llvm::DIFile Unit) {
  QualType QTy(Ty, 0);

  llvm::DICompositeType T(getTypeOrNull(QTy));

  // We may have cached a forward decl when we could have created
  // a non-forward decl. Go ahead and create a non-forward decl
  // now.
  if (T && !T.isForwardDecl()) return T;

  // Otherwise create the type.
  llvm::DICompositeType Res = CreateLimitedType(Ty);

  // Propagate members from the declaration to the definition
  // CreateType(const MeshType*) will overwrite this with the members in the
  // correct order if the full type is needed.
  Res.setTypeArray(T.getTypeArray());

  if (T && T.isForwardDecl())
    ReplaceMap.push_back(
        std::make_pair(QTy.getAsOpaquePtr(), static_cast<llvm::Value *>(T)));

  // And update the type cache.
  TypeCache[QTy.getAsOpaquePtr()] = Res;
  return Res;
}


//===----------------------------------------------------------------------===//
// Rectilinear Mesh debug support
//===----------------------------------------------------------------------===//

// ------ CreateType
//
llvm::DIType CGDebugInfo::CreateType(const RectilinearMeshType *Ty) {

  RectilinearMeshDecl *MD = Ty->getDecl();

  // Always emit declarations for types that aren't required to be
  // complete when in limit-debug-info mode.  If the type is later
  // found to be required to be complete this declaration will be
  // upgraded to a definition by 'completeRequiredType'.
  llvm::DICompositeType T(getTypeOrNull(QualType(Ty, 0)));

  // If we're already emitted the type just use that, even if it is only a
  // declaration.  The completeType(), completeRequiredType(), and
  // completeClassData() callbacks will handle promoting the declaration
  // to a definition.
  if (T || (DebugKind <= CodeGenOptions::LimitedDebugInfo &&
     // Under -flimit-debug-info emit only a declaration unless the type is
    // required to be complete.
    !MD->isCompleteDefinitionRequired() && CGM.getLangOpts().CPlusPlus)) {

    llvm::DIDescriptor FDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));
    if (!T)
      T = getOrCreateMeshFwdDecl(Ty, FDContext);
    return T;
  }

  return CreateTypeDefinition(Ty);
}

// ------ CreateTypeDefinition
//
llvm::DIType CGDebugInfo::CreateTypeDefinition(const RectilinearMeshType *Ty) {
  RectilinearMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Meshes can be recursive.  To handle them, we first generate a
  // debug descriptor for the mesh as a forward declaration.  Then
  // (if it is a definition) we go through and get debug information
  // for all of its members.  Finally, we create a descriptor for the
  // complete type (which may refer to forward decl if it is recursive)
  // and replace all uses of the forward declaration with the final
  // definition.

  llvm::DICompositeType FwdDecl(getOrCreateLimitedType(Ty, DefUnit));
  assert(FwdDecl.isCompositeType() &&
    "The debug type of RectilinearMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the mesh on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[MD] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;
  // what about nested types?

  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);

  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements);

  RegionMap[MD] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}


// TODO: Currently used for context chains when limiting debug info.
llvm::DICompositeType
CGDebugInfo::CreateLimitedType(const RectilinearMeshType *Ty) {
  RectilinearMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  llvm::DIDescriptor MDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));

  // If we ended up creating the type during the context chain construction,
  // just return that.
  // FIXME: this could be dealt with better if the type was recorded as
  // completed before we started this (see the CompletedTypeCache usage in
  // CGDebugInfo::CreateTypeDefinition(const MeshType*) - that would need to
  // be pushed to before context creation, but after it was known to be
  // destined for completion (might still have an issue if this caller only
  // required a declaration but the context construction ended up creating a
  // definition)
  llvm::DICompositeType T(
      getTypeOrNull(CGM.getContext().getRectilinearMeshType(MD)));
  if (T && (!T.isForwardDecl() || !MD->getDefinition()))
      return T;

  // If this is just a forward declaration, construct an appropriately
  // marked node and just return it.
  if (!MD->getDefinition())
    return getOrCreateMeshFwdDecl(Ty, MDContext);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DICompositeType RealDecl;

  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);

  RealDecl = DBuilder.createStructType(MDContext, MDName, DefUnit, Line,
                                       Size, Align, 0, llvm::DIType(),
                                       llvm::DIArray(), 0, 0, FullName);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(RealDecl);
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl;

  return RealDecl;
}

// Creates a forward declaration for a rectilinear mesh the given context.
llvm::DICompositeType
CGDebugInfo::getOrCreateMeshFwdDecl(const RectilinearMeshType *Ty,
                                    llvm::DIDescriptor Ctx) {
  const RectilinearMeshDecl *MD = Ty->getDecl();
  if (llvm::DIType T = getTypeOrNull(CGM.getContext().getRectilinearMeshType(MD)))
    return llvm::DICompositeType(T);

  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;

  // Create the type.
  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);
  return DBuilder.createForwardDecl(Tag, MDName, Ctx, DefUnit, Line, 0, 0, 0,
                                    FullName);
}

/// getOrCreateLimitedType - Get the type from the cache or create a new
/// limited type if necessary.
llvm::DIType CGDebugInfo::getOrCreateLimitedType(const RectilinearMeshType *Ty,
                                                 llvm::DIFile Unit) {
  QualType QTy(Ty, 0);

  llvm::DICompositeType T(getTypeOrNull(QTy));

  // We may have cached a forward decl when we could have created
  // a non-forward decl. Go ahead and create a non-forward decl
  // now.
  if (T && !T.isForwardDecl()) return T;

  // Otherwise create the type.
  llvm::DICompositeType Res = CreateLimitedType(Ty);

  // Propagate members from the declaration to the definition
  // CreateType(const MeshType*) will overwrite this with the members in the
  // correct order if the full type is needed.
  Res.setTypeArray(T.getTypeArray());

  if (T && T.isForwardDecl())
    ReplaceMap.push_back(
        std::make_pair(QTy.getAsOpaquePtr(), static_cast<llvm::Value *>(T)));

  // And update the type cache.
  TypeCache[QTy.getAsOpaquePtr()] = Res;
  return Res;
}


//===----------------------------------------------------------------------===//
// Structured Mesh debug support
//===----------------------------------------------------------------------===//

// ------ CreateType
//
llvm::DIType CGDebugInfo::CreateType(const StructuredMeshType *Ty) {

  StructuredMeshDecl *MD = Ty->getDecl();

  // Always emit declarations for types that aren't required to be
  // complete when in limit-debug-info mode.  If the type is later
  // found to be required to be complete this declaration will be
  // upgraded to a definition by 'completeRequiredType'.
  llvm::DICompositeType T(getTypeOrNull(QualType(Ty, 0)));

  // If we're already emitted the type just use that, even if it is only a
  // declaration.  The completeType(), completeRequiredType(), and
  // completeClassData() callbacks will handle promoting the declaration
  // to a definition.
  if (T || (DebugKind <= CodeGenOptions::LimitedDebugInfo &&
     // Under -flimit-debug-info emit only a declaration unless the type is
    // required to be complete.
    !MD->isCompleteDefinitionRequired() && CGM.getLangOpts().CPlusPlus)) {

    llvm::DIDescriptor FDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));
    if (!T)
      T = getOrCreateMeshFwdDecl(Ty, FDContext);
    return T;
  }

  return CreateTypeDefinition(Ty);
}


// ------ CreateTypeDefinition
//
llvm::DIType CGDebugInfo::CreateTypeDefinition(const StructuredMeshType *Ty) {
  StructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Meshes can be recursive.  To handle them, we first generate a
  // debug descriptor for the mesh as a forward declaration.  Then
  // (if it is a definition) we go through and get debug information
  // for all of its members.  Finally, we create a descriptor for the
  // complete type (which may refer to forward decl if it is recursive)
  // and replace all uses of the forward declaration with the final
  // definition.

  llvm::DICompositeType FwdDecl(getOrCreateLimitedType(Ty, DefUnit));
  assert(FwdDecl.isCompositeType() &&
    "The debug type of StructuredMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the mesh on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[MD] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;
  // what about nested types?

  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);

  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements);

  RegionMap[MD] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}


// TODO: Currently used for context chains when limiting debug info.
llvm::DICompositeType
CGDebugInfo::CreateLimitedType(const StructuredMeshType *Ty) {
  StructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  llvm::DIDescriptor MDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));

  // If we ended up creating the type during the context chain construction,
  // just return that.
  // FIXME: this could be dealt with better if the type was recorded as
  // completed before we started this (see the CompletedTypeCache usage in
  // CGDebugInfo::CreateTypeDefinition(const MeshType*) - that would need to
  // be pushed to before context creation, but after it was known to be
  // destined for completion (might still have an issue if this caller only
  // required a declaration but the context construction ended up creating a
  // definition)
  llvm::DICompositeType T(
      getTypeOrNull(CGM.getContext().getStructuredMeshType(MD)));

  if (T && (!T.isForwardDecl() || !MD->getDefinition()))
      return T;

  // If this is just a forward declaration, construct an appropriately
  // marked node and just return it.
  if (!MD->getDefinition())
    return getOrCreateMeshFwdDecl(Ty, MDContext);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DICompositeType RealDecl;

  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);

  RealDecl = DBuilder.createStructType(MDContext, MDName, DefUnit, Line,
                                       Size, Align, 0, llvm::DIType(),
                                       llvm::DIArray(), 0, 0, FullName);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(RealDecl);
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl;

  return RealDecl;
}

// Creates a forward declaration for a unstructured mesh the given context.
llvm::DICompositeType
CGDebugInfo::getOrCreateMeshFwdDecl(const StructuredMeshType *Ty,
                                    llvm::DIDescriptor Ctx) {
  const StructuredMeshDecl *MD = Ty->getDecl();
  if (llvm::DIType T = getTypeOrNull(
        CGM.getContext().getStructuredMeshType(MD)))
    return llvm::DICompositeType(T);

  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;

  // Create the type.
  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);
  return DBuilder.createForwardDecl(Tag, MDName, Ctx, DefUnit, Line, 0, 0, 0,
                                    FullName);
}

/// getOrCreateLimitedType - Get the type from the cache or create a new
/// limited type if necessary.
llvm::DIType CGDebugInfo::getOrCreateLimitedType(const StructuredMeshType *Ty,
                                                 llvm::DIFile Unit) {
  QualType QTy(Ty, 0);

  llvm::DICompositeType T(getTypeOrNull(QTy));

  // We may have cached a forward decl when we could have created
  // a non-forward decl. Go ahead and create a non-forward decl
  // now.
  if (T && !T.isForwardDecl()) return T;

  // Otherwise create the type.
  llvm::DICompositeType Res = CreateLimitedType(Ty);

  // Propagate members from the declaration to the definition
  // CreateType(const MeshType*) will overwrite this with the members in the
  // correct order if the full type is needed.
  Res.setTypeArray(T.getTypeArray());

  if (T && T.isForwardDecl())
    ReplaceMap.push_back(
        std::make_pair(QTy.getAsOpaquePtr(), static_cast<llvm::Value *>(T)));

  // And update the type cache.
  TypeCache[QTy.getAsOpaquePtr()] = Res;
  return Res;
}


//===----------------------------------------------------------------------===//
// Unstructured Mesh debug support
//===----------------------------------------------------------------------===//

// ----- CreateType
//
llvm::DIType CGDebugInfo::CreateType(const UnstructuredMeshType *Ty) {

  UnstructuredMeshDecl *MD = Ty->getDecl();

  // Always emit declarations for types that aren't required to be
  // complete when in limit-debug-info mode.  If the type is later
  // found to be required to be complete this declaration will be
  // upgraded to a definition by 'completeRequiredType'.
  llvm::DICompositeType T(getTypeOrNull(QualType(Ty, 0)));

  // If we're already emitted the type just use that, even if it is only a
  // declaration.  The completeType(), completeRequiredType(), and
  // completeClassData() callbacks will handle promoting the declaration
  // to a definition.
  if (T || (DebugKind <= CodeGenOptions::LimitedDebugInfo &&
     // Under -flimit-debug-info emit only a declaration unless the type is
    // required to be complete.
    !MD->isCompleteDefinitionRequired() && CGM.getLangOpts().CPlusPlus)) {

    llvm::DIDescriptor FDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));
    if (!T)
      T = getOrCreateMeshFwdDecl(Ty, FDContext);
    return T;
  }

  return CreateTypeDefinition(Ty);
}

// TODO: Currently used for context chains when limiting debug info.
llvm::DICompositeType
CGDebugInfo::CreateLimitedType(const UnstructuredMeshType *Ty) {
  UnstructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  llvm::DIDescriptor MDContext =
      getContextDescriptor(cast<Decl>(MD->getDeclContext()));

  // If we ended up creating the type during the context chain construction,
  // just return that.
  // FIXME: this could be dealt with better if the type was recorded as
  // completed before we started this (see the CompletedTypeCache usage in
  // CGDebugInfo::CreateTypeDefinition(const MeshType*) - that would need to
  // be pushed to before context creation, but after it was known to be
  // destined for completion (might still have an issue if this caller only
  // required a declaration but the context construction ended up creating a
  // definition)
  llvm::DICompositeType T(
      getTypeOrNull(CGM.getContext().getUnstructuredMeshType(MD)));
  if (T && (!T.isForwardDecl() || !MD->getDefinition()))
      return T;

  // If this is just a forward declaration, construct an appropriately
  // marked node and just return it.
  if (!MD->getDefinition())
    return getOrCreateMeshFwdDecl(Ty, MDContext);

  uint64_t Size = CGM.getContext().getTypeSize(Ty);
  uint64_t Align = CGM.getContext().getTypeAlign(Ty);
  llvm::DICompositeType RealDecl;

  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);

  RealDecl = DBuilder.createStructType(MDContext, MDName, DefUnit, Line,
                                       Size, Align, 0, llvm::DIType(),
                                       llvm::DIArray(), 0, 0, FullName);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(RealDecl);
  TypeCache[QualType(Ty, 0).getAsOpaquePtr()] = RealDecl;

  return RealDecl;
}


/// getOrCreateLimitedType - Get the type from the cache or create a new
/// limited type if necessary.
llvm::DIType CGDebugInfo::getOrCreateLimitedType(const UnstructuredMeshType *Ty,
                                                 llvm::DIFile Unit) {
  QualType QTy(Ty, 0);

  llvm::DICompositeType T(getTypeOrNull(QTy));

  // We may have cached a forward decl when we could have created
  // a non-forward decl. Go ahead and create a non-forward decl
  // now.
  if (T && !T.isForwardDecl()) return T;

  // Otherwise create the type.
  llvm::DICompositeType Res = CreateLimitedType(Ty);

  // Propagate members from the declaration to the definition
  // CreateType(const MeshType*) will overwrite this with the members in the
  // correct order if the full type is needed.
  Res.setTypeArray(T.getTypeArray());

  if (T && T.isForwardDecl())
    ReplaceMap.push_back(
        std::make_pair(QTy.getAsOpaquePtr(), static_cast<llvm::Value *>(T)));

  // And update the type cache.
  TypeCache[QTy.getAsOpaquePtr()] = Res;
  return Res;
}


// ----- CreateTypeDefinition
//
llvm::DIType CGDebugInfo::CreateTypeDefinition(const UnstructuredMeshType *Ty) {
  UnstructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Meshes can be recursive.  To handle them, we first generate a
  // debug descriptor for the mesh as a forward declaration.  Then
  // (if it is a definition) we go through and get debug information
  // for all of its members.  Finally, we create a descriptor for the
  // complete type (which may refer to forward decl if it is recursive)
  // and replace all uses of the forward declaration with the final
  // definition.

  llvm::DICompositeType FwdDecl(getOrCreateLimitedType(Ty, DefUnit));
  assert(FwdDecl.isCompositeType() &&
    "The debug type of UnstructuredMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the mesh on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[MD] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;
  // what about nested types?

  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);

  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements);

  RegionMap[MD] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}

// Creates a forward declaration for a RecordDecl in the given context.
llvm::DICompositeType
CGDebugInfo::getOrCreateMeshFwdDecl(const UnstructuredMeshType *Ty,
                                    llvm::DIDescriptor Ctx) {
  const UnstructuredMeshDecl *MD = Ty->getDecl();
  if (llvm::DIType T = getTypeOrNull(
        CGM.getContext().getUnstructuredMeshType(MD)))
    return llvm::DICompositeType(T);
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());
  unsigned Line = getLineNumber(MD->getLocation());
  StringRef MDName = MD->getName();

  unsigned Tag = llvm::dwarf::DW_TAG_structure_type;
  // Create the type.
  SmallString<256> FullName = getUniqueMeshTypeName(Ty, CGM, TheCU);
  return DBuilder.createForwardDecl(Tag, MDName, Ctx, DefUnit, Line, 0, 0, 0,
                                    FullName);
}






//===----------------------------------------------------------------------===//
// Mesh & field support routines
//===----------------------------------------------------------------------===//

// SC_TODO -- we use LLVM's record type flags below (RecordTy).  Need
// to ponder if we need to extend LLVM's types to flag meshes as
// special to get debugging to work better...

/// CollectMeshFields - A helper function to collect debug info for
/// mesh fields. This is used while creating debug info entry for a
/// Mesh.
void CGDebugInfo::
CollectMeshFields(const MeshDecl *mesh, llvm::DIFile tunit,
                  SmallVectorImpl<llvm::Value *> &elements,
                  llvm::DIType MeshTy) {

  const ASTMeshLayout &layout = CGM.getContext().getASTMeshLayout(mesh);

  // Field number for non-static fields.
  unsigned fieldNo = 0;

  //const MeshFieldDecl *LastFD = 0;

  // Static and non-static members should appear in the same order as
  // the corresponding declarations in the source program.
  for (MeshDecl::decl_iterator I = mesh->decls_begin(),
       E = mesh->decls_end(); I != E; ++I) {
    if (const VarDecl *V = dyn_cast<VarDecl>(*I))
      CollectMeshStaticField(V, elements, MeshTy);
    else if (MeshFieldDecl *field = dyn_cast<MeshFieldDecl>(*I)) {
      CollectMeshNormalField(field, layout.getFieldOffset(fieldNo),
                               tunit, elements, MeshTy);
        // Bump field number for next field.
        ++fieldNo;
    }
  }
}

/// CollectMeshNormalField - Helper for CollectMeshFields.
void CGDebugInfo::
CollectMeshNormalField(const MeshFieldDecl *field, uint64_t OffsetInBits,
                       llvm::DIFile tunit,
                       SmallVectorImpl<llvm::Value *> &elements,
                       llvm::DIType MeshTy) {
  StringRef name = field->getName();
  QualType type = field->getType();

  uint64_t SizeInBitsOverride = 0;
  if (field->isBitField()) {
    SizeInBitsOverride = field->getBitWidthValue(CGM.getContext());
    assert(SizeInBitsOverride && "found named 0-width bitfield");
  }

  llvm::DIType fieldType;
  fieldType = createFieldType(name, type, SizeInBitsOverride,
                              field->getLocation(), field->getAccess(),
                              OffsetInBits, tunit, MeshTy);

  elements.push_back(fieldType);
}

/// CollectMeshStaticField - Helper for CollectMeshFields.
void CGDebugInfo::
CollectMeshStaticField(const VarDecl *Var,
                       SmallVectorImpl<llvm::Value *> &elements,
                       llvm::DIType MeshTy) {
  // Create the descriptor for the static variable, with or without
  // constant initializers.
  llvm::DIFile VUnit = getOrCreateFile(Var->getLocation());
  llvm::DIType VTy = getOrCreateType(Var->getType(), VUnit);

  // Do not describe enums as static members.
  if (VTy.getTag() == llvm::dwarf::DW_TAG_enumeration_type)
    return;

  unsigned LineNumber = getLineNumber(Var->getLocation());
  StringRef VName = Var->getName();
  llvm::Constant *C = NULL;
  if (Var->getInit()) {
    const APValue *Value = Var->evaluateValue();
    if (Value) {
      if (Value->isInt())
        C = llvm::ConstantInt::get(CGM.getLLVMContext(), Value->getInt());
      if (Value->isFloat())
        C = llvm::ConstantFP::get(CGM.getLLVMContext(), Value->getFloat());
    }
  }

  unsigned Flags = 0;
  llvm::DIType GV = DBuilder.createStaticMemberType(MeshTy, VName, VUnit,
                                                    LineNumber, VTy, Flags, C);
  elements.push_back(GV);
  StaticDataMemberCache[Var->getCanonicalDecl()] = llvm::WeakVH(GV);
}
