#include "CGDebugInfo.h"
#include "CGBlocks.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Scout/MeshLayout.h"
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

using namespace clang;
using namespace clang::CodeGen;


llvm::DIType CGDebugInfo::CreateType(const UniformMeshType *Ty) {

  UniformMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the mesh as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the mesh is recursive) and replace all
  // uses of the forward declaration with the final definition.

  llvm::DICompositeType FwdDecl(
      getOrCreateLimitedType(QualType(Ty, 0), DefUnit));
  assert(FwdDecl.Verify() &&
         "The debug type of a UniformMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the struct on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;

  
  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);
  llvm::DIArray TParamsArray;
  
  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements, TParamsArray);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}

llvm::DIType CGDebugInfo::CreateType(const StructuredMeshType *Ty) {

  StructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the mesh as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the mesh is recursive) and replace all
  // uses of the forward declaration with the final definition.

  llvm::DICompositeType FwdDecl(
      getOrCreateLimitedType(QualType(Ty, 0), DefUnit));
  assert(FwdDecl.Verify() &&
         "The debug type of a UniformMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the struct on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;

  
  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);
  llvm::DIArray TParamsArray;
  
  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements, TParamsArray);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}

llvm::DIType CGDebugInfo::CreateType(const RectilinearMeshType *Ty) {

  RectilinearMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the mesh as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the mesh is recursive) and replace all
  // uses of the forward declaration with the final definition.

  llvm::DICompositeType FwdDecl(
      getOrCreateLimitedType(QualType(Ty, 0), DefUnit));
  assert(FwdDecl.Verify() &&
         "The debug type of a UniformMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the struct on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;

  
  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);
  llvm::DIArray TParamsArray;
  
  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements, TParamsArray);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}

llvm::DIType CGDebugInfo::CreateType(const UnstructuredMeshType *Ty) {

  UnstructuredMeshDecl *MD = Ty->getDecl();

  // Get overall information about the mesh type for the debug info.
  llvm::DIFile DefUnit = getOrCreateFile(MD->getLocation());

  // Records and classes and unions can all be recursive.  To handle them, we
  // first generate a debug descriptor for the mesh as a forward declaration.
  // Then (if it is a definition) we go through and get debug info for all of
  // its members.  Finally, we create a descriptor for the complete type (which
  // may refer to the forward decl if the mesh is recursive) and replace all
  // uses of the forward declaration with the final definition.

  llvm::DICompositeType FwdDecl(
      getOrCreateLimitedType(QualType(Ty, 0), DefUnit));
  assert(FwdDecl.Verify() &&
         "The debug type of a UniformMeshType should be a llvm::DICompositeType");

  if (FwdDecl.isForwardDecl())
    return FwdDecl;

  // Push the struct on region stack.
  LexicalBlockStack.push_back(&*FwdDecl);
  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);

  // Add this to the completed-type cache while we're completing it recursively.
  CompletedTypeCache[QualType(Ty, 0).getAsOpaquePtr()] = FwdDecl;

  // Convert all the elements.
  SmallVector<llvm::Value *, 16> EltTys;

  
  // Collect data fields (including static variables and any initializers).
  CollectMeshFields(MD, DefUnit, EltTys, FwdDecl);
  llvm::DIArray TParamsArray;
  
  LexicalBlockStack.pop_back();
  RegionMap.erase(Ty->getDecl());

  llvm::DIArray Elements = DBuilder.getOrCreateArray(EltTys);
  FwdDecl.setTypeArray(Elements, TParamsArray);

  RegionMap[Ty->getDecl()] = llvm::WeakVH(FwdDecl);
  return FwdDecl;
}


// SC_TODO -- we use LLVM's record type flags below (RecordTy).  Need
// to ponder if we need to extend LLVM's types to flag meshes as 
// special to get debugging to work better... 

/// CollectMeshFields - A helper function to collect debug info for
/// mesh fields. This is used while creating debug info entry for a 
/// Mesh.
void CGDebugInfo::
CollectMeshFields(const MeshDecl *mesh, llvm::DIFile tunit,
                  SmallVectorImpl<llvm::Value *> &elements,
                  llvm::DIType RecordTy) {

  const ASTMeshLayout &layout = CGM.getContext().getASTMeshLayout(mesh);

  // Field number for non-static fields.
  unsigned fieldNo = 0;

  //const MeshFieldDecl *LastFD = 0;

  // Static and non-static members should appear in the same order as
  // the corresponding declarations in the source program.
  for (MeshDecl::decl_iterator I = mesh->decls_begin(),
       E = mesh->decls_end(); I != E; ++I) {
    if (const VarDecl *V = dyn_cast<VarDecl>(*I))
      CollectMeshStaticField(V, elements, RecordTy);
    else if (MeshFieldDecl *field = dyn_cast<MeshFieldDecl>(*I)) {
      CollectMeshNormalField(field, layout.getFieldOffset(fieldNo),
                               tunit, elements, RecordTy);
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
                       llvm::DIType RecordTy) {
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
                              OffsetInBits, tunit, RecordTy);

  elements.push_back(fieldType);
}

/// CollectMeshStaticField - Helper for CollectMeshFields.
void CGDebugInfo::
CollectMeshStaticField(const VarDecl *Var,
                       SmallVectorImpl<llvm::Value *> &elements,
                       llvm::DIType RecordTy) {
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
  llvm::DIType GV = DBuilder.createStaticMemberType(RecordTy, VName, VUnit,
                                                    LineNumber, VTy, Flags, C);
  elements.push_back(GV);
  StaticDataMemberCache[Var->getCanonicalDecl()] = llvm::WeakVH(GV);
}
