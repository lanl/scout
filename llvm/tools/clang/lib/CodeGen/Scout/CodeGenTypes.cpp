/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
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

#include "CodeGenTypes.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGScoutRuntime.h"
#include "CGRecordLayout.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "clang/AST/Scout/MeshLayout.h"
#include "Scout/CGMeshLayout.h"

using namespace clang;
using namespace CodeGen;

static bool
isSafeToConvert(QualType T, CodeGenTypes &CGT,
                llvm::SmallPtrSet<const MeshDecl*, 16> &AlreadyChecked);


// Handle the cases for converting the following scout mesh
// types:
//  Type::UniformMesh
//  Type::StructuredMesh
//  Type::RectlinearMesh
//  Type::UnstructuredMesh
//
// put fields first and then the mesh dimensions (if required)
llvm::Type *CodeGenTypes::ConvertScoutMeshType(QualType T) {

  const Type *Ty = T.getTypePtr();

  // Implemented as a struct of n-dimensional array's type.
  MeshDecl *mesh = cast<MeshType>(Ty)->getDecl();
  const MeshType *meshType =  cast<MeshType>(T.getCanonicalType().getTypePtr());
  MeshType::MeshDimensions dims = meshType->dimensions();
  llvm::StringRef meshName = mesh->getName();

  unsigned int rank = 0;
  for(unsigned int i = 0; i < dims.size(); ++i) {
    if (dims[i] != 0) {
      rank++;
    }
  }

  // As we lower our mesh types to llvm we also add a set of metadata relevant to the
  // mesh so we can use it within LLVM during optimization and code generation. 
  llvm::NamedMDNode *MeshMD;
  MeshMD = CGM.getModule().getOrInsertNamedMetadata("scout.meshes");
  SmallVector<llvm::Value*, 16> MeshInfoMD;

  // Metadata - mesh type (uniform, rectilinear, etc).
  llvm::MDString *MDKind = llvm::MDString::get(getLLVMContext(), meshType->getTypeClassName());
  MeshInfoMD.push_back(MDKind);
  llvm::MDString *MDName = llvm::MDString::get(getLLVMContext(), meshName);
  MeshInfoMD.push_back(MDName);
  // Metadata - mesh dims.   
  llvm::IntegerType *Int32Ty = llvm::Type::getInt32Ty(getLLVMContext());
  llvm::Value *MDRank = llvm::ConstantInt::get(Int32Ty, rank);
  MeshInfoMD.push_back(MDRank);
  
  typedef llvm::ArrayType ArrayTy;
  MeshDecl::field_iterator it     = mesh->field_begin();
  MeshDecl::field_iterator it_end = mesh->field_end();

  std::vector< llvm::Type * > eltTys;  
  std::vector<llvm::Type*> MeshCellFields; 
  std::vector<llvm::Type*> MeshVertexFields;
  std::vector<llvm::Type*> MeshEdgeFields;
  std::vector<llvm::Type*> MeshFaceFields;     

  for( ; it != it_end; ++it) {
    // Do not generate code for implicit mesh member variables.
    if (! it->isImplicit()) {
      // Identify the type of each mesh member.
      llvm::Type *ty = ConvertType(it->getType());
      uint64_t numElts = 1;

      // Transform each member type into a pointer.
      for(unsigned i = 0; i < rank; ++i) {
        llvm::APSInt result;
        dims[i]->EvaluateAsInt(result, Context);
        numElts *= result.getSExtValue();
        eltTys.push_back(llvm::PointerType::getUnqual(ty));

        if (it->isCellLocated()) {
          MDName = llvm::MDString::get(getLLVMContext(), it->getName());          
          llvm::Value *MField = llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(MDName));
          MeshInfoMD.push_back(MField);
        } else if (it->isVertexLocated()) {

        } else if (it->isEdgeLocated()) {

        } else if (it->isFaceLocated()) {

        } else {
          llvm_unreachable("field has unknown location!");
        }
      }
    }
  }

  MeshMD->addOperand(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(MeshInfoMD)));
  
  if (isa<UniformMeshType>(meshType) || isa<RectilinearMeshType>(meshType)) {
    // put width/height/depth after fields
    for(size_t i = 0; i < rank; ++i) {
      eltTys.push_back(llvm::IntegerType::get(getLLVMContext(), 32));
    }
  }

  typedef llvm::ArrayRef< llvm::Type * > Array;
  typedef llvm::StructType StructTy;

  // Construct a struct of array's type.
  StructTy *structTy = StructTy::create(getLLVMContext(),
                                        Array(eltTys),
                                        meshName,
                                        false);

  return structTy;
}

//based on the way record works but untested.
void CodeGenTypes::UpdateCompletedType(const MeshDecl *MD) {

   if (MD->isDependentType()) return;

   // Only complete it if we converted it already.  If we haven't converted it
   // yet, we'll just do it lazily.
   if (MeshDeclTypes.count(Context.getMeshDeclType(MD).getTypePtr()))
     ConvertMeshDeclType(MD);

   // If necessary, provide the full definition of a type only used with a
   // declaration so far.
   if (CGDebugInfo *DI = CGM.getModuleDebugInfo())
     DI->completeType(MD);
}

void CodeGenTypes::addMeshTypeName(const MeshDecl *RD,
                                     llvm::StructType *Ty,
                                     StringRef suffix) {
  SmallString<256> TypeName;
  llvm::raw_svector_ostream OS(TypeName);
  OS << RD->getKindName() << '.';

  // Name the codegen type after the typedef name
  // if there is no tag type name available
  if (RD->getIdentifier()) {
    // FIXME: We should not have to check for a null decl context here.
    // Right now we do it because the implicit Obj-C decls don't have one.
    if (RD->getDeclContext())
      RD->printQualifiedName(OS);
    else
      RD->printName(OS);
  } else if (const TypedefNameDecl *TDD = RD->getTypedefNameForAnonDecl()) {
    // FIXME: We should not have to check for a null decl context here.
    // Right now we do it because the implicit Obj-C decls don't have one.
    if (TDD->getDeclContext())
      TDD->printQualifiedName(OS);
    else
      TDD->printName(OS);
  } else
    OS << "anon";

  if (!suffix.empty())
    OS << suffix;

  Ty->setName(OS.str());
}

/// isMeshLayoutComplete - Return true if the specified mesh is already
/// completely laid out.
bool CodeGenTypes::isMeshLayoutComplete(const Type *Ty) const {
  llvm::DenseMap<const Type*, llvm::StructType *>::const_iterator I =
  MeshDeclTypes.find(Ty);
  return I != MeshDeclTypes.end() && !I->second->isOpaque();
}


/// isSafeToConvert - Return true if it is safe to convert the specified mesh
/// decl to IR and lay it out, false if doing so would cause us to get into a
/// recursive compilation mess.
static bool
isSafeToConvert(const MeshDecl *MD, CodeGenTypes &CGT,
                llvm::SmallPtrSet<const MeshDecl*, 16> &AlreadyChecked) {
  // If we have already checked this type (maybe the same type is used by-value
  // multiple times in multiple structure fields, don't check again.
  if (!AlreadyChecked.insert(MD)) return true;

  const Type *Key = CGT.getContext().getMeshDeclType(MD).getTypePtr();

  // If this type is already laid out, converting it is a no-op.
  if (CGT.isMeshLayoutComplete(Key)) return true;

  // If this type is currently being laid out, we can't recursively compile it.
  if (CGT.isMeshBeingLaidOut(Key))
    return false;

   // If this type would require laying out members that are currently being laid
  // out, don't do it.
  for (MeshDecl::field_iterator I = MD->field_begin(),
       E = MD->field_end(); I != E; ++I)
    if (!isSafeToConvert(I->getType(), CGT, AlreadyChecked))
      return false;

  // If there are no problems, lets do it.
  return true;
}

/// isSafeToConvert - Return true if it is safe to convert the specified mesh
/// decl to IR and lay it out, false if doing so would cause us to get into a
/// recursive compilation mess.
static bool isSafeToConvert(const MeshDecl *MD, CodeGenTypes &CGT) {
  // If no meshes are being laid out, we can certainly do this one.
  if (CGT.noMeshesBeingLaidOut()) return true;

  llvm::SmallPtrSet<const MeshDecl*, 16> AlreadyChecked;
  return isSafeToConvert(MD, CGT, AlreadyChecked);
}

/// isSafeToConvert - Return true if it is safe to convert this mesh field
/// type, which requires the mesh elements contained by-value to all be
/// recursively safe to convert.
static bool
isSafeToConvert(QualType T, CodeGenTypes &CGT,
                llvm::SmallPtrSet<const MeshDecl*, 16> &AlreadyChecked) {
  T = T.getCanonicalType();

  // If this is a record, check it.
  if (const MeshType *MT = dyn_cast<MeshType>(T))
    return isSafeToConvert(MT->getDecl(), CGT, AlreadyChecked);

  // SC_TODO -- do we really want to return here or assert?
  //assert(false && "Non-mesh type being checked for conversion.");

  // Otherwise, there is no concern about transforming this.  We only care about
  // things that are contained by-value in a structure that can have another
  // mesh as a member.
  return true;
}

/// ConvertMeshDeclType - Layout a mesh decl type.
llvm::StructType *CodeGenTypes::ConvertMeshDeclType(const MeshDecl *MD) {
  // TagDecl's are not necessarily unique, instead use the (clang)
  // type connected to the decl.
  const Type *Key = Context.getMeshDeclType(MD).getTypePtr();

  llvm::StructType *&Entry = MeshDeclTypes[Key];

  // If we don't have a StructType at all yet, create the forward declaration.
  if (Entry == 0) {
    Entry = llvm::StructType::create(getLLVMContext());
    addMeshTypeName(MD, Entry, "");
  }
  llvm::StructType *Ty = Entry;

  // If this is still a forward declaration, or the LLVM type is already
  // complete, there's nothing more to do.
  MD = MD->getDefinition();
  if (MD == 0 || !MD->isCompleteDefinition() || !Ty->isOpaque())
    return Ty;

  // If converting this type would cause us to infinitely loop, don't do it!
  if (!isSafeToConvert(MD, *this)) {
    DeferredMeshes.push_back(MD);
    return Ty;
  }

  // Okay, this is a definition of a type.  Compile the implementation now.
  bool InsertResult = MeshesBeingLaidOut.insert(Key); (void)InsertResult;
  assert(InsertResult && "Recursively compiling a mesh?");

   // Layout fields.
  CGMeshLayout *Layout = ComputeMeshLayout(MD, Ty);
  CGMeshLayouts[Key] = Layout;

  // We're done laying out this mesh.
  bool EraseResult = MeshesBeingLaidOut.erase(Key); (void)EraseResult;
  assert(EraseResult && "struct not in RecordsBeingLaidOut set?");

  // If this struct blocked a FunctionType conversion, then recompute whatever
  // was derived from that.
  // FIXME: This is hugely overconservative.
  if (SkippedLayout)
    TypeCache.clear();

  // If we're done converting the outer-most record, then convert any deferred
  // structs as well.
  if (MeshesBeingLaidOut.empty())
    while (!DeferredMeshes.empty())
      ConvertMeshDeclType(DeferredMeshes.pop_back_val());

  return Ty;
}

/// getCGMeshLayout - Return mesh layout info for the given mesh decl.
const CGMeshLayout &
CodeGenTypes::getCGMeshLayout(const MeshDecl *MD) {
  const Type *Key = Context.getMeshDeclType(MD).getTypePtr();

  const CGMeshLayout *Layout = CGMeshLayouts.lookup(Key);
  if (!Layout) {
    // Compute the type information.
    ConvertMeshDeclType(MD);
    // Now try again.
    Layout = CGMeshLayouts.lookup(Key);
  }

  assert(Layout && "Unable to find mesh layout");
  return *Layout;
}


llvm::Type *CodeGenTypes::ConvertScoutRenderTargetType(QualType T) {
  const Type *Ty = T.getTypePtr();  
  return CGM.getScoutRuntime().convertScoutSpecificType(Ty);
}
