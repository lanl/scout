#include "CodeGenTypes.h"
#include "CGCXXABI.h"
#include "CGCall.h"
#include "CGOpenCLRuntime.h"
#include "CGRecordLayout.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
using namespace clang;

using namespace CodeGen;


// Handle the cases for converting the following scout mesh
// types: 
//  Type::UniformMesh
//  Type::StructuredMesh 
//  Type::RectlinearMesh
//  Type::UnstructuredMesh
//
llvm::Type *CodeGenTypes::ConvertScoutMeshType(QualType T) {

  const Type *Ty = T.getTypePtr();
  
  // Implemented as a struct of n-dimensional array's type.
  MeshDecl *mesh = cast<MeshType>(Ty)->getDecl();
  MeshType::MeshDimensions dims;
  dims = cast<MeshType>(T.getCanonicalType().getTypePtr())->dimensions();

  unsigned int rank = 0;
  for(unsigned int i = 0; i < dims.size(); ++i) {
    if (dims[i] != 0)
      rank++;
  }
  
  llvm::StringRef meshName = mesh->getName();
  typedef llvm::ArrayType ArrayTy;
  MeshDecl::field_iterator it     = mesh->field_begin();
  MeshDecl::field_iterator it_end = mesh->field_end();

  std::vector< llvm::Type * > eltTys;
  // mesh_flags__, width, height, depth
  for(size_t i = 0; i < rank+1 /* 4 */; ++i) {
    eltTys.push_back(llvm::IntegerType::get(getLLVMContext(), 32));
  }

  for( ; it != it_end; ++it) {
    // Do not generate code for implicit mesh member variables.
    if (! it->isImplicit()) {
      // Identify the type of each mesh member.
      llvm::Type *ty = ConvertType(it->getType());
      uint64_t numElts = 1;

      // Transform each member type into a pointer.
      for(unsigned i = 0; i < rank /*dims.size()*/; ++i) {
        llvm::APSInt result;
        dims[i]->EvaluateAsInt(result, Context);
        numElts *= result.getSExtValue();
      }

      eltTys.push_back(llvm::PointerType::getUnqual(ty));
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
