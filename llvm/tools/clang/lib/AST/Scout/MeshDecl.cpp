
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/scout/MeshDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>

using namespace clang;



SourceRange MeshDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(getOuterLocStart(), E);
}

void MeshDecl::startDefinition() {
  IsBeingDefined = true;
}

void MeshDecl::completeDefinition(ASTContext& C) {

  assert(StructRep && "MeshDecl::completeDefinition: uninitialized StructRep");
  
  MeshFieldDecl* Field = MeshFieldDecl::Create(C, StructRep,
                                               getLocation(),
                                               getLocation(),
                                               &C.Idents.get("mesh_flags__"),
                                               C.UnsignedIntTy,
                                               0,
                                               0,
                                               false,
                                               ICIS_NoInit,
                                               MeshFieldDecl::BuiltIn);
  
  Field->setAccess(AS_public);
  StructRep->addDecl(Field);
  
  Field = MeshFieldDecl::Create(C, StructRep,
                                getLocation(),
                                getLocation(),
                                &C.Idents.get("width"),
                                C.UnsignedIntTy,
                                0,
                                0,
                                false,
                                ICIS_NoInit,
                                MeshFieldDecl::BuiltIn);
  Field->setAccess(AS_public);
  StructRep->addDecl(Field);

  Field = MeshFieldDecl::Create(C, StructRep,
                                getLocation(),
                                getLocation(),
                                &C.Idents.get("height"),
                                C.UnsignedIntTy,
                                0,
                                0,
                                false,
                                ICIS_NoInit,
                                MeshFieldDecl::BuiltIn);
  Field->setAccess(AS_public);
  StructRep->addDecl(Field);
  
  Field = MeshFieldDecl::Create(C, StructRep,
                                getLocation(),
                                getLocation(),
                                &C.Idents.get("depth"),
                                C.UnsignedIntTy,
                                0,
                                0,
                                false,
                                ICIS_NoInit,
                                MeshFieldDecl::BuiltIn);
  Field->setAccess(AS_public);
  StructRep->addDecl(Field);
  
  for(MeshDecl::mesh_field_iterator itr = mesh_field_begin(),
        itrEnd = mesh_field_end();
      itr != itrEnd; ++itr) {

    MeshFieldDecl *field = *itr;
    
    if (! field->isImplicit()) {
      
      Field = MeshFieldDecl::Create(C, StructRep, field->getLocation(),
                                    field->getLocation(),
                                    &C.Idents.get(field->getName()),
                                    C.getPointerType(field->getType()),
                                    0,
                                    0,
                                    false,
                                    ICIS_NoInit,
                                    field->meshLocation());
      
      StructRep->addDecl(Field);
    }
  }
  
  StructRep->completeDefinition();
  
  IsDefinition = true;
  IsBeingDefined = false;
}

MeshDecl* MeshDecl::getDefinition() const{
  if(isDefinition()){
    return const_cast<MeshDecl*>(this);
  }
  
  return 0;
}

MeshDecl::mesh_field_iterator MeshDecl::mesh_field_begin() const{
  return mesh_field_iterator(decl_iterator(FirstDecl));
}

bool MeshDecl::canConvertTo(ASTContext& C, MeshDecl* MD) {
  mesh_field_iterator fromItr = mesh_field_begin();
  for(mesh_field_iterator itr = MD->mesh_field_begin(), itrEnd = MD->mesh_field_end();
      itr != itrEnd; ++itr) {
    
    if(fromItr == mesh_field_end()) { // SC_TODO -- what???  How many mesh_field_ends do we have?
      return false;
    }

    FieldDecl* fromField = *fromItr;
    FieldDecl* toField = *itr;

    if(!C.hasSameUnqualifiedType(fromField->getType(), toField->getType())){
      return false;
    }
    ++fromItr;
  }

  return true;
}

