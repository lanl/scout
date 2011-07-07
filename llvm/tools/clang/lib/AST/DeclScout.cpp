//===----------------------------------------------------------------------===//
//
// ndm - This file implements the Scout Decl subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclScout.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// MeshDecl Implementation
//===----------------------------------------------------------------------===//

//
//MeshDecl(Kind DK, DeclContext* DC,
//         SourceLocation L, IdentifierInfo* Id,
//         SourceLocation StartL)

MeshDecl* MeshDecl::Create(ASTContext& C, Kind K, DeclContext* DC,
                           SourceLocation StartLoc, SourceLocation IdLoc,
                           IdentifierInfo* Id, MeshDecl* PrevDecl){
  
  MeshDecl* M = new (C) MeshDecl(K, DC, StartLoc, IdLoc, Id, PrevDecl);
  
  // ndm
  // TODO - what does this do? does it apply to mesh definitions?
  //C.getTypeDeclType(M, PrevDecl);
  return M;
}

SourceLocation MeshDecl::getOuterLocStart() const {

}

SourceRange MeshDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(getOuterLocStart(), E);
}

void MeshDecl::startDefinition() {
  IsBeingDefined = true;
}

void MeshDecl::completeDefinition() {
  IsDefinition = true;
  IsBeingDefined = false;
}

MeshDecl* MeshDecl::getDefinition() const{
  if(isDefinition()){
    return const_cast<MeshDecl*>(this);
  }

  // ndm - not fully implemented
  
  return 0;
}

MeshDecl::field_iterator MeshDecl::field_begin() const{
  return field_iterator(decl_iterator(FirstDecl));
}

