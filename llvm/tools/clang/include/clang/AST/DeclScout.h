//===----------------------------------------------------------------------===//
//
//  ndm - This file defines Scout Decl subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECL_SCOUT_H
#define LLVM_CLANG_AST_DECL_SCOUT_H

#include "clang/AST/Decl.h"

namespace clang{

class MeshDecl : public TypeDecl, public DeclContext{
public:
  
  typedef llvm::SmallVector<Expr*, 3> MeshDimensionVec;
  
private:
  
  bool IsDefinition : 1;
  bool IsBeingDefined : 1;
  SourceLocation RBraceLoc;
  MeshDimensionVec Dimensions;

protected:
  MeshDecl(Kind DK, DeclContext* DC,
           SourceLocation L, SourceLocation StartL,
           IdentifierInfo* Id, MeshDecl* PrevDecl)
  : TypeDecl(DK, DC, L, Id, StartL),
  DeclContext(DK){
    IsDefinition = false;
    IsBeingDefined = false;
  }  
  
public:

  static MeshDecl* Create(ASTContext& C, Kind DK, DeclContext* DC, 
                          SourceLocation StartLoc, SourceLocation IdLoc, 
                          IdentifierInfo* Id, MeshDecl* PrevDecl);
  
  const MeshDimensionVec& dimensions() const{
    return Dimensions;
  }
  
  void setDimensions(const MeshDimensionVec& dimensions){
    Dimensions = dimensions;
  }
  
  void completeDefinition();
  
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }
  
  SourceLocation getInnerLocStart() const { return getLocStart(); }
  
  SourceLocation getOuterLocStart() const;
  virtual SourceRange getSourceRange() const;
  
  bool isThisDeclarationADefinition() const {
    return isDefinition();
  }
  
  bool isDefinition() const {
    return IsDefinition;
  }
  
  bool isBeingDefined() const {
    return IsBeingDefined;
  }
  
  void startDefinition();
  
  MeshDecl* getDefinition() const;
  
  typedef specific_decl_iterator<FieldDecl> field_iterator;
  
  field_iterator field_begin() const;
  
  field_iterator field_end() const{
    return field_iterator(decl_iterator());
  }
  
  bool field_empty() const{
    return field_begin() == field_end();
  }

  // ndm - TODO is this needed? If so, finish implementation.
  NestedNameSpecifierLoc getQualifierLoc() const {
    return NestedNameSpecifierLoc();
  }
    
  static bool classof(const Decl* D) { return classofKind(D->getKind()); }
  static bool classof(const MeshDecl* D) { return true; }
  static bool classofKind(Kind K) { return K == Mesh; }
  
  static DeclContext* castToDeclContext(const MeshDecl* D){
    return static_cast<DeclContext*>(const_cast<MeshDecl*>(D));
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

} // end namespace clang

#endif // #ifndef LLVM_CLANG_AST_DECL_SCOUT_H
