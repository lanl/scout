//===----------------------------------------------------------------------===//
//
// ndm - This file defines the Scout Stmt interfaces and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_SCOUT_H
#define LLVM_CLANG_AST_STMT_SCOUT_H

#include "clang/AST/Stmt.h"

namespace clang{

class ForAllStmt : public Stmt {

public:
  enum ForAllType{
    Cells,
    Vertices
  };
  
private:
  
  // IND is the induction variable: forall cells IND ....
  //                             OR forall IND => sten(c)
  // MESH is the mesh identifier: for all cells c of MESH
  // OP is the operation or condition: ... myMesh '(' OP ')' { ...
  // BODY is compound stmts : '{' BODY '}' 
  
  enum {IND, MESH, OP, BODY, END_EXPR};
    
  ForAllType Type;
  
  Stmt* SubExprs[END_EXPR];
  
  SourceLocation ForAllLoc;
  SourceLocation LParenLoc, RParenLoc;
  
public:
  ForAllStmt(ASTContext &C, ForAllType Type, Expr *Ind, Expr *Mesh, Expr *Op, 
             Stmt *Body, SourceLocation FL, SourceLocation LP, SourceLocation RP);
  
  explicit ForAllStmt(EmptyShell Empty) : Stmt(ForAllStmtClass, Empty) { }

  
  ForAllType getType(){
    return Type;
  }
  
  void setType(ForAllType T){
    Type = T;
  }

  Expr* getInd();
  const Expr* getInd() const;
  void setInd(Expr* expr);
  
  Expr* getMesh(){
    return reinterpret_cast<Expr*>(SubExprs[MESH]);
  }
  
  const Expr* getMesh() const{
    return reinterpret_cast<Expr*>(SubExprs[MESH]);
  }
  
  void setMesh(Expr* M){
    SubExprs[MESH] = reinterpret_cast<Stmt*>(M);
  }
  
  Expr* getOp(){
    return reinterpret_cast<Expr*>(SubExprs[OP]);
  }
  
  const Expr* getOp() const{
    return reinterpret_cast<Expr*>(SubExprs[OP]);
  }
  
  void setOp(Expr* O){
    SubExprs[OP] = reinterpret_cast<Stmt*>(O);
  }
  
  Stmt* getBody(){
    return SubExprs[BODY];
  }
  
  const Stmt* getBody() const{
    return SubExprs[BODY];
  }
  
  void setBody(Stmt* B){
    SubExprs[BODY] = reinterpret_cast<Stmt*>(B);
  }
  
  SourceLocation getForAllLoc() const { return ForAllLoc; }
  void setForAllLoc(SourceLocation L) { ForAllLoc = L; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ForAllStmtClass;
  }
  
  static bool classof(const ForAllStmt *) { return true; }
  
  SourceRange getSourceRange() const {
    return SourceRange(ForAllLoc, SubExprs[BODY]->getLocEnd());
  }
  
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
  
};

class RenderAllStmt : public Stmt {
public:
  enum RenderAllType{
    Cells,
    Faces,
    Edges
  };
  
private:
  
  
  // IND is the induction variable: renderall cells IND ...
  // MESH is the mesh identifier: renderall cells c of MESH ...
  // BODY is compound stmts : '{' BODY '}' 
  
  enum {IND, MESH, BODY, END_EXPR};
    
  Stmt* SubExprs[END_EXPR];
  
  SourceLocation RenderAllLoc;
  
  RenderAllType Type;
  
public:
  
  RenderAllStmt(ASTContext &C, RenderAllType Type, Expr *Ind, Expr *Mesh,
                Stmt *Body, SourceLocation RL);
  
  explicit RenderAllStmt(EmptyShell Empty) 
  : Stmt(RenderAllStmtClass, Empty) { }
  
  RenderAllType getType(){
    return Type;
  }

  void setType(RenderAllType T){
    Type = T;
  }
  
  Expr* getInd();
  const Expr* getInd() const;
  void setInd(Expr* expr);
  
  Expr* getMesh(){
    return reinterpret_cast<Expr*>(SubExprs[MESH]);
  }
  
  const Expr* getMesh() const{
    return reinterpret_cast<Expr*>(SubExprs[MESH]);
  }
  
  void setMesh(Expr* M){
    SubExprs[MESH] = reinterpret_cast<Stmt*>(M);
  }
  
  Stmt* getBody(){
    return SubExprs[BODY];
  }
  
  const Stmt* getBody() const{
    return SubExprs[BODY];
  }
  
  void setBody(Stmt* B){
    SubExprs[BODY] = reinterpret_cast<Stmt*>(B);
  }
  
  SourceLocation getRenderAllLoc() const { return RenderAllLoc; }
  void setRenderAllLoc(SourceLocation L) { RenderAllLoc = L; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == RenderAllStmtClass;
  }
  
  static bool classof(const RenderAllStmt *) { return true; }
  
  SourceRange getSourceRange() const {
    return SourceRange(RenderAllLoc, SubExprs[BODY]->getLocEnd());
  }
  
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
  
};

} // end namespace clang
  
#endif // LLVM_CLANG_AST_STMT_SCOUT_H
