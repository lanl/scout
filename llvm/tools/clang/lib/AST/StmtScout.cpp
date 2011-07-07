//===----------------------------------------------------------------------===//
//
// ndm - This file implements the Scout Stmt AST subclasses.
//
//===----------------------------------------------------------------------===//


#include "clang/AST/StmtScout.h"

using namespace clang;

ForAllStmt::ForAllStmt(ASTContext &C, ForAllType T, Expr *Ind, Expr *Mesh,
                       Expr *Op, Stmt *Body, SourceLocation FL, 
                       SourceLocation LP, SourceLocation RP)
: Stmt(ForAllStmtClass), 
Type(T),
ForAllLoc(FL),
LParenLoc(LP),
RParenLoc(RP){

}

Expr* ForAllStmt::getInd(){
  return reinterpret_cast<Expr*>(SubExprs[IND]);
}

const Expr* ForAllStmt::getInd() const{
  return reinterpret_cast<Expr*>(SubExprs[IND]);
}

void ForAllStmt::setInd(Expr* expr){
  SubExprs[IND] = reinterpret_cast<Stmt*>(expr);
}

RenderAllStmt::RenderAllStmt(ASTContext &C, RenderAllType T, Expr *Ind, Expr *Mesh, 
                             Stmt *Body, SourceLocation FL)
: Stmt(RenderAllStmtClass), 
Type(T),
RenderAllLoc(FL){
  
}

Expr* RenderAllStmt::getInd(){
  return reinterpret_cast<Expr*>(SubExprs[IND]);
}

const Expr* RenderAllStmt::getInd() const{
  return reinterpret_cast<Expr*>(SubExprs[IND]);
}

void RenderAllStmt::setInd(Expr* expr){
  SubExprs[IND] = reinterpret_cast<Stmt*>(expr);
}
