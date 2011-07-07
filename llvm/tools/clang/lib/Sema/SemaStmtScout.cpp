//===----------------------------------------------------------------------===//
//
// ndm - This file implements semantic analysis and AST building for 
// Scout statements.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"

using namespace clang;
using namespace sema;

// ndm - TODO - implement

StmtResult Sema::ActOnForAllStmt(SourceLocation ForAllLoc,
                                 ForAllStmt::ForAllType Type,
                                 Expr* Ind, Expr* Mesh,
                                 SourceLocation LParenLoc,
                                 Expr* Op, SourceLocation RParenLoc,
                                 Stmt* Body){
  
  return Owned(new (Context) ForAllStmt(Context, Type,
                                        Ind, Mesh,
                                        Op, Body, ForAllLoc, LParenLoc,
                                        RParenLoc));
}


StmtResult Sema::ActOnRenderAllStmt(SourceLocation RenderAllLoc,
                                    RenderAllStmt::RenderAllType Type,
                                    Expr* Ind,
                                    Expr* Mesh,
                                    Stmt* Body){
  return Owned(new (Context) RenderAllStmt(Context, Type,
                                           Ind, Mesh,
                                           Body, RenderAllLoc));
}

