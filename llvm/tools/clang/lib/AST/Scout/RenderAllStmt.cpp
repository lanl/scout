
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

RenderAllStmt::RenderAllStmt(ASTContext &C, ForAllType T, 
	                         const MeshType *MT,
                             IdentifierInfo* LII, IdentifierInfo* MII, 
                             VarDecl* MVD, Expr *Op, Stmt *Body, 
                             BlockExpr *Block, SourceLocation RL, 
                             SourceLocation RP, SourceLocation LP)
  : ForAllStmt(RenderAllStmtClass, C, T, MT, LII, MII, MVD,
               Op, Body, Block, RL, RP, LP),
    ElementColor(0),
    ElementRadius(0)
{
  
}

