
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

/*
ForAllArrayStmt::ForAllArrayStmt(ASTContext &C,
                                 SourceLocation FAL,
                                 Stmt* Body,
                                 BlockExpr* Block)
: ForAllStmt(ForAllArrayStmtClass, C, 
             ForAllStmt::Array, 0, 0, 0, 0, 0, Body, 
             Block, FAL, SourceLocation(), SourceLocation()),
XInductionVarII(0),
YInductionVarII(0),
ZInductionVarII(0)
{
  setBody(Body);
}
*/
