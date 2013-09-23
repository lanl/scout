
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



ForAllStmt::ForAllStmt(StmtClass SC, ASTContext &C, ForAllType T,
                       const MeshType *MT,IdentifierInfo* LII,
                       IdentifierInfo* MII, VarDecl* MVD, Expr *Op,
                       Stmt *Body, BlockExpr* Block,
                       SourceLocation FL, SourceLocation LP,
                       SourceLocation RP)
  : Stmt(SC), Type(T), meshType(MT),
    ForAllLoc(FL), LParenLoc(LP), RParenLoc(RP),
    MeshII(MII), LoopVariableII(LII), MeshVarDecl(MVD),
    XStart(0), XEnd(0), 
    XStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL)),
    YStart(0), YEnd(0), 
    YStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL)),
    ZStart(0), ZEnd(0),
    ZStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL))
{
#ifdef USE_FORALL_BLOCK
  setBlock(Block);
#endif
  setOp(Op);
  setBody(Body);
}

ForAllStmt::ForAllStmt(ASTContext &C, ForAllType T, const MeshType *MT,
                       IdentifierInfo* LII, IdentifierInfo* MII, VarDecl* MVD,
                       Expr *Op, Stmt *Body, BlockExpr* Block, SourceLocation FL,
                       SourceLocation LP, SourceLocation RP)
  : Stmt(ForAllStmtClass), Type(T), meshType(MT),
    ForAllLoc(FL), LParenLoc(LP), RParenLoc(RP),
    MeshII(MII), LoopVariableII(LII), MeshVarDecl(MVD),
    XStart(0), XEnd(0),
    XStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL)),
    YStart(0), YEnd(0), 
    YStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL)),
    ZStart(0), ZEnd(0),
    ZStride(IntegerLiteral::Create(C, llvm::APInt(32, 1), C.IntTy, FL))
{

#ifdef USE_FORALL_BLOCK
  setBlock(Block);
#endif
  setOp(Op);
  setBody(Body);
}
