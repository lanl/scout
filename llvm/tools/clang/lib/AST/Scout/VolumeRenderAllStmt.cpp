
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


VolumeRenderAllStmt::VolumeRenderAllStmt(ASTContext& C, SourceLocation VolRenLoc,
    SourceLocation LB, SourceLocation RB, 
    IdentifierInfo* MII, VarDecl* MVD, IdentifierInfo* CII,
    VarDecl* CVD, CompoundStmt* Body)
    : Stmt(VolumeRenderAllStmtClass),
      MeshII(MII), MeshVarDecl(MVD), CameraII(CII), CameraVarDecl(CVD), 
      VolRenLoc(VolRenLoc), LBracLoc(LB), RBracLoc(RB) 
{
      setOp(0);
      setBody(Body);
}

const MeshType* VolumeRenderAllStmt::getMeshType() const {
  return dyn_cast<MeshType>(MeshVarDecl->getType().getTypePtr());
}
