
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


// ----- RenderallStmt
//
// Constructor for a forall statement w/out a predicate expression.
//
RenderallStmt::RenderallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation RenderallLocation,
                       Stmt* Body)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    RenderallKWLoc(RenderallLocation) {

  SubExprs[PREDICATE] = 0;
  SubExprs[BODY]      = Body;
}


// ----- RenderallStmt::RenderallStmt
//
// Constructor for a forall statement w/ a predicate expression.
//
RenderallStmt::RenderallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation RenderallLocation,
                       Stmt* Body,
                       Expr* Predicate,
                       SourceLocation LeftParenLoc, SourceLocation RightParenLoc)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    RenderallKWLoc(RenderallLocation),
    LParenLoc(LeftParenLoc), RParenLoc(RightParenLoc) {

  SubExprs[PREDICATE] = Predicate;
  SubExprs[BODY]      = Body;
}






// ----- RenderallMeshStmt
//
// Constructor for a renderall mesh statement w/out a predicate expression.
//
RenderallMeshStmt::RenderallMeshStmt(MeshElementType RefElement,
                                     IdentifierInfo* RefVarInfo,
                                     IdentifierInfo* MeshInfo,
                                     VarDecl* MeshVarDecl,
                                     const MeshType* MT,
                                     SourceLocation ForallLocation,
                                     Stmt *Body)
  : RenderallStmt(ForallMeshStmtClass,
                  RefVarInfo,
                  MeshInfo, MeshVarDecl,
                  ForallLocation, Body) {

    MeshElementRef = RefElement;
    MeshRefType    = MT;
  }


// ----- RenderallMeshStmt
//
// Constructor for a renderall mesh statement w/ a predicate expression.
//
RenderallMeshStmt::RenderallMeshStmt(MeshElementType RefElement,
                                     IdentifierInfo* RefVarInfo,
                                     IdentifierInfo* MeshInfo,
                                     VarDecl* MeshVarDecl,
                                     const MeshType* MT,
                                     SourceLocation ForallLocation,
                                     Stmt *Body,
                                     Expr* Predicate,
                                     SourceLocation LeftParenLoc,
                                     SourceLocation RightParenLoc)
  : RenderallStmt(ForallMeshStmtClass,
                  RefVarInfo,
                  MeshInfo, MeshVarDecl,
                  ForallLocation, Body,
                  Predicate, LeftParenLoc, RightParenLoc) {

    MeshElementRef = RefElement;
    MeshRefType    = MT;
  }

bool RenderallMeshStmt::isUniformMesh() const {
  return MeshRefType->getTypeClass() == Type::UniformMesh;
}

bool RenderallMeshStmt::isRectilinearMesh() const {
  return MeshRefType->getTypeClass() == Type::RectilinearMesh;
}

bool RenderallMeshStmt::isStructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::StructuredMesh;
}

bool RenderallMeshStmt::isUnstructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::UnstructuredMesh;
}

