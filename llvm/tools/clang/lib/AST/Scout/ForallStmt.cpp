
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


// ----- ForallStmt
//
// Constructor for a forall statement w/out a predicate expression.
//
ForallStmt::ForallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation ForallLocation,
                       Stmt* Body)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    ForallKWLoc(ForallLocation) {

  SubExprs[PREDICATE] = 0;
  SubExprs[BODY]      = Body;
}


// ----- ForallStmt::ForallStmt
//
// Constructor for a forall statement w/ a predicate expression.
//
ForallStmt::ForallStmt(StmtClass StatementClass,
                       IdentifierInfo* RefVarInfo,
                       IdentifierInfo* ContainerInfo,
                       VarDecl *ContainerVD,
                       SourceLocation ForallLocation,
                       Stmt* Body,
                       Expr* Predicate,
                       SourceLocation LeftParenLoc, SourceLocation RightParenLoc)
  : Stmt(StatementClass),
    LoopRefVarInfo(RefVarInfo),
    ContainerRefVarInfo(ContainerInfo),
    ContainerVarDecl(ContainerVD),
    ForallKWLoc(ForallLocation),
    LParenLoc(LeftParenLoc), RParenLoc(RightParenLoc) {

  SubExprs[PREDICATE] = Predicate;
  SubExprs[BODY]      = Body;
}





// ----- ForallMeshStmt
//
// Constructor for a forall mesh statement w/out a predicate expression.
//
ForallMeshStmt::ForallMeshStmt(MeshElementType RefElement,
                               IdentifierInfo* RefVarInfo,
                               IdentifierInfo* MeshInfo,
                               VarDecl* MeshVarDecl,
                               const MeshType* MT,
                               SourceLocation ForallLocation,
                               Stmt *Body)
  : ForallStmt(ForallMeshStmtClass,
               RefVarInfo,
               MeshInfo, MeshVarDecl,
               ForallLocation, Body) {

    MeshElementRef = RefElement;




    MeshRefType    = MT;
  }


// ----- ForallMeshStmt
//
// Constructor for a forall mesh statement w/ a predicate expression.
//
ForallMeshStmt::ForallMeshStmt(MeshElementType RefElement,
                               IdentifierInfo* RefVarInfo,
                               IdentifierInfo* MeshInfo,
                               VarDecl* MeshVarDecl,
                               const MeshType* MT,
                               SourceLocation ForallLocation,
                               Stmt *Body,
                               Expr* Predicate,
                               SourceLocation LeftParenLoc, SourceLocation RightParenLoc)
  : ForallStmt(ForallMeshStmtClass,
               RefVarInfo,
               MeshInfo, MeshVarDecl,
               ForallLocation, Body,
               Predicate, LeftParenLoc, RightParenLoc) {

    MeshElementRef = RefElement;
    MeshRefType    = MT;
  }

bool ForallMeshStmt::isUniformMesh() const {
  return MeshRefType->getTypeClass() == Type::UniformMesh;
}

bool ForallMeshStmt::isRectilinearMesh() const {
  return MeshRefType->getTypeClass() == Type::RectilinearMesh;
}

bool ForallMeshStmt::isStructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::StructuredMesh;
}

bool ForallMeshStmt::isUnstructuredMesh() const {
  return MeshRefType->getTypeClass() == Type::UnstructuredMesh;
}

