#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>

using namespace clang;

ScoutVectorMemberExpr*
ScoutVectorMemberExpr::Create(ASTContext &C, 
							  Expr *base,
                              SourceLocation loc, 
                              unsigned index, 
                              QualType ty) {
  assert(base != 0); // SC_TODO -- is this safe or is null valid???

  ScoutVectorMemberExpr* E;
  E = new (C) ScoutVectorMemberExpr(base, loc, index, ty);
  return E;
}
