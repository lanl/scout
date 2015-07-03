//===--- Stmt.h - Classes for representing statements -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Stmt interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "clang/AST/DeclGroup.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/CapturedStmt.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>
// +==== Scout =============================================================+
#include <map>
// +========================================================================+

namespace llvm {
  class FoldingSetNodeID;
}

namespace clang {
  class ASTContext;
  class Attr;
  class CapturedDecl;
  class Decl;
  class Expr;
  class IdentifierInfo;
  class LabelDecl;
  class ParmVarDecl;
  class PrinterHelper;
  struct PrintingPolicy;
  class QualType;
  class RecordDecl;
  class SourceManager;
  class StringLiteral;
  class SwitchStmt;
  class Token;
  class VarDecl;

  // +==== Scout =============================================================+
  class MeshType;
  class BlockExpr;
  class MemberExpr;
  class QueryExpr;
  class ScoutExpr;
  class StencilShiftExpr;
  class FrameDecl;
  class SpecObjectExpr;
  class CallExpr;
  // =========================================================================+

  //===--------------------------------------------------------------------===//
  // ExprIterator - Iterators for iterating over Stmt* arrays that contain
  //  only Expr*.  This is needed because AST nodes use Stmt* arrays to store
  //  references to children (to be compatible with StmtIterator).
  //===--------------------------------------------------------------------===//

  class Stmt;
  class Expr;

  class ExprIterator : public std::iterator<std::forward_iterator_tag,
                                            Expr *&, ptrdiff_t,
                                            Expr *&, Expr *&> {
    Stmt** I;
  public:
    ExprIterator(Stmt** i) : I(i) {}
    ExprIterator() : I(nullptr) {}
    ExprIterator& operator++() { ++I; return *this; }
    ExprIterator operator-(size_t i) { return I-i; }
    ExprIterator operator+(size_t i) { return I+i; }
    Expr* operator[](size_t idx);
    // FIXME: Verify that this will correctly return a signed distance.
    signed operator-(const ExprIterator& R) const { return I - R.I; }
    Expr* operator*() const;
    Expr* operator->() const;
    bool operator==(const ExprIterator& R) const { return I == R.I; }
    bool operator!=(const ExprIterator& R) const { return I != R.I; }
    bool operator>(const ExprIterator& R) const { return I > R.I; }
    bool operator>=(const ExprIterator& R) const { return I >= R.I; }
  };

  class ConstExprIterator : public std::iterator<std::forward_iterator_tag,
                                                 const Expr *&, ptrdiff_t,
                                                 const Expr *&,
                                                 const Expr *&> {
    const Stmt * const *I;
  public:
    ConstExprIterator(const Stmt * const *i) : I(i) {}
    ConstExprIterator() : I(nullptr) {}
    ConstExprIterator& operator++() { ++I; return *this; }
    ConstExprIterator operator+(size_t i) const { return I+i; }
    ConstExprIterator operator-(size_t i) const { return I-i; }
    const Expr * operator[](size_t idx) const;
    signed operator-(const ConstExprIterator& R) const { return I - R.I; }
    const Expr * operator*() const;
    const Expr * operator->() const;
    bool operator==(const ConstExprIterator& R) const { return I == R.I; }
    bool operator!=(const ConstExprIterator& R) const { return I != R.I; }
    bool operator>(const ConstExprIterator& R) const { return I > R.I; }
    bool operator>=(const ConstExprIterator& R) const { return I >= R.I; }
  };

//===----------------------------------------------------------------------===//
// AST classes for statements.
//===----------------------------------------------------------------------===//

/// Stmt - This represents one statement.
///
class LLVM_ALIGNAS(LLVM_PTR_SIZE) Stmt {
public:
  enum StmtClass {
    NoStmtClass = 0,
#define STMT(CLASS, PARENT) CLASS##Class,
#define STMT_RANGE(BASE, FIRST, LAST) \
        first##BASE##Constant=FIRST##Class, last##BASE##Constant=LAST##Class,
#define LAST_STMT_RANGE(BASE, FIRST, LAST) \
        first##BASE##Constant=FIRST##Class, last##BASE##Constant=LAST##Class
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
  };

  // Make vanilla 'new' and 'delete' illegal for Stmts.
protected:
  void* operator new(size_t bytes) throw() {
    llvm_unreachable("Stmts cannot be allocated with regular 'new'.");
  }
  void operator delete(void* data) throw() {
    llvm_unreachable("Stmts cannot be released with regular 'delete'.");
  }

  class StmtBitfields {
    friend class Stmt;

    /// \brief The statement class.
    unsigned sClass : 8;
  };
  enum { NumStmtBits = 8 };

  class CompoundStmtBitfields {
    friend class CompoundStmt;
    unsigned : NumStmtBits;

    unsigned NumStmts : 32 - NumStmtBits;
  };

  class ExprBitfields {
    friend class Expr;
    friend class DeclRefExpr; // computeDependence
    friend class InitListExpr; // ctor
    friend class DesignatedInitExpr; // ctor
    friend class BlockDeclRefExpr; // ctor
    friend class ASTStmtReader; // deserialization
    friend class CXXNewExpr; // ctor
    friend class DependentScopeDeclRefExpr; // ctor
    friend class CXXConstructExpr; // ctor
    friend class CallExpr; // ctor
    friend class OffsetOfExpr; // ctor
    friend class ObjCMessageExpr; // ctor
    friend class ObjCArrayLiteral; // ctor
    friend class ObjCDictionaryLiteral; // ctor
    friend class ShuffleVectorExpr; // ctor
    friend class ParenListExpr; // ctor
    friend class CXXUnresolvedConstructExpr; // ctor
    friend class CXXDependentScopeMemberExpr; // ctor
    friend class OverloadExpr; // ctor
    friend class PseudoObjectExpr; // ctor
    friend class AtomicExpr; // ctor
    unsigned : NumStmtBits;

    unsigned ValueKind : 2;
    unsigned ObjectKind : 2;
    unsigned TypeDependent : 1;
    unsigned ValueDependent : 1;
    unsigned InstantiationDependent : 1;
    unsigned ContainsUnexpandedParameterPack : 1;
  };
  enum { NumExprBits = 16 };

  class CharacterLiteralBitfields {
    friend class CharacterLiteral;
    unsigned : NumExprBits;

    unsigned Kind : 2;
  };

  enum APFloatSemantics {
    IEEEhalf,
    IEEEsingle,
    IEEEdouble,
    x87DoubleExtended,
    IEEEquad,
    PPCDoubleDouble
  };

  class FloatingLiteralBitfields {
    friend class FloatingLiteral;
    unsigned : NumExprBits;

    unsigned Semantics : 3; // Provides semantics for APFloat construction
    unsigned IsExact : 1;
  };

  class UnaryExprOrTypeTraitExprBitfields {
    friend class UnaryExprOrTypeTraitExpr;
    unsigned : NumExprBits;

    unsigned Kind : 2;
    unsigned IsType : 1; // true if operand is a type, false if an expression.
  };

  class DeclRefExprBitfields {
    friend class DeclRefExpr;
    friend class ASTStmtReader; // deserialization
    unsigned : NumExprBits;

    unsigned HasQualifier : 1;
    unsigned HasTemplateKWAndArgsInfo : 1;
    unsigned HasFoundDecl : 1;
    unsigned HadMultipleCandidates : 1;
    unsigned RefersToEnclosingVariableOrCapture : 1;
  };

  class CastExprBitfields {
    friend class CastExpr;
    unsigned : NumExprBits;

    unsigned Kind : 6;
    unsigned BasePathSize : 32 - 6 - NumExprBits;
  };

  class CallExprBitfields {
    friend class CallExpr;
    unsigned : NumExprBits;

    unsigned NumPreArgs : 1;
  };

  class ExprWithCleanupsBitfields {
    friend class ExprWithCleanups;
    friend class ASTStmtReader; // deserialization

    unsigned : NumExprBits;

    unsigned NumObjects : 32 - NumExprBits;
  };

  class PseudoObjectExprBitfields {
    friend class PseudoObjectExpr;
    friend class ASTStmtReader; // deserialization

    unsigned : NumExprBits;

    // These don't need to be particularly wide, because they're
    // strictly limited by the forms of expressions we permit.
    unsigned NumSubExprs : 8;
    unsigned ResultIndex : 32 - 8 - NumExprBits;
  };

  class ObjCIndirectCopyRestoreExprBitfields {
    friend class ObjCIndirectCopyRestoreExpr;
    unsigned : NumExprBits;

    unsigned ShouldCopy : 1;
  };

  class InitListExprBitfields {
    friend class InitListExpr;

    unsigned : NumExprBits;

    /// Whether this initializer list originally had a GNU array-range
    /// designator in it. This is a temporary marker used by CodeGen.
    unsigned HadArrayRangeDesignator : 1;
  };

  class TypeTraitExprBitfields {
    friend class TypeTraitExpr;
    friend class ASTStmtReader;
    friend class ASTStmtWriter;

    unsigned : NumExprBits;

    /// \brief The kind of type trait, which is a value of a TypeTrait enumerator.
    unsigned Kind : 8;

    /// \brief If this expression is not value-dependent, this indicates whether
    /// the trait evaluated true or false.
    unsigned Value : 1;

    /// \brief The number of arguments to this type trait.
    unsigned NumArgs : 32 - 8 - 1 - NumExprBits;
  };

  union {
    StmtBitfields StmtBits;
    CompoundStmtBitfields CompoundStmtBits;
    ExprBitfields ExprBits;
    CharacterLiteralBitfields CharacterLiteralBits;
    FloatingLiteralBitfields FloatingLiteralBits;
    UnaryExprOrTypeTraitExprBitfields UnaryExprOrTypeTraitExprBits;
    DeclRefExprBitfields DeclRefExprBits;
    CastExprBitfields CastExprBits;
    CallExprBitfields CallExprBits;
    ExprWithCleanupsBitfields ExprWithCleanupsBits;
    PseudoObjectExprBitfields PseudoObjectExprBits;
    ObjCIndirectCopyRestoreExprBitfields ObjCIndirectCopyRestoreExprBits;
    InitListExprBitfields InitListExprBits;
    TypeTraitExprBitfields TypeTraitExprBits;
  };

  friend class ASTStmtReader;
  friend class ASTStmtWriter;

public:
  // Only allow allocation of Stmts using the allocator in ASTContext
  // or by doing a placement new.
  void* operator new(size_t bytes, const ASTContext& C,
                     unsigned alignment = 8);

  void* operator new(size_t bytes, const ASTContext* C,
                     unsigned alignment = 8) {
    return operator new(bytes, *C, alignment);
  }

  void* operator new(size_t bytes, void* mem) throw() {
    return mem;
  }

  void operator delete(void*, const ASTContext&, unsigned) throw() { }
  void operator delete(void*, const ASTContext*, unsigned) throw() { }
  void operator delete(void*, size_t) throw() { }
  void operator delete(void*, void*) throw() { }

public:
  /// \brief A placeholder type used to construct an empty shell of a
  /// type, that will be filled in later (e.g., by some
  /// de-serialization).
  struct EmptyShell { };

private:
  /// \brief Whether statistic collection is enabled.
  static bool StatisticsEnabled;

protected:
  /// \brief Construct an empty statement.
  explicit Stmt(StmtClass SC, EmptyShell) : Stmt(SC) {}

public:
  Stmt(StmtClass SC) {
    static_assert(sizeof(*this) % llvm::AlignOf<void *>::Alignment == 0,
                  "Insufficient alignment!");
    StmtBits.sClass = SC;
    if (StatisticsEnabled) Stmt::addStmtClass(SC);
  }

  StmtClass getStmtClass() const {
    return static_cast<StmtClass>(StmtBits.sClass);
  }
  const char *getStmtClassName() const;

  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  SourceRange getSourceRange() const LLVM_READONLY;
  SourceLocation getLocStart() const LLVM_READONLY;
  SourceLocation getLocEnd() const LLVM_READONLY;

  // global temp stats (until we have a per-module visitor)
  static void addStmtClass(const StmtClass s);
  static void EnableStatistics();
  static void PrintStats();

  /// \brief Dumps the specified AST fragment and all subtrees to
  /// \c llvm::errs().
  void dump() const;
  void dump(SourceManager &SM) const;
  void dump(raw_ostream &OS, SourceManager &SM) const;
  void dump(raw_ostream &OS) const;

  /// dumpColor - same as dump(), but forces color highlighting.
  void dumpColor() const;

  /// dumpPretty/printPretty - These two methods do a "pretty print" of the AST
  /// back to its original source language syntax.
  void dumpPretty(const ASTContext &Context) const;
  void printPretty(raw_ostream &OS, PrinterHelper *Helper,
                   const PrintingPolicy &Policy,
                   unsigned Indentation = 0) const;

  /// viewAST - Visualize an AST rooted at this Stmt* using GraphViz.  Only
  ///   works on systems with GraphViz (Mac OS X) or dot+gv installed.
  void viewAST() const;

  /// Skip past any implicit AST nodes which might surround this
  /// statement, such as ExprWithCleanups or ImplicitCastExpr nodes.
  Stmt *IgnoreImplicit();

  /// \brief Skip no-op (attributed, compound) container stmts and skip captured
  /// stmt at the top, if \a IgnoreCaptured is true.
  Stmt *IgnoreContainers(bool IgnoreCaptured = false);

  const Stmt *stripLabelLikeStatements() const;
  Stmt *stripLabelLikeStatements() {
    return const_cast<Stmt*>(
      const_cast<const Stmt*>(this)->stripLabelLikeStatements());
  }

  /// Child Iterators: All subclasses must implement 'children'
  /// to permit easy iteration over the substatements/subexpessions of an
  /// AST node.  This permits easy iteration over all nodes in the AST.
  typedef StmtIterator       child_iterator;
  typedef ConstStmtIterator  const_child_iterator;

  typedef StmtRange          child_range;
  typedef ConstStmtRange     const_child_range;

  child_range children();
  const_child_range children() const {
    return const_cast<Stmt*>(this)->children();
  }

  child_iterator child_begin() { return children().first; }
  child_iterator child_end() { return children().second; }

  const_child_iterator child_begin() const { return children().first; }
  const_child_iterator child_end() const { return children().second; }

  /// \brief Produce a unique representation of the given statement.
  ///
  /// \param ID once the profiling operation is complete, will contain
  /// the unique representation of the given statement.
  ///
  /// \param Context the AST context in which the statement resides
  ///
  /// \param Canonical whether the profile should be based on the canonical
  /// representation of this statement (e.g., where non-type template
  /// parameters are identified by index/level rather than their
  /// declaration pointers) or the exact representation of the statement as
  /// written in the source.
  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
               bool Canonical) const;
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types). Another example is ForStmt, where
/// the first statement can be an expression or a declaration.
///
class DeclStmt : public Stmt {
  DeclGroupRef DG;
  SourceLocation StartLoc, EndLoc;

public:
  DeclStmt(DeclGroupRef dg, SourceLocation startLoc,
           SourceLocation endLoc) : Stmt(DeclStmtClass), DG(dg),
                                    StartLoc(startLoc), EndLoc(endLoc) {}

  /// \brief Build an empty declaration statement.
  explicit DeclStmt(EmptyShell Empty) : Stmt(DeclStmtClass, Empty) { }

  /// isSingleDecl - This method returns true if this DeclStmt refers
  /// to a single Decl.
  bool isSingleDecl() const {
    return DG.isSingleDecl();
  }

  const Decl *getSingleDecl() const { return DG.getSingleDecl(); }
  Decl *getSingleDecl() { return DG.getSingleDecl(); }

  const DeclGroupRef getDeclGroup() const { return DG; }
  DeclGroupRef getDeclGroup() { return DG; }
  void setDeclGroup(DeclGroupRef DGR) { DG = DGR; }

  SourceLocation getStartLoc() const { return StartLoc; }
  void setStartLoc(SourceLocation L) { StartLoc = L; }
  SourceLocation getEndLoc() const { return EndLoc; }
  void setEndLoc(SourceLocation L) { EndLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return StartLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return EndLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DeclStmtClass;
  }

  // Iterators over subexpressions.
  child_range children() {
    return child_range(child_iterator(DG.begin(), DG.end()),
                       child_iterator(DG.end(), DG.end()));
  }

  typedef DeclGroupRef::iterator decl_iterator;
  typedef DeclGroupRef::const_iterator const_decl_iterator;
  typedef llvm::iterator_range<decl_iterator> decl_range;
  typedef llvm::iterator_range<const_decl_iterator> decl_const_range;

  decl_range decls() { return decl_range(decl_begin(), decl_end()); }
  decl_const_range decls() const {
    return decl_const_range(decl_begin(), decl_end());
  }
  decl_iterator decl_begin() { return DG.begin(); }
  decl_iterator decl_end() { return DG.end(); }
  const_decl_iterator decl_begin() const { return DG.begin(); }
  const_decl_iterator decl_end() const { return DG.end(); }

  typedef std::reverse_iterator<decl_iterator> reverse_decl_iterator;
  reverse_decl_iterator decl_rbegin() {
    return reverse_decl_iterator(decl_end());
  }
  reverse_decl_iterator decl_rend() {
    return reverse_decl_iterator(decl_begin());
  }
};

/// NullStmt - This is the null statement ";": C99 6.8.3p3.
///
class NullStmt : public Stmt {
  SourceLocation SemiLoc;

  /// \brief True if the null statement was preceded by an empty macro, e.g:
  /// @code
  ///   #define CALL(x)
  ///   CALL(0);
  /// @endcode
  bool HasLeadingEmptyMacro;
public:
  NullStmt(SourceLocation L, bool hasLeadingEmptyMacro = false)
    : Stmt(NullStmtClass), SemiLoc(L),
      HasLeadingEmptyMacro(hasLeadingEmptyMacro) {}

  /// \brief Build an empty null statement.
  explicit NullStmt(EmptyShell Empty) : Stmt(NullStmtClass, Empty),
      HasLeadingEmptyMacro(false) { }

  SourceLocation getSemiLoc() const { return SemiLoc; }
  void setSemiLoc(SourceLocation L) { SemiLoc = L; }

  bool hasLeadingEmptyMacro() const { return HasLeadingEmptyMacro; }

  SourceLocation getLocStart() const LLVM_READONLY { return SemiLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return SemiLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == NullStmtClass;
  }

  child_range children() { return child_range(); }

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// CompoundStmt - This represents a group of statements like { stmt stmt }.
///
class CompoundStmt : public Stmt {
  Stmt** Body;
  SourceLocation LBraceLoc, RBraceLoc;

  friend class ASTStmtReader;

public:
  CompoundStmt(const ASTContext &C, ArrayRef<Stmt*> Stmts,
               SourceLocation LB, SourceLocation RB);

  // \brief Build an empty compound statement with a location.
  explicit CompoundStmt(SourceLocation Loc)
    : Stmt(CompoundStmtClass), Body(nullptr), LBraceLoc(Loc), RBraceLoc(Loc) {
    CompoundStmtBits.NumStmts = 0;
  }

  // \brief Build an empty compound statement.
  explicit CompoundStmt(EmptyShell Empty)
    : Stmt(CompoundStmtClass, Empty), Body(nullptr) {
    CompoundStmtBits.NumStmts = 0;
  }

  void setStmts(const ASTContext &C, Stmt **Stmts, unsigned NumStmts);

  bool body_empty() const { return CompoundStmtBits.NumStmts == 0; }
  unsigned size() const { return CompoundStmtBits.NumStmts; }

  typedef Stmt** body_iterator;
  typedef llvm::iterator_range<body_iterator> body_range;

  body_range body() { return body_range(body_begin(), body_end()); }
  body_iterator body_begin() { return Body; }
  body_iterator body_end() { return Body + size(); }
  Stmt *body_front() { return !body_empty() ? Body[0] : nullptr; }
  Stmt *body_back() { return !body_empty() ? Body[size()-1] : nullptr; }

  void setLastStmt(Stmt *S) {
    assert(!body_empty() && "setLastStmt");
    Body[size()-1] = S;
  }

  typedef Stmt* const * const_body_iterator;
  typedef llvm::iterator_range<const_body_iterator> body_const_range;

  body_const_range body() const {
    return body_const_range(body_begin(), body_end());
  }
  const_body_iterator body_begin() const { return Body; }
  const_body_iterator body_end() const { return Body + size(); }
  const Stmt *body_front() const {
    return !body_empty() ? Body[0] : nullptr;
  }
  const Stmt *body_back() const {
    return !body_empty() ? Body[size() - 1] : nullptr;
  }

  typedef std::reverse_iterator<body_iterator> reverse_body_iterator;
  reverse_body_iterator body_rbegin() {
    return reverse_body_iterator(body_end());
  }
  reverse_body_iterator body_rend() {
    return reverse_body_iterator(body_begin());
  }

  typedef std::reverse_iterator<const_body_iterator>
          const_reverse_body_iterator;

  const_reverse_body_iterator body_rbegin() const {
    return const_reverse_body_iterator(body_end());
  }

  const_reverse_body_iterator body_rend() const {
    return const_reverse_body_iterator(body_begin());
  }

  SourceLocation getLocStart() const LLVM_READONLY { return LBraceLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return RBraceLoc; }

  SourceLocation getLBracLoc() const { return LBraceLoc; }
  SourceLocation getRBracLoc() const { return RBraceLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CompoundStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(Body, Body + CompoundStmtBits.NumStmts);
  }

  const_child_range children() const {
    return child_range(Body, Body + CompoundStmtBits.NumStmts);
  }
};

// SwitchCase is the base class for CaseStmt and DefaultStmt,
class SwitchCase : public Stmt {
protected:
  // A pointer to the following CaseStmt or DefaultStmt class,
  // used by SwitchStmt.
  SwitchCase *NextSwitchCase;
  SourceLocation KeywordLoc;
  SourceLocation ColonLoc;

  SwitchCase(StmtClass SC, SourceLocation KWLoc, SourceLocation ColonLoc)
    : Stmt(SC), NextSwitchCase(nullptr), KeywordLoc(KWLoc), ColonLoc(ColonLoc) {
  }

  SwitchCase(StmtClass SC, EmptyShell)
    : Stmt(SC), NextSwitchCase(nullptr) {}

public:
  const SwitchCase *getNextSwitchCase() const { return NextSwitchCase; }

  SwitchCase *getNextSwitchCase() { return NextSwitchCase; }

  void setNextSwitchCase(SwitchCase *SC) { NextSwitchCase = SC; }

  SourceLocation getKeywordLoc() const { return KeywordLoc; }
  void setKeywordLoc(SourceLocation L) { KeywordLoc = L; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  Stmt *getSubStmt();
  const Stmt *getSubStmt() const {
    return const_cast<SwitchCase*>(this)->getSubStmt();
  }

  SourceLocation getLocStart() const LLVM_READONLY { return KeywordLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CaseStmtClass ||
           T->getStmtClass() == DefaultStmtClass;
  }
};

class CaseStmt : public SwitchCase {
  SourceLocation EllipsisLoc;
  enum { LHS, RHS, SUBSTMT, END_EXPR };
  Stmt* SubExprs[END_EXPR];  // The expression for the RHS is Non-null for
                             // GNU "case 1 ... 4" extension
public:
  CaseStmt(Expr *lhs, Expr *rhs, SourceLocation caseLoc,
           SourceLocation ellipsisLoc, SourceLocation colonLoc)
    : SwitchCase(CaseStmtClass, caseLoc, colonLoc) {
    SubExprs[SUBSTMT] = nullptr;
    SubExprs[LHS] = reinterpret_cast<Stmt*>(lhs);
    SubExprs[RHS] = reinterpret_cast<Stmt*>(rhs);
    EllipsisLoc = ellipsisLoc;
  }

  /// \brief Build an empty switch case statement.
  explicit CaseStmt(EmptyShell Empty) : SwitchCase(CaseStmtClass, Empty) { }

  SourceLocation getCaseLoc() const { return KeywordLoc; }
  void setCaseLoc(SourceLocation L) { KeywordLoc = L; }
  SourceLocation getEllipsisLoc() const { return EllipsisLoc; }
  void setEllipsisLoc(SourceLocation L) { EllipsisLoc = L; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  Expr *getLHS() { return reinterpret_cast<Expr*>(SubExprs[LHS]); }
  Expr *getRHS() { return reinterpret_cast<Expr*>(SubExprs[RHS]); }
  Stmt *getSubStmt() { return SubExprs[SUBSTMT]; }

  const Expr *getLHS() const {
    return reinterpret_cast<const Expr*>(SubExprs[LHS]);
  }
  const Expr *getRHS() const {
    return reinterpret_cast<const Expr*>(SubExprs[RHS]);
  }
  const Stmt *getSubStmt() const { return SubExprs[SUBSTMT]; }

  void setSubStmt(Stmt *S) { SubExprs[SUBSTMT] = S; }
  void setLHS(Expr *Val) { SubExprs[LHS] = reinterpret_cast<Stmt*>(Val); }
  void setRHS(Expr *Val) { SubExprs[RHS] = reinterpret_cast<Stmt*>(Val); }

  SourceLocation getLocStart() const LLVM_READONLY { return KeywordLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    // Handle deeply nested case statements with iteration instead of recursion.
    const CaseStmt *CS = this;
    while (const CaseStmt *CS2 = dyn_cast<CaseStmt>(CS->getSubStmt()))
      CS = CS2;

    return CS->getSubStmt()->getLocEnd();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CaseStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[END_EXPR]);
  }
};

class DefaultStmt : public SwitchCase {
  Stmt* SubStmt;
public:
  DefaultStmt(SourceLocation DL, SourceLocation CL, Stmt *substmt) :
    SwitchCase(DefaultStmtClass, DL, CL), SubStmt(substmt) {}

  /// \brief Build an empty default statement.
  explicit DefaultStmt(EmptyShell Empty)
    : SwitchCase(DefaultStmtClass, Empty) { }

  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }
  void setSubStmt(Stmt *S) { SubStmt = S; }

  SourceLocation getDefaultLoc() const { return KeywordLoc; }
  void setDefaultLoc(SourceLocation L) { KeywordLoc = L; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return KeywordLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return SubStmt->getLocEnd();}

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DefaultStmtClass;
  }

  // Iterators
  child_range children() { return child_range(&SubStmt, &SubStmt+1); }
};

inline SourceLocation SwitchCase::getLocEnd() const {
  if (const CaseStmt *CS = dyn_cast<CaseStmt>(this))
    return CS->getLocEnd();
  return cast<DefaultStmt>(this)->getLocEnd();
}

/// LabelStmt - Represents a label, which has a substatement.  For example:
///    foo: return;
///
class LabelStmt : public Stmt {
  SourceLocation IdentLoc;
  LabelDecl *TheDecl;
  Stmt *SubStmt;

public:
  LabelStmt(SourceLocation IL, LabelDecl *D, Stmt *substmt)
      : Stmt(LabelStmtClass), IdentLoc(IL), TheDecl(D), SubStmt(substmt) {
    static_assert(sizeof(LabelStmt) ==
                      2 * sizeof(SourceLocation) + 2 * sizeof(void *),
                  "LabelStmt too big");
  }

  // \brief Build an empty label statement.
  explicit LabelStmt(EmptyShell Empty) : Stmt(LabelStmtClass, Empty) { }

  SourceLocation getIdentLoc() const { return IdentLoc; }
  LabelDecl *getDecl() const { return TheDecl; }
  void setDecl(LabelDecl *D) { TheDecl = D; }
  const char *getName() const;
  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }
  void setIdentLoc(SourceLocation L) { IdentLoc = L; }
  void setSubStmt(Stmt *SS) { SubStmt = SS; }

  SourceLocation getLocStart() const LLVM_READONLY { return IdentLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return SubStmt->getLocEnd();}

  child_range children() { return child_range(&SubStmt, &SubStmt+1); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == LabelStmtClass;
  }
};


/// \brief Represents an attribute applied to a statement.
///
/// Represents an attribute applied to a statement. For example:
///   [[omp::for(...)]] for (...) { ... }
///
class AttributedStmt : public Stmt {
  Stmt *SubStmt;
  SourceLocation AttrLoc;
  unsigned NumAttrs;

  friend class ASTStmtReader;

  AttributedStmt(SourceLocation Loc, ArrayRef<const Attr*> Attrs, Stmt *SubStmt)
    : Stmt(AttributedStmtClass), SubStmt(SubStmt), AttrLoc(Loc),
      NumAttrs(Attrs.size()) {
    memcpy(getAttrArrayPtr(), Attrs.data(), Attrs.size() * sizeof(Attr *));
  }

  explicit AttributedStmt(EmptyShell Empty, unsigned NumAttrs)
    : Stmt(AttributedStmtClass, Empty), NumAttrs(NumAttrs) {
    memset(getAttrArrayPtr(), 0, NumAttrs * sizeof(Attr *));
  }

  Attr *const *getAttrArrayPtr() const {
    return reinterpret_cast<Attr *const *>(this + 1);
  }
  Attr **getAttrArrayPtr() { return reinterpret_cast<Attr **>(this + 1); }

public:
  static AttributedStmt *Create(const ASTContext &C, SourceLocation Loc,
                                ArrayRef<const Attr*> Attrs, Stmt *SubStmt);
  // \brief Build an empty attributed statement.
  static AttributedStmt *CreateEmpty(const ASTContext &C, unsigned NumAttrs);

  SourceLocation getAttrLoc() const { return AttrLoc; }
  ArrayRef<const Attr*> getAttrs() const {
    return llvm::makeArrayRef(getAttrArrayPtr(), NumAttrs);
  }
  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }

  SourceLocation getLocStart() const LLVM_READONLY { return AttrLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return SubStmt->getLocEnd();}

  child_range children() { return child_range(&SubStmt, &SubStmt + 1); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == AttributedStmtClass;
  }
};


/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  enum { VAR, COND, THEN, ELSE, END_EXPR };
  Stmt* SubExprs[END_EXPR];

  SourceLocation IfLoc;
  SourceLocation ElseLoc;

public:
  IfStmt(const ASTContext &C, SourceLocation IL, VarDecl *var, Expr *cond,
         Stmt *then, SourceLocation EL = SourceLocation(),
         Stmt *elsev = nullptr);

  /// \brief Build an empty if/then/else statement
  explicit IfStmt(EmptyShell Empty) : Stmt(IfStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "if" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// if (int x = foo()) {
  ///   printf("x is %d", x);
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const;
  void setConditionVariable(const ASTContext &C, VarDecl *V);

  /// If this IfStmt has a condition variable, return the faux DeclStmt
  /// associated with the creation of that condition variable.
  const DeclStmt *getConditionVariableDeclStmt() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
  }

  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
  const Stmt *getThen() const { return SubExprs[THEN]; }
  void setThen(Stmt *S) { SubExprs[THEN] = S; }
  const Stmt *getElse() const { return SubExprs[ELSE]; }
  void setElse(Stmt *S) { SubExprs[ELSE] = S; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Stmt *getThen() { return SubExprs[THEN]; }
  Stmt *getElse() { return SubExprs[ELSE]; }

  SourceLocation getIfLoc() const { return IfLoc; }
  void setIfLoc(SourceLocation L) { IfLoc = L; }
  SourceLocation getElseLoc() const { return ElseLoc; }
  void setElseLoc(SourceLocation L) { ElseLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return IfLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    if (SubExprs[ELSE])
      return SubExprs[ELSE]->getLocEnd();
    else
      return SubExprs[THEN]->getLocEnd();
  }

  // Iterators over subexpressions.  The iterators will include iterating
  // over the initialization expression referenced by the condition variable.
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == IfStmtClass;
  }
};

/// SwitchStmt - This represents a 'switch' stmt.
///
class SwitchStmt : public Stmt {
  SourceLocation SwitchLoc;
  enum { VAR, COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  // This points to a linked list of case and default statements and, if the
  // SwitchStmt is a switch on an enum value, records whether all the enum
  // values were covered by CaseStmts.  The coverage information value is meant
  // to be a hint for possible clients.
  llvm::PointerIntPair<SwitchCase *, 1, bool> FirstCase;

public:
  SwitchStmt(const ASTContext &C, VarDecl *Var, Expr *cond);

  /// \brief Build a empty switch statement.
  explicit SwitchStmt(EmptyShell Empty) : Stmt(SwitchStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "switch" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// switch (int x = foo()) {
  ///   case 0: break;
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const;
  void setConditionVariable(const ASTContext &C, VarDecl *V);

  /// If this SwitchStmt has a condition variable, return the faux DeclStmt
  /// associated with the creation of that condition variable.
  const DeclStmt *getConditionVariableDeclStmt() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
  }

  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Stmt *getBody() const { return SubExprs[BODY]; }
  const SwitchCase *getSwitchCaseList() const { return FirstCase.getPointer(); }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }
  SwitchCase *getSwitchCaseList() { return FirstCase.getPointer(); }

  /// \brief Set the case list for this switch statement.
  void setSwitchCaseList(SwitchCase *SC) { FirstCase.setPointer(SC); }

  SourceLocation getSwitchLoc() const { return SwitchLoc; }
  void setSwitchLoc(SourceLocation L) { SwitchLoc = L; }

  void setBody(Stmt *S, SourceLocation SL) {
    SubExprs[BODY] = S;
    SwitchLoc = SL;
  }
  void addSwitchCase(SwitchCase *SC) {
    assert(!SC->getNextSwitchCase()
           && "case/default already added to a switch");
    SC->setNextSwitchCase(FirstCase.getPointer());
    FirstCase.setPointer(SC);
  }

  /// Set a flag in the SwitchStmt indicating that if the 'switch (X)' is a
  /// switch over an enum value then all cases have been explicitly covered.
  void setAllEnumCasesCovered() { FirstCase.setInt(true); }

  /// Returns true if the SwitchStmt is a switch of an enum value and all cases
  /// have been explicitly covered.
  bool isAllEnumCasesCovered() const { return FirstCase.getInt(); }

  SourceLocation getLocStart() const LLVM_READONLY { return SwitchLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY] ? SubExprs[BODY]->getLocEnd() : SubExprs[COND]->getLocEnd();
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SwitchStmtClass;
  }
};


/// WhileStmt - This represents a 'while' stmt.
///
class WhileStmt : public Stmt {
  SourceLocation WhileLoc;
  enum { VAR, COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
public:
  WhileStmt(const ASTContext &C, VarDecl *Var, Expr *cond, Stmt *body,
            SourceLocation WL);

  /// \brief Build an empty while statement.
  explicit WhileStmt(EmptyShell Empty) : Stmt(WhileStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "while" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// while (int x = random()) {
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const;
  void setConditionVariable(const ASTContext &C, VarDecl *V);

  /// If this WhileStmt has a condition variable, return the faux DeclStmt
  /// associated with the creation of that condition variable.
  const DeclStmt *getConditionVariableDeclStmt() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[VAR]);
  }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getWhileLoc() const { return WhileLoc; }
  void setWhileLoc(SourceLocation L) { WhileLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return WhileLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY]->getLocEnd();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == WhileStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
};

/// DoStmt - This represents a 'do/while' stmt.
///
class DoStmt : public Stmt {
  SourceLocation DoLoc;
  enum { BODY, COND, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  SourceLocation WhileLoc;
  SourceLocation RParenLoc;  // Location of final ')' in do stmt condition.

public:
  DoStmt(Stmt *body, Expr *cond, SourceLocation DL, SourceLocation WL,
         SourceLocation RP)
    : Stmt(DoStmtClass), DoLoc(DL), WhileLoc(WL), RParenLoc(RP) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
  }

  /// \brief Build an empty do-while statement.
  explicit DoStmt(EmptyShell Empty) : Stmt(DoStmtClass, Empty) { }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getDoLoc() const { return DoLoc; }
  void setDoLoc(SourceLocation L) { DoLoc = L; }
  SourceLocation getWhileLoc() const { return WhileLoc; }
  void setWhileLoc(SourceLocation L) { WhileLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return DoLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return RParenLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DoStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
};


/// ForStmt - This represents a 'for (init;cond;inc)' stmt.  Note that any of
/// the init/cond/inc parts of the ForStmt will be null if they were not
/// specified in the source.
///
class ForStmt : public Stmt {
  SourceLocation ForLoc;
  enum { INIT, CONDVAR, COND, INC, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[INIT] is an expression or declstmt.
  SourceLocation LParenLoc, RParenLoc;

public:
  ForStmt(const ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
          Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
          SourceLocation RP);

  /// \brief Build an empty for statement.
  explicit ForStmt(EmptyShell Empty) : Stmt(ForStmtClass, Empty) { }

  Stmt *getInit() { return SubExprs[INIT]; }

  /// \brief Retrieve the variable declared in this "for" statement, if any.
  ///
  /// In the following example, "y" is the condition variable.
  /// \code
  /// for (int x = random(); int y = mangle(x); ++x) {
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const;
  void setConditionVariable(const ASTContext &C, VarDecl *V);

  /// If this ForStmt has a condition variable, return the faux DeclStmt
  /// associated with the creation of that condition variable.
  const DeclStmt *getConditionVariableDeclStmt() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[CONDVAR]);
  }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForLoc() const { return ForLoc; }
  void setForLoc(SourceLocation L) { ForLoc = L; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return ForLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY]->getLocEnd();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ForStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
};

/// GotoStmt - This represents a direct goto.
///
class GotoStmt : public Stmt {
  LabelDecl *Label;
  SourceLocation GotoLoc;
  SourceLocation LabelLoc;
public:
  GotoStmt(LabelDecl *label, SourceLocation GL, SourceLocation LL)
    : Stmt(GotoStmtClass), Label(label), GotoLoc(GL), LabelLoc(LL) {}

  /// \brief Build an empty goto statement.
  explicit GotoStmt(EmptyShell Empty) : Stmt(GotoStmtClass, Empty) { }

  LabelDecl *getLabel() const { return Label; }
  void setLabel(LabelDecl *D) { Label = D; }

  SourceLocation getGotoLoc() const { return GotoLoc; }
  void setGotoLoc(SourceLocation L) { GotoLoc = L; }
  SourceLocation getLabelLoc() const { return LabelLoc; }
  void setLabelLoc(SourceLocation L) { LabelLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return GotoLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return LabelLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GotoStmtClass;
  }

  // Iterators
  child_range children() { return child_range(); }
};

/// IndirectGotoStmt - This represents an indirect goto.
///
class IndirectGotoStmt : public Stmt {
  SourceLocation GotoLoc;
  SourceLocation StarLoc;
  Stmt *Target;
public:
  IndirectGotoStmt(SourceLocation gotoLoc, SourceLocation starLoc,
                   Expr *target)
    : Stmt(IndirectGotoStmtClass), GotoLoc(gotoLoc), StarLoc(starLoc),
      Target((Stmt*)target) {}

  /// \brief Build an empty indirect goto statement.
  explicit IndirectGotoStmt(EmptyShell Empty)
    : Stmt(IndirectGotoStmtClass, Empty) { }

  void setGotoLoc(SourceLocation L) { GotoLoc = L; }
  SourceLocation getGotoLoc() const { return GotoLoc; }
  void setStarLoc(SourceLocation L) { StarLoc = L; }
  SourceLocation getStarLoc() const { return StarLoc; }

  Expr *getTarget() { return reinterpret_cast<Expr*>(Target); }
  const Expr *getTarget() const {return reinterpret_cast<const Expr*>(Target);}
  void setTarget(Expr *E) { Target = reinterpret_cast<Stmt*>(E); }

  /// getConstantTarget - Returns the fixed target of this indirect
  /// goto, if one exists.
  LabelDecl *getConstantTarget();
  const LabelDecl *getConstantTarget() const {
    return const_cast<IndirectGotoStmt*>(this)->getConstantTarget();
  }

  SourceLocation getLocStart() const LLVM_READONLY { return GotoLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return Target->getLocEnd(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == IndirectGotoStmtClass;
  }

  // Iterators
  child_range children() { return child_range(&Target, &Target+1); }
};


/// ContinueStmt - This represents a continue.
///
class ContinueStmt : public Stmt {
  SourceLocation ContinueLoc;
public:
  ContinueStmt(SourceLocation CL) : Stmt(ContinueStmtClass), ContinueLoc(CL) {}

  /// \brief Build an empty continue statement.
  explicit ContinueStmt(EmptyShell Empty) : Stmt(ContinueStmtClass, Empty) { }

  SourceLocation getContinueLoc() const { return ContinueLoc; }
  void setContinueLoc(SourceLocation L) { ContinueLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return ContinueLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return ContinueLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ContinueStmtClass;
  }

  // Iterators
  child_range children() { return child_range(); }
};

/// BreakStmt - This represents a break.
///
class BreakStmt : public Stmt {
  SourceLocation BreakLoc;

public:
  BreakStmt(SourceLocation BL) : Stmt(BreakStmtClass), BreakLoc(BL) {
    static_assert(sizeof(BreakStmt) == 2 * sizeof(SourceLocation),
                  "BreakStmt too large");
  }

  /// \brief Build an empty break statement.
  explicit BreakStmt(EmptyShell Empty) : Stmt(BreakStmtClass, Empty) { }

  SourceLocation getBreakLoc() const { return BreakLoc; }
  void setBreakLoc(SourceLocation L) { BreakLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return BreakLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return BreakLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == BreakStmtClass;
  }

  // Iterators
  child_range children() { return child_range(); }
};


/// ReturnStmt - This represents a return, optionally of an expression:
///   return;
///   return 4;
///
/// Note that GCC allows return with no argument in a function declared to
/// return a value, and it allows returning a value in functions declared to
/// return void.  We explicitly model this in the AST, which means you can't
/// depend on the return type of the function and the presence of an argument.
///
class ReturnStmt : public Stmt {
  SourceLocation RetLoc;
  Stmt *RetExpr;
  const VarDecl *NRVOCandidate;

public:
  explicit ReturnStmt(SourceLocation RL) : ReturnStmt(RL, nullptr, nullptr) {}

  ReturnStmt(SourceLocation RL, Expr *E, const VarDecl *NRVOCandidate)
      : Stmt(ReturnStmtClass), RetLoc(RL), RetExpr((Stmt *)E),
        NRVOCandidate(NRVOCandidate) {}

  /// \brief Build an empty return expression.
  explicit ReturnStmt(EmptyShell Empty) : Stmt(ReturnStmtClass, Empty) { }

  const Expr *getRetValue() const;
  Expr *getRetValue();
  void setRetValue(Expr *E) { RetExpr = reinterpret_cast<Stmt*>(E); }

  SourceLocation getReturnLoc() const { return RetLoc; }
  void setReturnLoc(SourceLocation L) { RetLoc = L; }

  /// \brief Retrieve the variable that might be used for the named return
  /// value optimization.
  ///
  /// The optimization itself can only be performed if the variable is
  /// also marked as an NRVO object.
  const VarDecl *getNRVOCandidate() const { return NRVOCandidate; }
  void setNRVOCandidate(const VarDecl *Var) { NRVOCandidate = Var; }

  SourceLocation getLocStart() const LLVM_READONLY { return RetLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return RetExpr ? RetExpr->getLocEnd() : RetLoc;
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ReturnStmtClass;
  }

  // Iterators
  child_range children() {
    if (RetExpr) return child_range(&RetExpr, &RetExpr+1);
    return child_range();
  }
};

/// AsmStmt is the base class for GCCAsmStmt and MSAsmStmt.
///
class AsmStmt : public Stmt {
protected:
  SourceLocation AsmLoc;
  /// \brief True if the assembly statement does not have any input or output
  /// operands.
  bool IsSimple;

  /// \brief If true, treat this inline assembly as having side effects.
  /// This assembly statement should not be optimized, deleted or moved.
  bool IsVolatile;

  unsigned NumOutputs;
  unsigned NumInputs;
  unsigned NumClobbers;

  Stmt **Exprs;

  AsmStmt(StmtClass SC, SourceLocation asmloc, bool issimple, bool isvolatile,
          unsigned numoutputs, unsigned numinputs, unsigned numclobbers) :
    Stmt (SC), AsmLoc(asmloc), IsSimple(issimple), IsVolatile(isvolatile),
    NumOutputs(numoutputs), NumInputs(numinputs), NumClobbers(numclobbers) { }

  friend class ASTStmtReader;

public:
  /// \brief Build an empty inline-assembly statement.
  explicit AsmStmt(StmtClass SC, EmptyShell Empty) :
    Stmt(SC, Empty), Exprs(nullptr) { }

  SourceLocation getAsmLoc() const { return AsmLoc; }
  void setAsmLoc(SourceLocation L) { AsmLoc = L; }

  bool isSimple() const { return IsSimple; }
  void setSimple(bool V) { IsSimple = V; }

  bool isVolatile() const { return IsVolatile; }
  void setVolatile(bool V) { IsVolatile = V; }

  SourceLocation getLocStart() const LLVM_READONLY { return SourceLocation(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return SourceLocation(); }

  //===--- Asm String Analysis ---===//

  /// Assemble final IR asm string.
  std::string generateAsmString(const ASTContext &C) const;

  //===--- Output operands ---===//

  unsigned getNumOutputs() const { return NumOutputs; }

  /// getOutputConstraint - Return the constraint string for the specified
  /// output operand.  All output constraints are known to be non-empty (either
  /// '=' or '+').
  StringRef getOutputConstraint(unsigned i) const;

  /// isOutputPlusConstraint - Return true if the specified output constraint
  /// is a "+" constraint (which is both an input and an output) or false if it
  /// is an "=" constraint (just an output).
  bool isOutputPlusConstraint(unsigned i) const {
    return getOutputConstraint(i)[0] == '+';
  }

  const Expr *getOutputExpr(unsigned i) const;

  /// getNumPlusOperands - Return the number of output operands that have a "+"
  /// constraint.
  unsigned getNumPlusOperands() const;

  //===--- Input operands ---===//

  unsigned getNumInputs() const { return NumInputs; }

  /// getInputConstraint - Return the specified input constraint.  Unlike output
  /// constraints, these can be empty.
  StringRef getInputConstraint(unsigned i) const;

  const Expr *getInputExpr(unsigned i) const;

  //===--- Other ---===//

  unsigned getNumClobbers() const { return NumClobbers; }
  StringRef getClobber(unsigned i) const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GCCAsmStmtClass ||
      T->getStmtClass() == MSAsmStmtClass;
  }

  // Input expr iterators.

  typedef ExprIterator inputs_iterator;
  typedef ConstExprIterator const_inputs_iterator;
  typedef llvm::iterator_range<inputs_iterator> inputs_range;
  typedef llvm::iterator_range<const_inputs_iterator> inputs_const_range;

  inputs_iterator begin_inputs() {
    return &Exprs[0] + NumOutputs;
  }

  inputs_iterator end_inputs() {
    return &Exprs[0] + NumOutputs + NumInputs;
  }

  inputs_range inputs() { return inputs_range(begin_inputs(), end_inputs()); }

  const_inputs_iterator begin_inputs() const {
    return &Exprs[0] + NumOutputs;
  }

  const_inputs_iterator end_inputs() const {
    return &Exprs[0] + NumOutputs + NumInputs;
  }

  inputs_const_range inputs() const {
    return inputs_const_range(begin_inputs(), end_inputs());
  }

  // Output expr iterators.

  typedef ExprIterator outputs_iterator;
  typedef ConstExprIterator const_outputs_iterator;
  typedef llvm::iterator_range<outputs_iterator> outputs_range;
  typedef llvm::iterator_range<const_outputs_iterator> outputs_const_range;

  outputs_iterator begin_outputs() {
    return &Exprs[0];
  }
  outputs_iterator end_outputs() {
    return &Exprs[0] + NumOutputs;
  }
  outputs_range outputs() {
    return outputs_range(begin_outputs(), end_outputs());
  }

  const_outputs_iterator begin_outputs() const {
    return &Exprs[0];
  }
  const_outputs_iterator end_outputs() const {
    return &Exprs[0] + NumOutputs;
  }
  outputs_const_range outputs() const {
    return outputs_const_range(begin_outputs(), end_outputs());
  }

  child_range children() {
    return child_range(&Exprs[0], &Exprs[0] + NumOutputs + NumInputs);
  }
};

/// This represents a GCC inline-assembly statement extension.
///
class GCCAsmStmt : public AsmStmt {
  SourceLocation RParenLoc;
  StringLiteral *AsmStr;

  // FIXME: If we wanted to, we could allocate all of these in one big array.
  StringLiteral **Constraints;
  StringLiteral **Clobbers;
  IdentifierInfo **Names;

  friend class ASTStmtReader;

public:
  GCCAsmStmt(const ASTContext &C, SourceLocation asmloc, bool issimple,
             bool isvolatile, unsigned numoutputs, unsigned numinputs,
             IdentifierInfo **names, StringLiteral **constraints, Expr **exprs,
             StringLiteral *asmstr, unsigned numclobbers,
             StringLiteral **clobbers, SourceLocation rparenloc);

  /// \brief Build an empty inline-assembly statement.
  explicit GCCAsmStmt(EmptyShell Empty) : AsmStmt(GCCAsmStmtClass, Empty),
    Constraints(nullptr), Clobbers(nullptr), Names(nullptr) { }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  //===--- Asm String Analysis ---===//

  const StringLiteral *getAsmString() const { return AsmStr; }
  StringLiteral *getAsmString() { return AsmStr; }
  void setAsmString(StringLiteral *E) { AsmStr = E; }

  /// AsmStringPiece - this is part of a decomposed asm string specification
  /// (for use with the AnalyzeAsmString function below).  An asm string is
  /// considered to be a concatenation of these parts.
  class AsmStringPiece {
  public:
    enum Kind {
      String,  // String in .ll asm string form, "$" -> "$$" and "%%" -> "%".
      Operand  // Operand reference, with optional modifier %c4.
    };
  private:
    Kind MyKind;
    std::string Str;
    unsigned OperandNo;

    // Source range for operand references.
    CharSourceRange Range;
  public:
    AsmStringPiece(const std::string &S) : MyKind(String), Str(S) {}
    AsmStringPiece(unsigned OpNo, const std::string &S, SourceLocation Begin,
                   SourceLocation End)
      : MyKind(Operand), Str(S), OperandNo(OpNo),
        Range(CharSourceRange::getCharRange(Begin, End)) {
    }

    bool isString() const { return MyKind == String; }
    bool isOperand() const { return MyKind == Operand; }

    const std::string &getString() const {
      return Str;
    }

    unsigned getOperandNo() const {
      assert(isOperand());
      return OperandNo;
    }

    CharSourceRange getRange() const {
      assert(isOperand() && "Range is currently used only for Operands.");
      return Range;
    }

    /// getModifier - Get the modifier for this operand, if present.  This
    /// returns '\0' if there was no modifier.
    char getModifier() const;
  };

  /// AnalyzeAsmString - Analyze the asm string of the current asm, decomposing
  /// it into pieces.  If the asm string is erroneous, emit errors and return
  /// true, otherwise return false.  This handles canonicalization and
  /// translation of strings from GCC syntax to LLVM IR syntax, and handles
  //// flattening of named references like %[foo] to Operand AsmStringPiece's.
  unsigned AnalyzeAsmString(SmallVectorImpl<AsmStringPiece> &Pieces,
                            const ASTContext &C, unsigned &DiagOffs) const;

  /// Assemble final IR asm string.
  std::string generateAsmString(const ASTContext &C) const;

  //===--- Output operands ---===//

  IdentifierInfo *getOutputIdentifier(unsigned i) const {
    return Names[i];
  }

  StringRef getOutputName(unsigned i) const {
    if (IdentifierInfo *II = getOutputIdentifier(i))
      return II->getName();

    return StringRef();
  }

  StringRef getOutputConstraint(unsigned i) const;

  const StringLiteral *getOutputConstraintLiteral(unsigned i) const {
    return Constraints[i];
  }
  StringLiteral *getOutputConstraintLiteral(unsigned i) {
    return Constraints[i];
  }

  Expr *getOutputExpr(unsigned i);

  const Expr *getOutputExpr(unsigned i) const {
    return const_cast<GCCAsmStmt*>(this)->getOutputExpr(i);
  }

  //===--- Input operands ---===//

  IdentifierInfo *getInputIdentifier(unsigned i) const {
    return Names[i + NumOutputs];
  }

  StringRef getInputName(unsigned i) const {
    if (IdentifierInfo *II = getInputIdentifier(i))
      return II->getName();

    return StringRef();
  }

  StringRef getInputConstraint(unsigned i) const;

  const StringLiteral *getInputConstraintLiteral(unsigned i) const {
    return Constraints[i + NumOutputs];
  }
  StringLiteral *getInputConstraintLiteral(unsigned i) {
    return Constraints[i + NumOutputs];
  }

  Expr *getInputExpr(unsigned i);
  void setInputExpr(unsigned i, Expr *E);

  const Expr *getInputExpr(unsigned i) const {
    return const_cast<GCCAsmStmt*>(this)->getInputExpr(i);
  }

private:
  void setOutputsAndInputsAndClobbers(const ASTContext &C,
                                      IdentifierInfo **Names,
                                      StringLiteral **Constraints,
                                      Stmt **Exprs,
                                      unsigned NumOutputs,
                                      unsigned NumInputs,
                                      StringLiteral **Clobbers,
                                      unsigned NumClobbers);
public:

  //===--- Other ---===//

  /// getNamedOperand - Given a symbolic operand reference like %[foo],
  /// translate this into a numeric value needed to reference the same operand.
  /// This returns -1 if the operand name is invalid.
  int getNamedOperand(StringRef SymbolicName) const;

  StringRef getClobber(unsigned i) const;
  StringLiteral *getClobberStringLiteral(unsigned i) { return Clobbers[i]; }
  const StringLiteral *getClobberStringLiteral(unsigned i) const {
    return Clobbers[i];
  }

  SourceLocation getLocStart() const LLVM_READONLY { return AsmLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return RParenLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GCCAsmStmtClass;
  }
};

/// This represents a Microsoft inline-assembly statement extension.
///
class MSAsmStmt : public AsmStmt {
  SourceLocation LBraceLoc, EndLoc;
  StringRef AsmStr;

  unsigned NumAsmToks;

  Token *AsmToks;
  StringRef *Constraints;
  StringRef *Clobbers;

  friend class ASTStmtReader;

public:
  MSAsmStmt(const ASTContext &C, SourceLocation asmloc,
            SourceLocation lbraceloc, bool issimple, bool isvolatile,
            ArrayRef<Token> asmtoks, unsigned numoutputs, unsigned numinputs,
            ArrayRef<StringRef> constraints,
            ArrayRef<Expr*> exprs, StringRef asmstr,
            ArrayRef<StringRef> clobbers, SourceLocation endloc);

  /// \brief Build an empty MS-style inline-assembly statement.
  explicit MSAsmStmt(EmptyShell Empty) : AsmStmt(MSAsmStmtClass, Empty),
    NumAsmToks(0), AsmToks(nullptr), Constraints(nullptr), Clobbers(nullptr) { }

  SourceLocation getLBraceLoc() const { return LBraceLoc; }
  void setLBraceLoc(SourceLocation L) { LBraceLoc = L; }
  SourceLocation getEndLoc() const { return EndLoc; }
  void setEndLoc(SourceLocation L) { EndLoc = L; }

  bool hasBraces() const { return LBraceLoc.isValid(); }

  unsigned getNumAsmToks() { return NumAsmToks; }
  Token *getAsmToks() { return AsmToks; }

  //===--- Asm String Analysis ---===//
  StringRef getAsmString() const { return AsmStr; }

  /// Assemble final IR asm string.
  std::string generateAsmString(const ASTContext &C) const;

  //===--- Output operands ---===//

  StringRef getOutputConstraint(unsigned i) const {
    assert(i < NumOutputs);
    return Constraints[i];
  }

  Expr *getOutputExpr(unsigned i);

  const Expr *getOutputExpr(unsigned i) const {
    return const_cast<MSAsmStmt*>(this)->getOutputExpr(i);
  }

  //===--- Input operands ---===//

  StringRef getInputConstraint(unsigned i) const {
    assert(i < NumInputs);
    return Constraints[i + NumOutputs];
  }

  Expr *getInputExpr(unsigned i);
  void setInputExpr(unsigned i, Expr *E);

  const Expr *getInputExpr(unsigned i) const {
    return const_cast<MSAsmStmt*>(this)->getInputExpr(i);
  }

  //===--- Other ---===//

  ArrayRef<StringRef> getAllConstraints() const {
    return llvm::makeArrayRef(Constraints, NumInputs + NumOutputs);
  }
  ArrayRef<StringRef> getClobbers() const {
    return llvm::makeArrayRef(Clobbers, NumClobbers);
  }
  ArrayRef<Expr*> getAllExprs() const {
    return llvm::makeArrayRef(reinterpret_cast<Expr**>(Exprs),
                              NumInputs + NumOutputs);
  }

  StringRef getClobber(unsigned i) const { return getClobbers()[i]; }

private:
  void initialize(const ASTContext &C, StringRef AsmString,
                  ArrayRef<Token> AsmToks, ArrayRef<StringRef> Constraints,
                  ArrayRef<Expr*> Exprs, ArrayRef<StringRef> Clobbers);
public:

  SourceLocation getLocStart() const LLVM_READONLY { return AsmLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return EndLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == MSAsmStmtClass;
  }

  child_range children() {
    return child_range(&Exprs[0], &Exprs[NumInputs + NumOutputs]);
  }
};

class SEHExceptStmt : public Stmt {
  SourceLocation  Loc;
  Stmt           *Children[2];

  enum { FILTER_EXPR, BLOCK };

  SEHExceptStmt(SourceLocation Loc,
                Expr *FilterExpr,
                Stmt *Block);

  friend class ASTReader;
  friend class ASTStmtReader;
  explicit SEHExceptStmt(EmptyShell E) : Stmt(SEHExceptStmtClass, E) { }

public:
  static SEHExceptStmt* Create(const ASTContext &C,
                               SourceLocation ExceptLoc,
                               Expr *FilterExpr,
                               Stmt *Block);

  SourceLocation getLocStart() const LLVM_READONLY { return getExceptLoc(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

  SourceLocation getExceptLoc() const { return Loc; }
  SourceLocation getEndLoc() const { return getBlock()->getLocEnd(); }

  Expr *getFilterExpr() const {
    return reinterpret_cast<Expr*>(Children[FILTER_EXPR]);
  }

  CompoundStmt *getBlock() const {
    return cast<CompoundStmt>(Children[BLOCK]);
  }

  child_range children() {
    return child_range(Children,Children+2);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SEHExceptStmtClass;
  }

};

class SEHFinallyStmt : public Stmt {
  SourceLocation  Loc;
  Stmt           *Block;

  SEHFinallyStmt(SourceLocation Loc,
                 Stmt *Block);

  friend class ASTReader;
  friend class ASTStmtReader;
  explicit SEHFinallyStmt(EmptyShell E) : Stmt(SEHFinallyStmtClass, E) { }

public:
  static SEHFinallyStmt* Create(const ASTContext &C,
                                SourceLocation FinallyLoc,
                                Stmt *Block);

  SourceLocation getLocStart() const LLVM_READONLY { return getFinallyLoc(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

  SourceLocation getFinallyLoc() const { return Loc; }
  SourceLocation getEndLoc() const { return Block->getLocEnd(); }

  CompoundStmt *getBlock() const { return cast<CompoundStmt>(Block); }

  child_range children() {
    return child_range(&Block,&Block+1);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SEHFinallyStmtClass;
  }

};

class SEHTryStmt : public Stmt {
  bool            IsCXXTry;
  SourceLocation  TryLoc;
  Stmt           *Children[2];

  enum { TRY = 0, HANDLER = 1 };

  SEHTryStmt(bool isCXXTry, // true if 'try' otherwise '__try'
             SourceLocation TryLoc,
             Stmt *TryBlock,
             Stmt *Handler);

  friend class ASTReader;
  friend class ASTStmtReader;
  explicit SEHTryStmt(EmptyShell E) : Stmt(SEHTryStmtClass, E) { }

public:
  static SEHTryStmt* Create(const ASTContext &C, bool isCXXTry,
                            SourceLocation TryLoc, Stmt *TryBlock,
                            Stmt *Handler);

  SourceLocation getLocStart() const LLVM_READONLY { return getTryLoc(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return getEndLoc(); }

  SourceLocation getTryLoc() const { return TryLoc; }
  SourceLocation getEndLoc() const { return Children[HANDLER]->getLocEnd(); }

  bool getIsCXXTry() const { return IsCXXTry; }

  CompoundStmt* getTryBlock() const {
    return cast<CompoundStmt>(Children[TRY]);
  }

  Stmt *getHandler() const { return Children[HANDLER]; }

  /// Returns 0 if not defined
  SEHExceptStmt  *getExceptHandler() const;
  SEHFinallyStmt *getFinallyHandler() const;

  child_range children() {
    return child_range(Children,Children+2);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SEHTryStmtClass;
  }
};

/// Represents a __leave statement.
///
class SEHLeaveStmt : public Stmt {
  SourceLocation LeaveLoc;
public:
  explicit SEHLeaveStmt(SourceLocation LL)
      : Stmt(SEHLeaveStmtClass), LeaveLoc(LL) {}

  /// \brief Build an empty __leave statement.
  explicit SEHLeaveStmt(EmptyShell Empty) : Stmt(SEHLeaveStmtClass, Empty) { }

  SourceLocation getLeaveLoc() const { return LeaveLoc; }
  void setLeaveLoc(SourceLocation L) { LeaveLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return LeaveLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return LeaveLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SEHLeaveStmtClass;
  }

  // Iterators
  child_range children() { return child_range(); }
};

/// \brief This captures a statement into a function. For example, the following
/// pragma annotated compound statement can be represented as a CapturedStmt,
/// and this compound statement is the body of an anonymous outlined function.
/// @code
/// #pragma omp parallel
/// {
///   compute();
/// }
/// @endcode
class CapturedStmt : public Stmt {
public:
  /// \brief The different capture forms: by 'this', by reference, capture for
  /// variable-length array type etc.
  enum VariableCaptureKind {
    VCK_This,
    VCK_ByRef,
    VCK_VLAType,
  };

  /// \brief Describes the capture of either a variable, or 'this', or
  /// variable-length array type.
  class Capture {
    llvm::PointerIntPair<VarDecl *, 2, VariableCaptureKind> VarAndKind;
    SourceLocation Loc;

  public:
    /// \brief Create a new capture.
    ///
    /// \param Loc The source location associated with this capture.
    ///
    /// \param Kind The kind of capture (this, ByRef, ...).
    ///
    /// \param Var The variable being captured, or null if capturing this.
    ///
    Capture(SourceLocation Loc, VariableCaptureKind Kind,
            VarDecl *Var = nullptr)
      : VarAndKind(Var, Kind), Loc(Loc) {
      switch (Kind) {
      case VCK_This:
        assert(!Var && "'this' capture cannot have a variable!");
        break;
      case VCK_ByRef:
        assert(Var && "capturing by reference must have a variable!");
        break;
      case VCK_VLAType:
        assert(!Var &&
               "Variable-length array type capture cannot have a variable!");
        break;
      }
    }

    /// \brief Determine the kind of capture.
    VariableCaptureKind getCaptureKind() const { return VarAndKind.getInt(); }

    /// \brief Retrieve the source location at which the variable or 'this' was
    /// first used.
    SourceLocation getLocation() const { return Loc; }

    /// \brief Determine whether this capture handles the C++ 'this' pointer.
    bool capturesThis() const { return getCaptureKind() == VCK_This; }

    /// \brief Determine whether this capture handles a variable.
    bool capturesVariable() const { return getCaptureKind() == VCK_ByRef; }

    /// \brief Determine whether this capture handles a variable-length array
    /// type.
    bool capturesVariableArrayType() const {
      return getCaptureKind() == VCK_VLAType;
    }

    /// \brief Retrieve the declaration of the variable being captured.
    ///
    /// This operation is only valid if this capture captures a variable.
    VarDecl *getCapturedVar() const {
      assert(capturesVariable() &&
             "No variable available for 'this' or VAT capture");
      return VarAndKind.getPointer();
    }
    friend class ASTStmtReader;
  };

private:
  /// \brief The number of variable captured, including 'this'.
  unsigned NumCaptures;

  /// \brief The pointer part is the implicit the outlined function and the
  /// int part is the captured region kind, 'CR_Default' etc.
  llvm::PointerIntPair<CapturedDecl *, 1, CapturedRegionKind> CapDeclAndKind;

  /// \brief The record for captured variables, a RecordDecl or CXXRecordDecl.
  RecordDecl *TheRecordDecl;

  /// \brief Construct a captured statement.
  CapturedStmt(Stmt *S, CapturedRegionKind Kind, ArrayRef<Capture> Captures,
               ArrayRef<Expr *> CaptureInits, CapturedDecl *CD, RecordDecl *RD);

  /// \brief Construct an empty captured statement.
  CapturedStmt(EmptyShell Empty, unsigned NumCaptures);

  Stmt **getStoredStmts() const {
    return reinterpret_cast<Stmt **>(const_cast<CapturedStmt *>(this) + 1);
  }

  Capture *getStoredCaptures() const;

  void setCapturedStmt(Stmt *S) { getStoredStmts()[NumCaptures] = S; }

public:
  static CapturedStmt *Create(const ASTContext &Context, Stmt *S,
                              CapturedRegionKind Kind,
                              ArrayRef<Capture> Captures,
                              ArrayRef<Expr *> CaptureInits,
                              CapturedDecl *CD, RecordDecl *RD);

  static CapturedStmt *CreateDeserialized(const ASTContext &Context,
                                          unsigned NumCaptures);

  /// \brief Retrieve the statement being captured.
  Stmt *getCapturedStmt() { return getStoredStmts()[NumCaptures]; }
  const Stmt *getCapturedStmt() const {
    return const_cast<CapturedStmt *>(this)->getCapturedStmt();
  }

  /// \brief Retrieve the outlined function declaration.
  CapturedDecl *getCapturedDecl() { return CapDeclAndKind.getPointer(); }
  const CapturedDecl *getCapturedDecl() const {
    return const_cast<CapturedStmt *>(this)->getCapturedDecl();
  }

  /// \brief Set the outlined function declaration.
  void setCapturedDecl(CapturedDecl *D) {
    assert(D && "null CapturedDecl");
    CapDeclAndKind.setPointer(D);
  }

  /// \brief Retrieve the captured region kind.
  CapturedRegionKind getCapturedRegionKind() const {
    return CapDeclAndKind.getInt();
  }

  /// \brief Set the captured region kind.
  void setCapturedRegionKind(CapturedRegionKind Kind) {
    CapDeclAndKind.setInt(Kind);
  }

  /// \brief Retrieve the record declaration for captured variables.
  const RecordDecl *getCapturedRecordDecl() const { return TheRecordDecl; }

  /// \brief Set the record declaration for captured variables.
  void setCapturedRecordDecl(RecordDecl *D) {
    assert(D && "null RecordDecl");
    TheRecordDecl = D;
  }

  /// \brief True if this variable has been captured.
  bool capturesVariable(const VarDecl *Var) const;

  /// \brief An iterator that walks over the captures.
  typedef Capture *capture_iterator;
  typedef const Capture *const_capture_iterator;
  typedef llvm::iterator_range<capture_iterator> capture_range;
  typedef llvm::iterator_range<const_capture_iterator> capture_const_range;

  capture_range captures() {
    return capture_range(capture_begin(), capture_end());
  }
  capture_const_range captures() const {
    return capture_const_range(capture_begin(), capture_end());
  }

  /// \brief Retrieve an iterator pointing to the first capture.
  capture_iterator capture_begin() { return getStoredCaptures(); }
  const_capture_iterator capture_begin() const { return getStoredCaptures(); }

  /// \brief Retrieve an iterator pointing past the end of the sequence of
  /// captures.
  capture_iterator capture_end() const {
    return getStoredCaptures() + NumCaptures;
  }

  /// \brief Retrieve the number of captures, including 'this'.
  unsigned capture_size() const { return NumCaptures; }

  /// \brief Iterator that walks over the capture initialization arguments.
  typedef Expr **capture_init_iterator;
  typedef llvm::iterator_range<capture_init_iterator> capture_init_range;

  capture_init_range capture_inits() const {
    return capture_init_range(capture_init_begin(), capture_init_end());
  }

  /// \brief Retrieve the first initialization argument.
  capture_init_iterator capture_init_begin() const {
    return reinterpret_cast<Expr **>(getStoredStmts());
  }

  /// \brief Retrieve the iterator pointing one past the last initialization
  /// argument.
  capture_init_iterator capture_init_end() const {
    return capture_init_begin() + NumCaptures;
  }

  SourceLocation getLocStart() const LLVM_READONLY {
    return getCapturedStmt()->getLocStart();
  }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return getCapturedStmt()->getLocEnd();
  }
  SourceRange getSourceRange() const LLVM_READONLY {
    return getCapturedStmt()->getSourceRange();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CapturedStmtClass;
  }

  child_range children();

  friend class ASTStmtReader;
};

// +==== Scout ===============================================================+


// -----  Stmt : base class for forall statements
//
// This class handles the basic functionality of all forall-style statements.
// At this level we have the following information about the loop statement:
//
//   (1) A predicate expression for the overall loop and the body (statements)
//       within the loop.
//
//   (2) The source locations of the forall keyword, the left and
//       right parens that wrap the predicate (if any).
//
class ForallStmt : public Stmt {

protected:


  enum {
    PREDICATE = 0, // Loop predicate.
    // the following are used to add implicit variables
    // into the AST much like what is done in a ForStmt
    // INIT is used for mesh case
    // all 3 may be used in the array case.
    INIT      = 1,
    INITY     = 2,
    INITZ     = 3,
    BODY      = 4, // Body of forall.
    END_EXPR  = 5
  };

  // Store the various components (collection of statements) of the
  // forall...  In this case we store the predicate and the body of
  // the forall.
  Stmt* SubExprs[END_EXPR];

public:

  ForallStmt(StmtClass SC,
             SourceLocation ForallLoc, Stmt *Body);

  ForallStmt(StmtClass SC,
             SourceLocation ForallLoc, Stmt *Body,
             Expr* Predicate,
             SourceLocation LeftParenLoc, SourceLocation RightParenLoc);

  ForallStmt(StmtClass SC, EmptyShell Empty) : Stmt(SC, Empty) {
    SubExprs[PREDICATE] = 0;
    SubExprs[INIT]      = 0;
    SubExprs[INITY]     = 0;
    SubExprs[INITZ]     = 0;
    SubExprs[BODY]      = 0;
  }

  // ===--- Init ---===
  //forall mesh case
  const Stmt *getInit() const { return SubExprs[INIT]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }

  //forall array case
  const Stmt *getInit(size_t axis) const { return SubExprs[INIT+axis]; }

  void setInit(size_t axis, Stmt *S) { SubExprs[INIT+axis] = S; }


  // ===--- Predicate ---===
  Expr* getPredicate() { return reinterpret_cast<Expr*>(SubExprs[PREDICATE]); }

  const Expr* getPredicate() const {
    return reinterpret_cast<Expr*>(SubExprs[PREDICATE]);
  }

  void setPredicate(Expr* P) {
    SubExprs[PREDICATE] = reinterpret_cast<Stmt*>(PREDICATE);
  }

  bool hasPredicate() const {
    return SubExprs[PREDICATE] != 0;
  }

  // ====--- Body ----===
  Stmt* getBody() {
    return SubExprs[BODY];
  }

  const Stmt* getBody() const {
    return SubExprs[BODY];

  }
  void setBody(Stmt* B) {
    SubExprs[BODY] = reinterpret_cast<Stmt*>(B);
  }

  bool hasBodyStatements() const {
    return SubExprs[BODY] != 0;
  }

  // ===--- Source Locations ---===
  SourceLocation getForAllLoc() const {
    return ForallKWLoc;
  }

  void setForAllLoc(SourceLocation L) {
    ForallKWLoc = L;
  }

  SourceLocation getLParenLoc() const {
    return LParenLoc;
  }

  void setLParenLoc(SourceLocation L) {
    LParenLoc = L;
  }

  SourceLocation getRParenLoc() const {
    return RParenLoc;
  }

  void setRParenLoc(SourceLocation L) {
    RParenLoc = L;
  }

  SourceLocation getLocStart() const LLVM_READONLY {
    return ForallKWLoc;
  }

  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY]->getLocEnd();
  }

  SourceRange getSourceRange() const {
    return SourceRange(ForallKWLoc, SubExprs[BODY]->getLocEnd());
  }

  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }

  // ===--- Class identification support ---===
  // **NOTE - we should *never* have an instance that is the base class of the
  // forall inheritance tree.  The following classes are for completeness but
  // will all assert with an error condition if called.
  static bool classof(const ForallStmt *) {
    assert(false && "query of base forall statement class is a no-no");
    return true;
  }

protected:
  SourceLocation  ForallKWLoc, LParenLoc, RParenLoc;
};
// ----- ForallMeshStmt -- forall mesh statement
//
// forall <mesh-loc-kw> <ref-element> in <ref-mesh> {
//   ...
// }
//
// where:
//
//   mesh-loc-kw  - provides the mesh location (elements) to operate over.
//                  This is a keyword that identifies 'cells', 'edges',
//                  'vertices', or 'faces'.
//
//   ref-element  - the reference variable for each field location element in
//                  the mesh being operated on.  This should be a valid
//                  identifier (variable name).  It takes on the value of each
//                  element (cell, edge, vertex, or face) in the mesh within
//                  the body of the 'forall'.
//
//   ref-mesh     - the mesh instance that should be computed over.
//
// Predicated form:
//
//   forall <mesh-loc-kw> <ref-element> in <ref-mesh> (predicated-expr) {
//     ...
//   }
//

// Our mesh-based semantics operate over a specified set of locations
// within the mesh.  These locations are provided in the form of a
// keyword that IDs cells, vertices, edges, or faces of the mesh.  We
// use the following enum value to track the mesh elements referenced
// by many statements and language constructs.
enum MeshElementType {
  Undefined  =  0x0,
    Cells    =  0x1,
    Vertices =  0x2,
    Edges    =  0x4,
    Faces    =  0x8
};
  
class ForallMeshStmt : public ForallStmt {

 public:

 private:
  // The loop's reference element variable is an implicitly declared
   // instance whose type matches that of the mesh location specified
   // by the forall loop construct (for example, the reference variable
   // is a cell in the case of a 'forall cells in mesh' statement).
   // The  reference element is only accessible within the body of the
   // 'forall' -- its value may not be changed within the loop.
   IdentifierInfo* LoopRefVarInfo;

   // The loop's reference mesh variable is the mes that
   // has been specified within the forall statement. We keep track
   // of not only the mesh's identifier info but also the var decl.
   // This is primarily the data gathered at parsing and during semantic
   // checks -- it is useful to keep around for both analysis, optimization
   // and codegen passes...
   IdentifierInfo  *MeshRefVarInfo;
   VarDecl         *MeshVarDecl;
   VarDecl         *MeshQueryVarDecl;
  

  // The mesh location/element we're looping over -- this provides us
  // information about looping over cells, edges, faces, etc.  Is the
  // isOver*() member functions for helpers to determine the element
  // type.
  MeshElementType MeshElementRef;

  // In addition to the container information held in our parent class we also
  // capture the mesh type that the loop is operating over.  Note at this point
  // we are dealing with a generic (base) mesh type and not a specific case
  // (e.g. uniform or rectilinear).  Any operations over the forall should make
  // sure appropriate steps are taken for the various mesh types (and safely
  // cast this type as needed to the appropriate subclass).
  const MeshType* MeshRefType;

public:

  ForallMeshStmt(MeshElementType RefElement,
                 IdentifierInfo* RefVarInfo,
                 IdentifierInfo* MeshInfo,
                 VarDecl* MVD,
                 const MeshType* MT,
                 SourceLocation ForallLocation,
                 DeclStmt* Init, VarDecl* QD, Stmt *Body);

  ForallMeshStmt(MeshElementType RefElement,
                 IdentifierInfo* RefVarInfo,
                 IdentifierInfo* MeshInfo,
                 VarDecl* MVD,
                 const MeshType* MT,
                 SourceLocation ForallLocation,
                 DeclStmt *Init, VarDecl* QD,
                 Stmt *Body,
                 Expr* Predicate,
                 SourceLocation LeftParenLoc,
                 SourceLocation RightParenLoc);

  explicit ForallMeshStmt(EmptyShell Empty)
    : ForallStmt(ForallMeshStmtClass, Empty) { }

  ForallMeshStmt(StmtClass SC, EmptyShell Empty) : ForallStmt(SC, Empty) { }


  // ===--- Mesh details ---===

  // ===--- Reference variable info ---===
   IdentifierInfo* getRefVarInfo() {
     return LoopRefVarInfo;
   }
  
   const IdentifierInfo* getRefElementInfo() const {
    return LoopRefVarInfo;
  }


  MeshElementType getMeshElementRef() const {
    return MeshElementRef;
  }

  void setMeshElementRef(MeshElementType ref) {
    MeshElementRef = ref;
  }

  void setMeshElementRef(tok::TokenKind &token);

  // Determine which mesh elements the statement is operating over...
  bool isOverCells() const    { return MeshElementRef == Cells; }
  bool isOverEdges() const    { return MeshElementRef == Edges; }
  bool isOverVertices() const { return MeshElementRef == Vertices; }
  bool isOverFaces() const    { return MeshElementRef == Faces; }

  // Determine which mesh type the statement is operating over...
  bool isUniformMesh() const;
  bool isRectilinearMesh() const;
  bool isStructuredMesh() const;
  bool isUnstructuredMesh() const;

  IdentifierInfo* getMeshInfo() {
    return  MeshRefVarInfo;
  }

  const IdentifierInfo* getMeshInfo() const {
    return MeshRefVarInfo;
  }

  VarDecl* getMeshVarDecl() {
    return MeshVarDecl;
  }
  
  const VarDecl* getMeshVarDecl() const {
    return MeshVarDecl;
  }

  const MeshType* getMeshType() const { return MeshRefType; }

  VarDecl* getQueryVarDecl() const{
    return MeshQueryVarDecl;
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ForallMeshStmtClass;
  }

  static bool classof(const ForallMeshStmt *) { return true; }
};

// An array-based forall statement
class ForallArrayStmt : public ForallStmt {
  IdentifierInfo* InductionVarInfo[3];
  VarDecl* InductionVarDecl[3];
  Expr* Start[3];
  Expr* End[3];
  Expr* Stride[3];
  size_t Dims;

 public:

   ForallArrayStmt(IdentifierInfo* InductionVarInfo[],
       VarDecl* InductionVarDecl[],
       Expr* Start[], Expr* End[], Expr* Stride[], size_t Dims,
       SourceLocation ForallLoc,
       DeclStmt* Init[], Stmt* Body);

   explicit ForallArrayStmt(EmptyShell Empty) :
   ForallStmt(ForallArrayStmtClass, Empty) { }


   const IdentifierInfo* getInductionVarInfo(size_t axis) const {
     return InductionVarInfo[axis];
   }

   void setInductionVarInfo(size_t axis, IdentifierInfo *Info) {
     InductionVarInfo[axis] = Info;
   }


   const VarDecl* getInductionVarDecl(size_t axis) const {
     return InductionVarDecl[axis];
   }

   void setInductionVarDecl(size_t axis, VarDecl *VD) {
     InductionVarDecl[axis] = VD;
   }

   void setDims(size_t dims) {
     Dims = dims;
   }

   size_t getDims(void) const {
     return Dims;
   }

   void setStart(int axis, Expr *E) {
     Start[axis] = E;
   }

   Expr *getStart(int axis) const {
     return Start[axis];
   }

   void setEnd(int axis, Expr *E) {
     End[axis] = E;
    }

   Expr *getEnd(int axis) const {
     return End[axis];
   }

   void setStride(int axis, Expr *E) {
     Stride[axis] = E;
   }

   Expr *getStride(int axis) const {
    return Stride[axis];
   }

   static bool classof(const Stmt *T) {
     return T->getStmtClass() == ForallArrayStmtClass;
   }

   static bool classof(const ForallArrayStmt *) { return true; }
};


// ----- RenderallStmt -- base class for renderall statements
//
// This class handles the basic functionality of all renderall-style
// statements.  At this level we have the following information about the
// construct:
//
//   (1) The information about the reference element -- the loop's implicit
//       reference variable that represents the active element within the
//       loop scope.
//
//   (2) The variable declaration for the container we are looping over
//       (i.e. a mesh).
//
//   (3) A predicate expression for the overall loop and the body (statements)
//       within the loop.
//
//   (4) The source locations of the renderall keyword, the left and right
//       parens that wrap the predicate (if any).
//
class RenderallStmt : public Stmt {

protected:

  enum {
    PREDICATE = 0, // Loop predicate.
    BODY      = 1, // Body of forall.
    END_EXPR  = 2
  };

  // Store the various components (collection of statements) of the
  // renderall...  In this case we store the predicate and the body of
  // the forall.
  Stmt* SubExprs[END_EXPR];

public:



  RenderallStmt(StmtClass SC,
                IdentifierInfo* RefVarInfo,
                IdentifierInfo* ContainerInfo,
                IdentifierInfo* RenderTargetInfo, 
                VarDecl *ContainerVarDecl,
                VarDecl *RenderTargetDecl,
                SourceLocation ForallLoc, Stmt *Body);

  RenderallStmt(StmtClass SC,
                IdentifierInfo* RefElementInfo,
                IdentifierInfo* ContainerInfo,
                IdentifierInfo* RenderTargetInfo, 
                VarDecl *ContainerVarDecl,
                VarDecl *RenderTargetDecl,
                SourceLocation RenderallLoc, Stmt *Body,
                Expr* Predicate,
                SourceLocation LeftParenLoc,
                SourceLocation RightParenLoc);

  RenderallStmt(StmtClass SC, EmptyShell Empty) : Stmt(SC, Empty) {
    SubExprs[PREDICATE] = 0;
    SubExprs[BODY]      = 0;
    LoopRefVarInfo      = 0;
    RenderTargetInfo    = 0;
    ContainerRefVarInfo = 0;
    ContainerVarDecl    = 0;
    RenderTargetVarDecl = 0;
  }

  // ===--- Reference variable info ---===
  IdentifierInfo* getRefVarInfo() { return LoopRefVarInfo; }
  const IdentifierInfo* getRefElementInfo() const
  { return LoopRefVarInfo; }

  // ===--- Container variable info ---===
  IdentifierInfo* getContainerVarInfo() { return ContainerRefVarInfo; }
  const IdentifierInfo* getContainerVarInfo() const
  { return ContainerRefVarInfo; }

  // ===--- Render target variable info ---===
  IdentifierInfo* getRenderTargetInfo() { return RenderTargetInfo; }
  const IdentifierInfo *getRenderTargetInfo() const
  { return RenderTargetInfo; }

  // ===--- Container variable declaration ---===
  VarDecl* getContainerVarDecl() { return ContainerVarDecl; }
  const VarDecl* getContainerVarDecl() const { return ContainerVarDecl; }


  // ===--- Render target variable declaration ---===
  VarDecl *getRenderTargetVarDecl() { return RenderTargetVarDecl; }
  const VarDecl* getRenderTargetVarDecl() const 
  { return RenderTargetVarDecl; }

  // ===--- Predicate ---===
  Expr* getPredicate() {
    return reinterpret_cast<Expr*>(SubExprs[PREDICATE]);
  }

  const Expr* getPredicate() const {
    return reinterpret_cast<Expr*>(SubExprs[PREDICATE]);
  }

  void setPredicate(Expr* P) {
    SubExprs[PREDICATE] = reinterpret_cast<Stmt*>(PREDICATE);
  }

  bool hasPredicate() const {
    return SubExprs[PREDICATE] != 0;
  }

  // ====--- Body ----===
  Stmt* getBody() { return SubExprs[BODY]; }
  const Stmt* getBody() const { return SubExprs[BODY]; }
  void setBody(Stmt* B) { SubExprs[BODY] = reinterpret_cast<Stmt*>(B); }
  bool hasBodyStatements() const { return SubExprs[BODY] != 0; }

  // ===--- Source Locations ---===
  SourceLocation getRenderallAllLoc() const { return RenderallKWLoc; }
  void setRenderAllLoc(SourceLocation L) { RenderallKWLoc = L; }

  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return RenderallKWLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY]->getLocEnd();
  }

  SourceRange getSourceRange() const {
    return SourceRange(RenderallKWLoc, SubExprs[BODY]->getLocEnd());
  }

  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }

  // ===--- Class identification support ---===
  // **NOTE - we should *never* have an instance that is the base class of the
  // forall inheritance tree.  The following classes are for completeness but
  // will all assert with an error condition if called.
  static bool classof(const RenderallStmt *) {
    assert(false && "query of base forall statement class is a no-no");
    return true;
  }

  void setLast(bool flag) const{
    Last = flag;
  }
  
  bool isLast() const{
    return Last;
  }
  
protected:
  // The loop's reference element variable is an implicitly declared
  // instance whose type matches that of the mesh location specified
  // by the forall loop construct (for example, the reference variable
  // is a cell in the case of a 'forall cells in mesh' statement).
  // The  reference element is only accessible within the body of the
  // 'forall' -- its value may not be changed within the loop.
  IdentifierInfo* LoopRefVarInfo;

  // The render target for the renderall controls where the rendered
  // image is displayed/saved.
  IdentifierInfo* RenderTargetInfo;
  VarDecl*        RenderTargetVarDecl;  

  // The loop's reference container variable is the container that
  // has been specified within the forall statement. We keep track
  // of not only the mesh's identifier info but also the var decl.
  // This is primarily the data gathered at parsing and during semantic
  // checks -- it is useful to keep around for both analysis, optimization
  // and codegen passes...
  IdentifierInfo  *ContainerRefVarInfo;
  VarDecl         *ContainerVarDecl;
  SourceLocation  RenderallKWLoc, LParenLoc, RParenLoc;
  mutable bool Last;
};


// ----- RenderallMeshStmt -- renderall mesh statement
//
// renderall <mesh-loc-kw> <ref-element> in <ref-mesh> {
//   ...
// }
//
// where:
//
//   mesh-loc-kw  - provides the mesh location (elements) to operate over.
//                  This is a keyword that identifies 'cells', 'edges',
//                  'vertices', or 'faces'.
//
//   ref-element  - the reference variable for each field location element in
//                  the mesh being operated on.  This should be a valid
//                  identifier (variable name).  It takes on the value of each
//                  element (cell, edge, vertex, or face) in the mesh within
//                  the body of the 'renderall'.
//
//   ref-mesh     - the mesh instance that should be computed over.
//
// Predicated form:
//
//   renderall <mesh-loc-kw> <ref-element> in <ref-mesh> (predicated-expr) {
//     ...
//   }
//
class RenderallMeshStmt : public RenderallStmt {

public:

private:
  // The mesh location/element we're looping over -- this provides us
  // information about looping over cells, edges, faces, etc.  Is the
  // isOver*() member functions for helpers to determine the element
  // type.
  MeshElementType MeshElementRef;

  // In addition to the container information held in our parent class we also
  // capture the mesh type that the loop is operating over.  Note at this point
  // we are dealing with a generic (base) mesh type and not a specific case
  // (e.g. uniform or rectilinear).  Any operations over the renderall should
  // make sure appropriate steps are taken for the various mesh types (and
  // safely cast this type as needed to the appropriate subclass).
  const MeshType* MeshRefType;

public:

  RenderallMeshStmt(MeshElementType RefElement,
                    IdentifierInfo* RefVarInfo,
                    IdentifierInfo* MeshInfo,
                    IdentifierInfo* RenderTargetInfo, 
                    VarDecl* MeshVarDecl,
                    VarDecl* RenderTargetVarDecl, 
                    const MeshType* MT,
                    SourceLocation  RenderallLocation,
                    Stmt *Body);

  RenderallMeshStmt(MeshElementType RefElement,
                    IdentifierInfo* RefVarInfo,
                    IdentifierInfo* MeshInfo,
                    IdentifierInfo* RenderTargetInfo, 
                    VarDecl* MeshVarDecl,
                    VarDecl* RenderTargetVarDecl,                    
                    const MeshType* MT,
                    SourceLocation  RenderallLocation,
                    Stmt *Body, Expr* Predicate,
                    SourceLocation LeftParenLoc,
                    SourceLocation RightParenLoc);

  explicit RenderallMeshStmt(EmptyShell Empty)
    : RenderallStmt(ForallMeshStmtClass, Empty) { }

  RenderallMeshStmt(StmtClass SC, EmptyShell Empty)
    : RenderallStmt(SC, Empty) { }


  // ===--- Mesh details ---===

  MeshElementType getMeshElementRef() const {
    return MeshElementRef;
  }

  void setMeshElementRef(MeshElementType ref) {
    MeshElementRef = ref;
  }

  void setMeshElementRef(tok::TokenKind &token);

  // Determine which mesh elements the statement is operating over...
  bool isOverCells() const    { return MeshElementRef == Cells; }
  bool isOverEdges() const    { return MeshElementRef == Edges; }
  bool isOverVertices() const { return MeshElementRef == Vertices; }
  bool isOverFaces() const    { return MeshElementRef == Faces; }

  // Determine which mesh type the statement is operating over...
  bool isUniformMesh() const;
  bool isRectilinearMesh() const;
  bool isStructuredMesh() const;
  bool isUnstructuredMesh() const;

  IdentifierInfo* getMeshInfo() {
    return RenderallStmt::getContainerVarInfo();
  }

  const IdentifierInfo* getMeshInfo() const {
    return RenderallStmt::getContainerVarInfo();
  }

  VarDecl* getMeshVarDecl() {
    return RenderallStmt::getContainerVarDecl();
  }

  const VarDecl* getMeshVarDecl() const {
    return RenderallStmt::getContainerVarDecl();
  }

  const MeshType* getMeshType() const { return MeshRefType; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == RenderallMeshStmtClass;
  }

  static bool classof(const RenderallMeshStmt *) { return true; }
};
 
class ScoutStmt : public Stmt {
public:
  enum ScoutStmtKind{
    FrameCapture,
    Plot
  };
  
  ScoutStmt(ScoutStmtKind K)
  : Stmt(ScoutStmtClass),
    Kind(K){}
  
  virtual ~ScoutStmt(){}
  
  explicit ScoutStmt(EmptyShell Empty)
  : Stmt(ScoutStmtClass, Empty){}
  
  static bool classof(const Stmt *T){
    return T->getStmtClass() == ScoutStmtClass;
  }
  
  SourceLocation getLocStart() const LLVM_READONLY{
    return StartLoc;
  }
  
  void setLocStart(SourceLocation Loc){
    StartLoc = Loc;
  }
  
  SourceLocation getLocEnd() const LLVM_READONLY{
    return EndLoc;
  }
  
  void setLocEnd(SourceLocation Loc){
    EndLoc = Loc;
  }
  
  ScoutStmtKind kind() const{
    return Kind;
  }
  
  static bool classof(const ScoutStmt *){ return true; }
  
  virtual child_range children(){
    return child_range();
  }
  
private:
  ScoutStmtKind Kind;
  SourceLocation StartLoc;
  SourceLocation EndLoc;
};

class FrameCaptureStmt : public ScoutStmt{
public:
  FrameCaptureStmt(const VarDecl* FV, SpecObjectExpr* S)
  : ScoutStmt(FrameCapture),
  FrameVar(FV),
  Spec(S){}
  
  const VarDecl* getFrameVar() const{
    return FrameVar;
  }
  
  const SpecObjectExpr* getSpec() const{
    return Spec;
  }
  
private:
  const VarDecl* FrameVar;
  SpecObjectExpr* Spec;
};
  
class PlotStmt : public ScoutStmt{
public:
  
  using CallMap = std::map<const CallExpr*, uint32_t>;
  using VarMap = std::map<std::string, std::pair<const VarDecl*, uint32_t>>;
  using VarIdMap = std::map<const VarDecl*, uint32_t>;
  
  PlotStmt(const FrameDecl* FD, const VarDecl* FV, const VarDecl* RV, SpecObjectExpr* S)
  : ScoutStmt(Plot),
  Frame(FD),
  FrameVar(FV),
  RenderTargetVar(RV),
  Spec(S),
  nextVarId_(65536){}

  const FrameDecl* getFrameDecl() const{
    return Frame;
  }
  
  const VarDecl* getFrameVar() const{
    return FrameVar;
  }
  
  const VarDecl* getRenderTargetVar() const{
    return RenderTargetVar;
  }
  
  const SpecObjectExpr* getSpec() const{
    return Spec;
  }
  
  void addVar(const VarDecl* v);
  
  const VarMap& varMap() const{
    return VMap;
  }
  
  uint32_t getVarId(const VarDecl* v) const{
    for(auto& itr : VMap){
      if(itr.second.first == v){
        return itr.second.second;
      }
    }
    
    return 0;
  }
  
  uint32_t getVarId(const CallExpr* c) const{
    auto itr = CMap.find(c);
    if(itr == CMap.end()){
      return 0;
    }
    
    return itr->second;
  }
  
  uint32_t nextVarId() const{
    return nextVarId_++;
  }
  
  uint32_t addCall(const CallExpr* c) const{
    uint32_t varId = nextVarId_++;
    assert(CMap.find(c) == CMap.end());
    CMap.insert({c, varId});
    return varId;
  }
  
  const CallMap& callMap() const{
    return CMap;
  }
  
  uint32_t addExtVar(const VarDecl* v) const{
    uint32_t varId = nextVarId_++;
    ExtVarMap[v] = varId;
    return varId;
  }
  
  uint32_t getExtVarId(const VarDecl* v) const{
    auto itr = ExtVarMap.find(v);
    if(itr == ExtVarMap.end()){
      return 0;
    }
    
    return itr->second;
  }
  
  const VarIdMap& extVarMap() const{
    return ExtVarMap;
  }
  
private:
  const FrameDecl* Frame;
  const VarDecl* FrameVar;
  const VarDecl* RenderTargetVar;
  SpecObjectExpr* Spec;
  
  mutable uint32_t nextVarId_;
  
  VarMap VMap;
  mutable CallMap CMap;
  
  mutable VarIdMap ExtVarMap;
};
  
// +==========================================================================+

}  // end namespace clang

#endif
