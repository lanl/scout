//===----------------------------------------------------------------------===//
//
// SCOUTCODE ndm - This file defines the Scout AST viewer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_AST_VIEW_SCOUT_H
#define LLVM_CLANG_PARSE_AST_VIEW_SCOUT_H

#include <string>
#include <vector>

namespace clang{
  
class DeclGroupRef;
class Decl;
class Sema;
  
class ASTViewScout{
public:
  ASTViewScout(Sema& sema);

  ~ASTViewScout();

  void outputGraphviz(DeclGroupRef declGroup);

private:
  Sema& sema_;
};

} // end namespace clang

#endif // LLVM_CLANG_PARSE_AST_VIEW_SCOUT_H
// ENDSCOUTCODE

