//===----------------------------------------------------------------------===//
//
// ndm - This file defines the Scout AST viewer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_AST_VIEW_SCOUT_H
#define LLVM_CLANG_AST_AST_VIEW_SCOUT_H

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

  void addDeclGroup(DeclGroupRef declGroup);

  void generateGraphViz(const std::string& outFilePath);

private:
  typedef std::vector<Decl*> DeclVec;

  DeclVec declVec;
  Sema& sema;
};

} // end namespace clang

#endif // LLVM_CLANG_AST_AST_VIEW_SCOUT_H

