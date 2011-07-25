//===----------------------------------------------------------------------===//
//
// ndm - This file implements the Scout AST viewer functionality used
// by the scc command for debugging purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTViewScout.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/SourceManager.h"

#include <iostream>

using namespace clang;

ASTViewScout::ASTViewScout(Sema& sema)
: sema(sema){

}

ASTViewScout::~ASTViewScout(){

}

void ASTViewScout::addDeclGroup(DeclGroupRef declGroup){
  
  DeclGroupRef::iterator itr = declGroup.begin();
  while(itr != declGroup.end()){
    Decl* decl = *itr;
 
    if(sema.getSourceManager().isFromMainFile(decl->getLocation())){
    
      if(FunctionDecl* fd = dyn_cast<FunctionDecl>(decl)){
        fd->dump();

        if(fd->hasBody()){
          Stmt* body = fd->getBody();
          body->dump();
        }
      }
      else{

      }
    }
    ++itr;
  }
}

void ASTViewScout::generateGraphViz(const std::string& filePath){

}

