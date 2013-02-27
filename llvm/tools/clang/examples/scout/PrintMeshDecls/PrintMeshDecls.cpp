#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

  // --- MeshDeclVisitor
  //
  class MeshDeclVisitor : public RecursiveASTVisitor<MeshDeclVisitor> {

   public:
/*
    bool VisitDecl(Decl *D) {
      llvm::errs() << "\tvisit a decl (" << D->getDeclKindName() << "):\n";
      D->dump(llvm::errs());
      return true;
    }
*/
    bool VisitStmt(Stmt *S) {
      if (isa<DeclStmt>(S)) {
        S->dumpColor();
      }
      return true;
    }

    bool VisitType(Type *T) {
      if (T->isMeshType()) {
        llvm::errs() << "\tvisit a mesh type.\n";
      }
      
      return true;
    }        

  };
  

  // --- PrintMeshDeclsConsumer
  //
  class PrintMeshDeclsConsumer : public ASTConsumer {

   public:

    void HandleTranslationUnit(ASTContext &Context) {
      llvm::errs() << "handling a translation unit...\n";
      Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

   private:
    MeshDeclVisitor Visitor;
  };
    
  // --- PrintMeshDeclsAction
  // 
  class PrintMeshDeclsAction: public PluginASTAction {
    
   protected:
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
      return new PrintMeshDeclsConsumer;
    };

    bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string>& args) {
      return true;
    }

  };
  
}

static FrontendPluginRegistry::Add<PrintMeshDeclsAction>
X("print-mesh-decls", "print mesh declarations");
