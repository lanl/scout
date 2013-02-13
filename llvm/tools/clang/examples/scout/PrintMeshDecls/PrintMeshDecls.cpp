#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {

  // --- PrintMeshDeclsConsumer
  //
  class PrintMeshDeclsConsumer : public ASTConsumer {

   public:

    PrintMeshDeclsConsumer(CompilerInstance &ci)
    : ASTConsumer(),
      CI(ci)
    {
      llvm::errs() << "building instance of print mesh decls plug-in...\n";
    }

    bool HandleTopLevelDecl(DeclGroupRef DG) {
      llvm::errs() << "top-level-decl:\n";
      for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
        const Decl *D = *i;
        if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
          llvm::errs() << "\t'" << ND->getNameAsString() << "'\n";
      }
      
      return true;
    }
    
    void HandleTagDeclDefinition(TagDecl *D)  {
      llvm::errs() << "tag-decl: " << D->getNameAsString() << "\n";      
      if (const MeshDecl *MD = dyn_cast<MeshDecl>(D)) {
        llvm::errs() << "mesh-decl: " << MD->getNameAsString() << "\n";
      }
    }

    void HandleInterestingDecl(DeclGroupRef DG)  {
      llvm::errs() << "interesting-decl:\n";
      for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
        const Decl *D = *i;
        if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
          llvm::errs() << "\t'" << ND->getNameAsString() << "'\n";
      }
    }
    
   private:
    CompilerInstance &CI;
  };


  // --- PrintMeshDeclsAction
  // 
  class PrintMeshDeclsAction: public PluginASTAction {
    
   protected:
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
      return new PrintMeshDeclsConsumer(CI);
    };

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string>& args) {
      return true;
    }
  };
  
}

static FrontendPluginRegistry::Add<PrintMeshDeclsAction>
X("print-mesh-decls", "print mesh declarations");
