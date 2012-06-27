//===-- cc1_main.cpp - Clang CC1 Compiler Frontend ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/LinkAllPasses.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Rewrite/Rewriter.h"
#include "clang/Rewrite/Rewriters.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

// AST walking -------------------------------------------------------

// scout - AST visitors - to walk the AST and insert code around
// various parts of the Scout AST nodes we are interested in
class ScoutVisitor : public RecursiveASTVisitor<ScoutVisitor>
{
 public:
  ScoutVisitor(Rewriter& rewriter)
  : rewriter_(rewriter){

  }

  bool VisitStmt(Stmt* s){
    // Only care about 'forall' statements.
    if(isa<ForAllStmt>(s)){
      ForAllStmt* fas = cast<ForAllStmt>(s);
      rewriter_.InsertText(fas->getLocStart(),
                           "begin_forall();\n", true, true);
      
      rewriter_.InsertText(fas->getLocEnd().getLocWithOffset(1),
                           "\nend_forall();\n", true, true);      
    }

    return true;
  }

  bool VisitVolumeRenderAllStmt(VolumeRenderAllStmt* vras) {
    
    std::string bc;
        
    bc = "__sc_init_volume_renderall(";
    
    // Get dimensions of the mesh and insert as arguments to the call
    const MeshType *MT = cast<MeshType>(vras->getMeshType());
    MeshType::MeshDimensionVec dims = MT->dimensions();
    
    for(size_t i = 0; i < 3; ++i){
      if(i > 0){
        bc += ", ";
      }
      
      if(i >= dims.size()){
        bc += "0";
      }
      else{
        bc += rewriter_.ConvertToString(dims[i]);
      }
    }
    
    // One argument to the call is an apple block to hold body of renderall (transfer function closure)
    bc += ", 1024, 1024, NULL, \
    ^int(scout::block_t* block, scout::point_3d_t* pos, scout::rgba_t& color){";
    // Insert code to get values of data fields for transfer function closure
    size_t FieldCount = 0;
    const MeshDecl* MD = MT->getDecl();
    
    for(MeshDecl::field_iterator FI = MD->field_begin(),
        FE = MD->field_end(); FI != FE; ++FI){
      FieldDecl* FD = *FI;
      if(!FD->isMeshImplicit() &&
         FD->meshFieldType() == FieldDecl::FieldCells){
        bc += FD->getType().getAsString() + " " + FD->getName().str() + ";";
        std::stringstream fieldCountStr;
        fieldCountStr << FieldCount;
        bc +=
        "if (scout::block_get_value(block, " + fieldCountStr.str()
        + ", pos->x3d, pos->y3d, pos->z3d, &" + FD->getName().str() + ") == HPGV_FALSE) \
        { \
        return 0; \
        } ";
        ++FieldCount;
      }
    }
    
    // Add user's renderall body to the string 
    std::string SStr;
    llvm::raw_string_ostream S(SStr);
    LangOptions LO = rewriter_.getLangOpts();
    PrintingPolicy printingPolicy(LO);
    printingPolicy.SuppressMemberBase = true;
    vras->getBody()->printPretty(S, 0, printingPolicy);
    bc += S.str();
    
    // Finish the sc_init_volume_renderall function call.
    bc += "return 1;});\n";
    rewriter_.InsertText(vras->getLocStart(), bc);
    
    return true;
  }
  
 private:
  Rewriter& rewriter_;
};

// The ASTConsumer interface is for reading the AST produced by the parser.
class ScoutConsumer : public ASTConsumer
{
 public:
  ScoutConsumer(Rewriter& rewriter)
  : visitor_(rewriter){

  }

  // Override the method that gets called for each parsed top-level
  // declaration.
  virtual bool HandleTopLevelDecl(DeclGroupRef dr) {
    for(DeclGroupRef::iterator b = dr.begin(), e = dr.end(); b != e; ++b){
      // Traverse the declaration using our AST visitor.
      visitor_.TraverseDecl(*b);
    }
    
    return true;
  }


 private:
  ScoutVisitor visitor_;
};

// end AST walking ------------------------------------------------

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

// FIXME: Define the need for this testing away.
static int cc1_test(DiagnosticsEngine &Diags,
                    const char **ArgBegin, const char **ArgEnd) {
  using namespace clang::driver;

  llvm::errs() << "cc1 argv:";
  for (const char **i = ArgBegin; i != ArgEnd; ++i)
    llvm::errs() << " \"" << *i << '"';
  llvm::errs() << "\n";

  // Parse the arguments.
  OptTable *Opts = createDriverOptTable();
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList *Args = Opts->ParseArgs(ArgBegin, ArgEnd,
                                       MissingArgIndex, MissingArgCount);

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(clang::diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Dump the parsed arguments.
  llvm::errs() << "cc1 parsed options:\n";
  for (ArgList::const_iterator it = Args->begin(), ie = Args->end();
       it != ie; ++it)
    (*it)->dump();

  // Create a compiler invocation.
  llvm::errs() << "cc1 creating invocation.\n";
  CompilerInvocation Invocation;
  if (!CompilerInvocation::CreateFromArgs(Invocation, ArgBegin, ArgEnd, Diags))
    return 1;

  // Convert the invocation back to argument strings.
  std::vector<std::string> InvocationArgs;
  Invocation.toArgs(InvocationArgs);

  // Dump the converted arguments.
  SmallVector<const char*, 32> Invocation2Args;
  llvm::errs() << "invocation argv :";
  for (unsigned i = 0, e = InvocationArgs.size(); i != e; ++i) {
    Invocation2Args.push_back(InvocationArgs[i].c_str());
    llvm::errs() << " \"" << InvocationArgs[i] << '"';
  }
  llvm::errs() << "\n";

  // Convert those arguments to another invocation, and check that we got the
  // same thing.
  CompilerInvocation Invocation2;
  if (!CompilerInvocation::CreateFromArgs(Invocation2, Invocation2Args.begin(),
                                          Invocation2Args.end(), Diags))
    return 1;

  // FIXME: Implement CompilerInvocation comparison.
  if (true) {
    //llvm::errs() << "warning: Invocations differ!\n";

    std::vector<std::string> Invocation2Args;
    Invocation2.toArgs(Invocation2Args);
    llvm::errs() << "invocation2 argv:";
    for (unsigned i = 0, e = Invocation2Args.size(); i != e; ++i)
      llvm::errs() << " \"" << Invocation2Args[i] << '"';
    llvm::errs() << "\n";
  }

  return 0;
}

int cc1_main(const char **ArgBegin, const char **ArgEnd,
             const char *Argv0, void *MainAddr) {
  OwningPtr<CompilerInstance> Clang(new CompilerInstance());
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Run clang -cc1 test.
  if (ArgBegin != ArgEnd && StringRef(ArgBegin[0]) == "-cc1test") {
    DiagnosticsEngine Diags(DiagID, new TextDiagnosticPrinter(llvm::errs(), 
                                                       DiagnosticOptions()));
    return cc1_test(Diags, ArgBegin + 1, ArgEnd);
  }

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, DiagsBuffer);
  bool Success;
  Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                               ArgBegin, ArgEnd, Diags);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics(ArgEnd - ArgBegin, const_cast<char**>(ArgBegin));
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                  static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return 1;

  // ------------------------------------
  // scout - hook into the Compiler instance to pass the AST consumer
  // and rewriter

  Rewriter rewriter;
  ScoutConsumer consumer(rewriter);

  Clang->setScoutASTConsumer(&consumer);
  Clang->setScoutRewriter(&rewriter);
  // ------------------------------------

  // Execute the frontend actions.
  Success = ExecuteCompilerInvocation(Clang.get());

  // scout - get the modified code and output
  SourceManager &sourceMgr = Clang->getSourceManager();

  const RewriteBuffer* rewriteBuffer =
    rewriter.getRewriteBufferFor(sourceMgr.getMainFileID());

  llvm::outs() << std::string(rewriteBuffer->begin(), rewriteBuffer->end());

  // ---------------------------------------------

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    if (llvm::AreStatisticsEnabled() || Clang->getFrontendOpts().ShowStats)
      llvm::PrintStatistics();
    Clang.take();
    return !Success;
  }

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return !Success;
}
