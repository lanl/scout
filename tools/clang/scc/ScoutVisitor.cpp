/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 
#include <sstream>
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "llvm/Support/Host.h"
#include "ScoutVisitor.h"
#include "scout/Runtime/opengl/glSDL.h" // for window height/width

using namespace clang;

// hack for testing
// if SC_USE_RT_REWRITER is set then use rewriter to inject
// runtime calls. Otherwise inject comment, and use InsertCPPCode() in parser
#ifdef SC_USE_RT_REWRITER
#define USE_RT_REWRITER 1
#else
#define USE_RT_REWRITER 0
#endif

// add dimensions to an argument string
void ScoutVisitor::addDims(std::string* s, MeshType::MeshDimensionVec dims) {
  for(size_t i = 0; i < 3; ++i){
    if(i > 0){
      *s += ", ";
    }

    if(i >= dims.size()){
      *s += "0";
    } else {
      *s += rewriter_.ConvertToString(dims[i]);
    }
  }
}

bool ScoutVisitor::VisitStmt(Stmt* s) {

  // add comments to 'forall' statements.
  if(isa<ForAllStmt>(s)) {
    ForAllStmt* fas = cast<ForAllStmt>(s);
    rewriter_.InsertText(fas->getLocStart(),
                         "/*begin_forall();*/", true, true);

    rewriter_.InsertText(fas->getLocEnd().getLocWithOffset(1),
                         "/*end_forall();*/", true, true);
  }

  // renderall statement runtime code rewrite
  // this will not work correctly in the multifile case
  if(isa<RenderAllStmt>(s)) {
    RenderAllStmt* ras = cast<RenderAllStmt>(s);
    std::string begin, end;

    if (!USE_RT_REWRITER) {
      begin = "/*";
      end = "/*";
    }

    begin += "__scrt_renderall_uniform_begin(";

    // Get dimensions of the mesh and insert as arguments to the call
    const MeshType *MT = cast<MeshType>(ras->getMeshType());
    MeshType::MeshDimensionVec dims = MT->dimensions();
    addDims(&begin, dims);

    begin += ");";
    if (!USE_RT_REWRITER) begin += "*/";

    rewriter_.InsertText(ras->getLocStart(),
                         begin, true, true);


    end += "__scrt_renderall_end();";
    if (!USE_RT_REWRITER) end += "*/";

    rewriter_.InsertText(ras->getLocEnd().getLocWithOffset(1),
                         end, true, true);
  }

  return true;
}

bool ScoutVisitor::VisitFunctionDecl(FunctionDecl* f) {
  // add call to runtime init at top of main
  if (f->hasBody() && f->isMain()) {
    Stmt* s = f->getBody();
    std::string scinit;

    if (!USE_RT_REWRITER) scinit = "/*";

    if(Clang_->getLangOpts().ScoutNvidiaGPU) {
      scinit += "__scrt_init(ScoutGPUCUDA);";
    } else if(Clang_->getLangOpts().ScoutAMDGPU){
      scinit += "__scrt_init(ScoutGPUOpenCL);";
    } else {
      scinit += "__scrt_init(ScoutGPUNone);";
    }
    if (!USE_RT_REWRITER) scinit += "*/";
    rewriter_.InsertText(s->getLocStart().getLocWithOffset(1),
                         scinit, true, true);
  }
  return true;
}

bool ScoutVisitor::VisitVolumeRenderAllStmt(VolumeRenderAllStmt* vras) {

  std::string bc;

  bc = "__scrt_renderall_volume_init(__volren_gcomm,";

  // Get dimensions of the mesh and insert as arguments to the call
  const MeshType *MT = cast<MeshType>(vras->getMeshType());
  MeshType::MeshDimensionVec dims = MT->dimensions();
  addDims(&bc, dims);

  // window width/height arguments
  bc += ", __scrt_initial_window_width, __scrt_initial_window_height, ";

  // camera argument
  IdentifierInfo* CameraII = vras->getCamera();

  std::string cameraName;
  if (CameraII != 0) {
    cameraName = "&" + CameraII->getName().str();
  } else {
        cameraName = "NULL";
  }

  bc += cameraName + ", ";
  
  // One argument to the call is an apple block to hold body of renderall (transfer function closure)
  bc += "^int(scout::block_t* block, scout::point_3d_t* pos, scout::rgba_t& color){";

  // Insert code to get values of data fields for transfer function closure
  size_t FieldCount = 0;
  const MeshDecl* MD = MT->getDecl();
  
  for(MeshDecl::mesh_field_iterator FI = MD->mesh_field_begin(),
      FE = MD->mesh_field_end(); FI != FE; ++FI){
    MeshFieldDecl* FD = *FI;
    if(! FD->isImplicit() && FD->isCellLocated()) {
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
  vras->getBody()->printPretty(S, 0, printingPolicy);
  bc += S.str();

  // Finish the sc_init_volume_renderall function call.
  bc += "return 1;});\n";
  rewriter_.InsertText(vras->getLocStart(), bc);

  return true;
}
  
