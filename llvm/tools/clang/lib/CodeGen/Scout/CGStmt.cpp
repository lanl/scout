/*
 *
 * ###########################################################################
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
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
 *
 */

#include "CodeGenFunction.h"
#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Dominators.h"

#include <stdio.h>
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/AST/Decl.h"
#include "CGBlocks.h"

#include "Scout/CGScoutRuntime.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"

using namespace clang;
using namespace CodeGen;


static const char *DimNames[]   = { "width", "height", "depth" };
static const char *IndexNames[] = { "x", "y", "z"};

// We use 'IRNameStr' to hold the generated names we use for
// various values in the IR building.  We've added a static
// buffer to avoid the need for a lot of fine-grained new and
// delete calls...  We're likely safe with 160 character long
// strings.
static char IRNameStr[160];


llvm::Value *CodeGenFunction::TranslateExprToValue(const Expr *E) {

  switch(E->getStmtClass()) {
    case Expr::IntegerLiteralClass:
    case Expr::BinaryOperatorClass:
      return EmitScalarExpr(E);
    default:
      return Builder.CreateLoad(EmitLValue(E).getAddress());
  }
}

llvm::Value *CodeGenFunction::GetMeshBaseAddr(const ForallMeshStmt &S) {
  const VarDecl *MeshVarDecl = S.getMeshVarDecl();
  llvm::Value *BaseAddr = 0;

  if (MeshVarDecl->hasGlobalStorage()) {
    BaseAddr = Builder.CreateLoad(CGM.GetAddrOfGlobalVar(MeshVarDecl));
  } else {
    BaseAddr = LocalDeclMap[MeshVarDecl];
    if (MeshVarDecl->getType().getTypePtr()->isReferenceType()) {
      BaseAddr = Builder.CreateLoad(BaseAddr);
    }
  }

  return BaseAddr;
}

//SC_TODO: remove Renderall/Forall duplication here.
llvm::Value *CodeGenFunction::GetMeshBaseAddr(const RenderallMeshStmt &S) {
  const VarDecl *MeshVarDecl = S.getMeshVarDecl();
  llvm::Value *BaseAddr = 0;

  if (MeshVarDecl->hasGlobalStorage()) {
    BaseAddr = Builder.CreateLoad(CGM.GetAddrOfGlobalVar(MeshVarDecl));
  } else {
    BaseAddr = LocalDeclMap[MeshVarDecl];
    if (MeshVarDecl->getType().getTypePtr()->isReferenceType()) {
      BaseAddr = Builder.CreateLoad(BaseAddr);
    }
  }

  return BaseAddr;
}



// ----- EmitforallMeshStmt
//
// Forall statements are transformed into a nested loop
// structure (with a loop per rank of the mesh) that
// uses a single linear address variable.  In other words,
// a structure that looks something like this:
//
//  linear_index = 0;
//  for(z = 0; z < mesh.depth; z++)
//    for(y = 0; y < mesh.height; y++)
//      for(x = 0; x < mesh.width; x++) {
//        ... body goes here ...
//        linear_index++;
//      }
//
// At this point in time we don't kill ourselves in trying
// to over optimize the loop structure (partially in hope
// that our restrictions at the language level will help the
// standard optimizers do an OK job).  That said, there is
// likely room for improvement here...   At this point we're
// more interested in code readability than performance.
//
// The guts of this code follow the techniques used in the
// EmitForStmt member function.
//
// SC_TODO - we need to find a way to share the loop index
// across code gen routines.
//
// SC_TODO - need to eventually add support for predicated
// induction variable ranges.
//
// SC_TODO - need to handle cases with edge and vertex
// fields (the implementation below is cell centric).
//

void CodeGenFunction::EmitForallMeshStmt(const ForallMeshStmt &S) {
  const VarDecl* VD = S.getMeshVarDecl();

  unsigned int rank = S.getMeshType()->rankOf();

  VertexIndex = 0;

  // handle nested forall, e.g: forall vertices within a forall cells
  if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(VD)){
    VD = IP->getMeshVarDecl();

    ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();
    ImplicitMeshParamDecl::MeshElementType ET = IP->getElementType();

    if(FET == ForallMeshStmt::Vertices){
      assert(ET == ImplicitMeshParamDecl::Cells &&
             "EmitForAllMeshStmt element type nesting combination not implemented");

      llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.vertices.entry");
      (void)EntryBlock; //suppress warning

      llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
      llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
      llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
      llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);

      llvm::Value* vertexPosPtr = Builder.CreateAlloca(Int32Ty, 0, "vertex.pos.ptr");
      VertexIndex = Builder.CreateAlloca(Int32Ty, 0, "vertex.index.ptr");

      Builder.CreateStore(Zero, vertexPosPtr);

      llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");
      llvm::Value* indVar =  Builder.CreateLoad(InductionVar[3], "indVar");

      llvm::Value* width1;
      llvm::Value* idvw;

      if(rank > 1){
        width1 =  Builder.CreateAdd(width, One);
        idvw = Builder.CreateUDiv(indVar, width, "idvw");
      }

      llvm::BasicBlock *LoopBlock = createBasicBlock("forall.vertices.loop");
      Builder.CreateBr(LoopBlock);

      EmitBlock(LoopBlock);

      llvm::Value* vertexPos = Builder.CreateLoad(vertexPosPtr, "vertex.pos");

      if(rank > 1){
        llvm::Value* v1 = Builder.CreateUDiv(vertexPos, Two);
        llvm::Value* v2 = Builder.CreateMul(v1, width1);
        llvm::Value* v3 = Builder.CreateURem(vertexPos, Two);
        llvm::Value* v4 = Builder.CreateAdd(v2, v3);
        llvm::Value* v5 = Builder.CreateAdd(v4, indVar);
        llvm::Value* newVertexIndex = Builder.CreateAdd(v5, idvw, "vertex.index.new");

        Builder.CreateStore(newVertexIndex, VertexIndex);
      }
      else{
        llvm::Value* newVertexIndex = Builder.CreateAdd(vertexPos, indVar, "vertex.index.new");
        Builder.CreateStore(newVertexIndex, VertexIndex);
      }

      llvm::Value* newVertexPos = Builder.CreateAdd(vertexPos, One);
      Builder.CreateStore(newVertexPos, vertexPosPtr);

      EmitStmt(S.getBody());

      VertexIndex = 0;

      llvm::Value* Cond;

      if(rank > 1){
        Cond = Builder.CreateICmpSLT(vertexPos, Three, "cond");
      }
      else{
        Cond = Builder.CreateICmpSLT(vertexPos, One, "cond");
      }

      llvm::BasicBlock *ExitBlock = createBasicBlock("forall.vertices.exit");
      Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
      EmitBlock(ExitBlock);
      return;
    }
    else{
      assert(false && "EmitForAllMeshStmt element type nesting combination not implemented");
    }
  }

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value *ConstantOne  = llvm::ConstantInt::get(Int32Ty, 1);

  //used by builtins to keep track of which dims are used
  for(unsigned int i = 0; i < rank; i++) {
    DimExists[i] = ConstantOne;
  }

  // Track down the mesh meta data. 
  //llvm::NamedMDNode *MeshMD = CGM.getModule().getNamedMetadata("scout.meshmd");
  //assert(MeshMD != 0 && "unable to find module-level mesh metadata!");
  //llvm::errs() << "forall mesh type name = '" << S.getMeshVarDecl()->getTypeSourceInfo()->getType().getTypePtr()->getTypeClassName() << "'\n";
  ResetVars();

  if ((VD->hasLinkage() || VD->isStaticDataMember()) && VD->getTLSKind() != VarDecl::TLS_Dynamic) {
    llvm::Value* V = CGM.GetAddrOfGlobalVar(VD);
    llvm::Value* VP = Builder.CreateLoad(V);
    LocalDeclMap[VD] = VP;
  }

  //need a marker for start of Forall for CodeExtraction
  llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");

  // Create the induction variables for eack rank.
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "forall.induct.%s.ptr", IndexNames[i]);
    InductionVar[i] = Builder.CreateAlloca(Int32Ty, 0, IRNameStr);
    //zero-initialize induction var
    Builder.CreateStore(ConstantZero, InductionVar[i]);

  }
  // create linear loop index as 4th element and zero-initialize.
  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.linearidx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[3]);

  EmitForallMeshLoop(S, 3);

  // reset Loopbounds and induction var
  // so width/height etc can't be called after forall
  ResetVars();

  //need a marker for end of Forall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("forall.exit");

  // Extract Blocks to function and replace w/ call to function
  ExtractRegion(entry, exit, "ForallMeshFunction");
}


//generate one of the nested loops
void CodeGenFunction::EmitForallMeshLoop(const ForallMeshStmt &S, unsigned r) {
  RegionCounter Cnt = getPGORegionCounter(&S);
  (void)Cnt; //suppress warning 
 
  const VarDecl* VD = S.getMeshVarDecl();
  if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(VD)){
    VD = IP->getMeshVarDecl();
  }

  llvm::Value *MeshBaseAddr = LocalDeclMap[VD];
  llvm::StringRef MeshName = S.getMeshType()->getName();

  // find number of fields
  MeshDecl* MD =  S.getMeshType()->getDecl();
  unsigned int nfields = MD->fields();

  CGDebugInfo *DI = getDebugInfo();

  llvm::Value *ConstantZero  = 0;
  llvm::Value *ConstantOne   = 0;
  ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
  ConstantOne  = llvm::ConstantInt::get(Int32Ty, 1);

  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[r-1]);

  sprintf(IRNameStr, "forall.%s.end", DimNames[r-1]);
  JumpDest LoopExit = getJumpDestInCurrentScope(IRNameStr);
  RunCleanupsScope ForallScope(*this);

  if (DI)
    DI->EmitLexicalBlockStart(Builder, S.getSourceRange().getBegin());

  // Extract the loop bounds from the mesh for this rank, this requires
  // a GEP from the mesh and a load from returned address...
  // note: width/height depth are stored after mesh fields
  sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[r-1]);
  LoopBounds[r-1] = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+r-1, IRNameStr);
  sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), DimNames[r-1]);
  llvm::LoadInst *LoopBound  = Builder.CreateLoad(LoopBounds[r-1], IRNameStr);

  // Next we create a block that tests the induction variables value to
  // the rank's dimension.
  sprintf(IRNameStr, "forall.cond.%s", DimNames[r-1]);
  JumpDest Continue = getJumpDestInCurrentScope(IRNameStr);
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  RunCleanupsScope ConditionScope(*this);

  sprintf(IRNameStr, "forall.induct.%s", IndexNames[r-1]);
  llvm::LoadInst *IVar = Builder.CreateLoad(InductionVar[r-1], IRNameStr);

  sprintf(IRNameStr, "forall.done.%s", IndexNames[r-1]);
  llvm::Value *CondValue = Builder.CreateICmpSLT(IVar,
                                                 LoopBound,
                                                 IRNameStr);

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();

  // If there are any cleanups between here and the loop-exit
  // scope, create a block to stage a loop exit along.  (We're
  // following Clang's lead here in generating a for loop.)
  if (ForallScope.requiresCleanups()) {
    sprintf(IRNameStr, "forall.cond.cleanup.%s", DimNames[r-1]);
    ExitBlock = createBasicBlock(IRNameStr);
  }

  llvm::BasicBlock *LoopBody = createBasicBlock(IRNameStr);
  Builder.CreateCondBr(CondValue, LoopBody, ExitBlock);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);

  sprintf(IRNameStr, "forall.incblk.%s", IndexNames[r-1]);
  Continue = getJumpDestInCurrentScope(IRNameStr);

  // Store the blocks to use for break and continue.

  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));
  if (r == 1) {  // This is our innermost rank, generate the loop body.
    EmitStmt(S.getBody());

    // Increment the loop index stored as last element of InductionVar
    llvm::LoadInst* liv = Builder.CreateLoad(InductionVar[3], "forall.linearidx");
    llvm::Value *IncLoopIndexVar = Builder.CreateAdd(liv,
                                                     ConstantOne,
                                                     "forall.linearidx.inc");

    Builder.CreateStore(IncLoopIndexVar, InductionVar[3]);
  } else { // generate nested loop
    EmitForallMeshLoop(S, r-1);
  }

  EmitBlock(Continue.getBlock());

  sprintf(IRNameStr, "forall.induct.%s", IndexNames[r-1]);
  llvm::LoadInst* iv = Builder.CreateLoad(InductionVar[r-1], IRNameStr);

  sprintf(IRNameStr, "forall.inc.%s", IndexNames[r-1]);
  llvm::Value *IncInductionVar = Builder.CreateAdd(iv,
                                                   ConstantOne,
                                                   IRNameStr);

  Builder.CreateStore(IncInductionVar, InductionVar[r-1]);

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();

  EmitBranch(CondBlock);
  ForallScope.ForceCleanup();

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getSourceRange().getEnd());

  EmitBlock(LoopExit.getBlock(), true);
}


// reset Loopbounds and induction var
void CodeGenFunction::ResetVars(void) {
    LoopBounds.clear();
    InductionVar.clear();
    for(unsigned int i = 0; i < 3; i++) {
       LoopBounds.push_back(0);
       InductionVar.push_back(0);
    }
    // create linear loop index as 4th element
    InductionVar.push_back(0);
}

// Emit a branch and block. used as markers for code extraction
llvm::BasicBlock *CodeGenFunction::EmitMarkerBlock(const std::string name) {
  llvm::BasicBlock *entry = createBasicBlock(name);
  Builder.CreateBr(entry);
  EmitBlock(entry);
  return entry;
}

// Extract blocks to function and replace w/ call to function
void CodeGenFunction:: ExtractRegion(llvm::BasicBlock *entry, llvm::BasicBlock *exit, const std::string name) {
  std::vector< llvm::BasicBlock * > Blocks;

  llvm::Function::iterator BB = CurFn->begin();
  // find start marker
  for( ; BB->getName() != entry->getName(); ++BB) { }

  // collect forall basic blocks up to exit
  for( ; BB->getName() != exit->getName(); ++BB) {
    Blocks.push_back(BB);
  }

  //SC_TODO: should we be using a DominatorTree?
  //llvm::DominatorTree DT;

  llvm::CodeExtractor codeExtractor(Blocks, 0/*&DT*/, false);

  llvm::Function *ForallFn = codeExtractor.extractCodeRegion();

  ForallFn->setName(name);
}


void CodeGenFunction::EmitForallArrayStmt(const ForallArrayStmt &S) {

  //need a marker for start of Forall for CodeExtraction
  llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");

  EmitForallArrayLoop(S, S.getDims());

  //need a marker for end of Forall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("forall.exit");

  // Extract Blocks to function and replace w/ call to function
  ExtractRegion(entry, exit, "ForallArrayFunction");
}

void CodeGenFunction::EmitForallArrayLoop(const ForallArrayStmt &S, unsigned r) {
  RegionCounter Cnt = getPGORegionCounter(&S);
  (void)Cnt; //suppress warning
  
  CGDebugInfo *DI = getDebugInfo();

  llvm::Value *Start = EmitScalarExpr(S.getStart(r-1));
  llvm::Value *End = EmitScalarExpr(S.getEnd(r-1));
  llvm::Value *Stride = EmitScalarExpr(S.getStride(r-1));

  //initialize induction var
  const VarDecl *VD = S.getInductionVarDecl(r-1);
  EmitAutoVarDecl(*VD); //add induction var to LocalDeclmap.
  llvm::Value* InductVar = LocalDeclMap[VD];
  Builder.CreateStore(Start, InductVar);

  sprintf(IRNameStr, "forall.%s.end", DimNames[r-1]);
  JumpDest LoopExit = getJumpDestInCurrentScope(IRNameStr);
  RunCleanupsScope ForallScope(*this);

  if (DI)
    DI->EmitLexicalBlockStart(Builder, S.getSourceRange().getBegin());

  // Next we create a block that tests the induction variables value to
  // the rank's dimension.
  sprintf(IRNameStr, "forall.cond.%s", DimNames[r-1]);
  JumpDest Continue = getJumpDestInCurrentScope(IRNameStr);
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  RunCleanupsScope ConditionScope(*this);

  llvm::LoadInst *IVar = Builder.CreateLoad(InductVar, VD->getName().str().c_str());

  sprintf(IRNameStr, "forall.done.%s", IndexNames[r-1]);
  llvm::Value *CondValue = Builder.CreateICmpSLT(IVar,
      End,
      IRNameStr);

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();

  // If there are any cleanups between here and the loop-exit
  if (ForallScope.requiresCleanups()) {
    sprintf(IRNameStr, "forall.cond.cleanup.%s", DimNames[r-1]);
    ExitBlock = createBasicBlock(IRNameStr);
  }

  llvm::BasicBlock *LoopBody = createBasicBlock(IRNameStr);
  Builder.CreateCondBr(CondValue, LoopBody, ExitBlock);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);

  sprintf(IRNameStr, "forall.incblk.%s", IndexNames[r-1]);
  Continue = getJumpDestInCurrentScope(IRNameStr);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  if (r == 1) {  // This is our innermost rank, generate the loop body.
    EmitStmt(S.getBody());
  } else { // generate nested loop
    EmitForallArrayLoop(S, r-1);
  }

  EmitBlock(Continue.getBlock());

  llvm::LoadInst* iv = Builder.CreateLoad(InductVar, VD->getName().str().c_str());

  sprintf(IRNameStr, "%s.inc", VD->getName().str().c_str());
  llvm::Value *IncInductionVar = Builder.CreateAdd(iv,
      Stride,
      IRNameStr);

  Builder.CreateStore(IncInductionVar, InductVar);

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();

  EmitBranch(CondBlock);
  ForallScope.ForceCleanup();

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getSourceRange().getEnd());

  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitRenderallStmt(const RenderallMeshStmt &S) {

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value *ConstantOne = llvm::ConstantInt::get(Int32Ty, 1);

  const VarDecl* VD = S.getMeshVarDecl();
  if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(VD)){
    VD = IP->getMeshVarDecl();
  }

  llvm::Value *MeshBaseAddr = LocalDeclMap[VD];
  llvm::StringRef MeshName = S.getMeshType()->getName();

  // find number of fields
  MeshDecl* MD =  S.getMeshType()->getDecl();
  unsigned int nfields = MD->fields();

  unsigned int rank = S.getMeshType()->rankOf();
  //used by builtins to keep track of which dims are used
  for(unsigned int i = 0; i < rank; i++) {
    DimExists[i] = ConstantOne;
  }
  ResetVars();

  llvm::SmallVector< llvm::Value *, 3 > Args;
  Args.clear();

  //need a marker for start of Renderall for CodeExtraction
  llvm::BasicBlock *entry = EmitMarkerBlock("renderall.entry");

  // Create the induction variables for eack rank.
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "renderall.induct.%s.ptr", IndexNames[i]);
    InductionVar[i] = Builder.CreateAlloca(Int32Ty, 0, IRNameStr);
    //zero-initialize induction var
    Builder.CreateStore(ConstantZero, InductionVar[i]);
  }

  //build argument list for renderall setup runtime function
  for(unsigned int i = 0; i < 3; i++) {
     Args.push_back(0);
     sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
     LoopBounds[i] = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+i, IRNameStr);
     Args[i] = Builder.CreateLoad(LoopBounds[i], IRNameStr);
  }

  // create linear loop index as 4th element and zero-initialize
  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "renderall.linearidx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[3]);

        // call renderall setup runtime function
        llvm::Function *BeginFunc = CGM.getScoutRuntime().RenderallUniformBeginFunction();
        Builder.CreateCall(BeginFunc, ArrayRef<llvm::Value *>(Args));

  // call renderall color buffer setup
        llvm::Value *RuntimeColorPtr = CGM.getScoutRuntime().RenderallUniformColorsGlobal(*this);
        Color = Builder.CreateLoad(RuntimeColorPtr, "color");

  // renderall loops + body
  EmitRenderallMeshLoop(S, 3);

  // call renderall cleanup runtime function
  llvm::Function *EndFunc = CGM.getScoutRuntime().RenderallEndFunction();
  std::vector<llvm::Value*> EmptyArgs;
  Builder.CreateCall(EndFunc, ArrayRef<llvm::Value *>(EmptyArgs));

  // reset Loopbounds and induction var
  // so width/height etc can't be called after renderall
  ResetVars();

  //need a marker for end of Renderall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("renderall.exit");

  ExtractRegion(entry, exit, "RenderallFunction");
}

//generate one of the nested loops
void CodeGenFunction::EmitRenderallMeshLoop(const RenderallMeshStmt &S, unsigned r) {
  RegionCounter Cnt = getPGORegionCounter(&S);
  (void)Cnt; //suppress warning

  llvm::StringRef MeshName = S.getMeshType()->getName();

  CGDebugInfo *DI = getDebugInfo();

  llvm::Value *ConstantZero  = 0;
  llvm::Value *ConstantOne   = 0;
  ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
  ConstantOne  = llvm::ConstantInt::get(Int32Ty, 1);

  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[r-1]);

  sprintf(IRNameStr, "renderall.%s.end", DimNames[r-1]);
  JumpDest LoopExit = getJumpDestInCurrentScope(IRNameStr);
  RunCleanupsScope RenderallScope(*this);

  if (DI)
    DI->EmitLexicalBlockStart(Builder, S.getSourceRange().getBegin());

  // Extract the loop bounds from the mesh for this rank, this requires
  // a GEP from the mesh and a load from returned address...
  // note: width/height depth are stored after mesh fields
  // GEP is done in EmitRenderallStmt so just load here.
  sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), DimNames[r-1]);
  llvm::LoadInst *LoopBound  = Builder.CreateLoad(LoopBounds[r-1], IRNameStr);


  // Next we create a block that tests the induction variables value to
  // the rank's dimension.
  sprintf(IRNameStr, "renderall.cond.%s", DimNames[r-1]);
  JumpDest Continue = getJumpDestInCurrentScope(IRNameStr);
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  RunCleanupsScope ConditionScope(*this);

  sprintf(IRNameStr, "renderall.induct.%s", IndexNames[r-1]);
  llvm::LoadInst *IVar = Builder.CreateLoad(InductionVar[r-1], IRNameStr);

  sprintf(IRNameStr, "renderall.done.%s", IndexNames[r-1]);
  llvm::Value *CondValue = Builder.CreateICmpSLT(IVar,
                                                  LoopBound,
                                                  IRNameStr);

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();

  // If there are any cleanups between here and the loop-exit
  // scope, create a block to stage a loop exit along.  (We're
  // following Clang's lead here in generating a for loop.)
  if (RenderallScope.requiresCleanups()) {
    sprintf(IRNameStr, "renderall.cond.cleanup.%s", DimNames[r-1]);
    ExitBlock = createBasicBlock(IRNameStr);
  }

  llvm::BasicBlock *LoopBody = createBasicBlock(IRNameStr);
  Builder.CreateCondBr(CondValue, LoopBody, ExitBlock);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);

  sprintf(IRNameStr, "renderall.incblk.%s", IndexNames[r-1]);
  Continue = getJumpDestInCurrentScope(IRNameStr);

  // Store the blocks to use for break and continue.

  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  if (r == 1) {  // This is our innermost rank, generate the loop body.
    EmitStmt(S.getBody());
    // Increment the loop index stored as last element of InductionVar
    llvm::LoadInst* liv = Builder.CreateLoad(InductionVar[3], "renderall.linearidx");
    llvm::Value *IncLoopIndexVar = Builder.CreateAdd(liv,
        ConstantOne,
        "renderall.linearidx.inc");

    Builder.CreateStore(IncLoopIndexVar, InductionVar[3]);
  } else { // generate nested loop
    EmitRenderallMeshLoop(S, r-1);
  }

  EmitBlock(Continue.getBlock());

  sprintf(IRNameStr, "renderall.induct.%s", IndexNames[r-1]);
  llvm::LoadInst* iv = Builder.CreateLoad(InductionVar[r-1], IRNameStr);

  sprintf(IRNameStr, "renderall.inc.%s", IndexNames[r-1]);
  llvm::Value *IncInductionVar = Builder.CreateAdd(iv,
      ConstantOne,
      IRNameStr);

  Builder.CreateStore(IncInductionVar, InductionVar[r-1]);

  BreakContinueStack.pop_back();
  ConditionScope.ForceCleanup();

  EmitBranch(CondBlock);
  RenderallScope.ForceCleanup();

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getSourceRange().getEnd());

  EmitBlock(LoopExit.getBlock(), true);
}

