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

#include <stdio.h>
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/AST/Decl.h"
#include "CGBlocks.h"
#include "clang/Analysis/Analyses/Dominators.h"

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



// ----- EmitforallStmt
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

  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);
  unsigned int rank = S.getMeshType()->dimensions().size();

  LoopBounds.clear();
  InductionVar.clear();

  // Create the induction variables for eack rank.
  for(unsigned int i = 0; i < 3; i++) {
    LoopBounds.push_back(0);
    InductionVar.push_back(0);
    sprintf(IRNameStr, "forall.induct.%s.ptr", IndexNames[i]);
    InductionVar[i] = Builder.CreateAlloca(Int32Ty, 0, IRNameStr);

  }
  // create linear loop index as 4th element and zero-initialize.
  InductionVar.push_back(0);
  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.linearidx.ptr");
  Builder.CreateStore(ConstantZero, InductionVar[3]);

  EmitForallMeshLoop(S, rank);
}

//generate one of the nested loops
void CodeGenFunction::EmitForallMeshLoop(const ForallMeshStmt &S, unsigned r) {

  llvm::Value *MeshBaseAddr = GetMeshBaseAddr(S);
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
    EmitForallBody(S);

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

// ----- EmitForallBody
//
void CodeGenFunction::EmitForallBody(const ForallStmt &S) {
  EmitStmt(S.getBody());
}


void CodeGenFunction::EmitForallArrayStmt(const ForallArrayStmt &S) {
  EmitForallArrayLoop(S, S.getDims());
}

void CodeGenFunction::EmitForallArrayLoop(const ForallArrayStmt &S, unsigned r) {
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
    EmitForallBody(S);
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

#if 0
void CodeGenFunction::EmitForAllStmtWrapper(const ForallMeshStmt &S) {

  MeshMembers.clear();
  ScoutMeshSizes.clear();

  llvm::StringRef MeshName = S.getMeshType()->getName();
  const MeshType *MT = S.getMeshType();
  MeshType::MeshDimensions dims = MT->dimensions();
  MeshDecl *MD = MT->getDecl();

  unsigned int rank = dims.size();  // just for clarity below...

  typedef std::map<std::string, bool> MeshFieldMap;
  MeshFieldMap meshFieldMap;

  MeshBaseAddr = GetMeshBaseAddr(S);

  for(unsigned i = 0; i < rank; ++i) {
    sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), DimNames[i]);
    llvm::Value* lval = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, i+1, IRNameStr);
    ScoutMeshSizes.push_back(lval);
  }

  typedef MeshDecl::field_iterator MeshFieldIterator;
  MeshFieldIterator it = MD->field_begin(), it_end = MD->field_end();

  for(unsigned i = 0; it != it_end; ++it, ++i) {

    MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(*it);
    llvm::StringRef FieldName = MFD->getName();
    QualType Ty = MFD->getType();

    meshFieldMap[FieldName.str()] = true;

    if (! MFD->isImplicit()) {
      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), FieldName.str().c_str());
      llvm::Value *FieldPtr = Builder.CreateStructGEP(MeshBaseAddr, i+rank+1, IRNameStr);

      FieldPtr = Builder.CreateLoad(FieldPtr);
      sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), FieldName.str().c_str());
      llvm::Value *FieldVar = Builder.CreateAlloca(FieldPtr->getType(), 0, IRNameStr);

      Builder.CreateStore(FieldPtr, FieldVar);
      MeshMembers[FieldName] = std::make_pair(Builder.CreateLoad(FieldVar) , Ty);
      MeshMembers[FieldName].first->setName(FieldVar->getName());
    }
  }

  // Acquire a local copy of colors buffer.
  // SC_TODO -- this should go in a separate call stack (for dealing with
  // rendall specifically).
  /*
  if (isa< RenderAllStmt >(S)) {
    llvm::Type *fltTy = llvm::Type::getFloatTy(getLLVMContext());
    llvm::Type *flt4Ty = llvm::VectorType::get(fltTy, 4);
    llvm::Type *flt4PtrTy = llvm::PointerType::get(flt4Ty, 0);

    if (!CGM.getModule().getNamedGlobal("__scrt_renderall_uniform_colors")) {

      new llvm::GlobalVariable(CGM.getModule(),
                               flt4PtrTy,
                               false,
                               llvm::GlobalValue::ExternalLinkage,
                               0,
                               "__scrt_renderall_uniform_colors");
    }

    llvm::Value *local_colors  = Builder.CreateAlloca(flt4PtrTy, 0, "colors");
    llvm::Value *global_colors =
    CGM.getModule().getNamedGlobal("__scrt_renderall_uniform_colors");

    Builder.CreateStore(Builder.CreateLoad(global_colors), local_colors);
    Colors = Builder.CreateLoad(local_colors, "colors");
  }
  */

  llvm::BasicBlock *entry = createBasicBlock("forall_entry");
  Builder.CreateBr(entry);
  EmitBlock(entry);

  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  llvm::Instruction *ForallAllocaInsertPt =
    new llvm::BitCastInst(Undef, Int32Ty, "", Builder.GetInsertBlock());
  ForallAllocaInsertPt->setName("forall.allocapt");

  // Save the AllocaInsertPt.
  llvm::Instruction *savedAllocaInsertPt = AllocaInsertPt;
  AllocaInsertPt = ForallAllocaInsertPt;

  DeclMapTy curLocalDeclMap = LocalDeclMap; // Save LocalDeclMap.

  CallsPrintf = callsPrintf(&cast< Stmt >(S));

  // Generate body of function.
  EmitForallMeshStmt(S);

  LocalDeclMap = curLocalDeclMap; // Restore LocalDeclMap.

  // Restore the AllocaInsertPtr.
  AllocaInsertPt = savedAllocaInsertPt;

  llvm::Value *zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::ReturnInst *ret = llvm::ReturnInst::Create(getLLVMContext(), zero,
                                                   Builder.GetInsertBlock());

  std::vector< llvm::BasicBlock * > region;

  typedef llvm::Function::iterator BBIterator;
  BBIterator BB = CurFn->begin(), BB_end = CurFn->end();

  llvm::BasicBlock *split;
  for( ; BB->getName() != entry->getName(); ++BB) {
    split = BB;
  }

  typedef llvm::BasicBlock::iterator InstIterator;

  for( ; BB != BB_end; ++BB) {
    region.push_back(BB);
  }

  llvm::DominatorTree DT;
  DT.runOnFunction(*CurFn);

  llvm::Function *ForallFn;

  llvm::CodeExtractor codeExtractor(region, &DT, false);

  typedef llvm::SetVector<llvm::Value *> ValueSet;
  ValueSet ce_inputs, ce_outputs;
  codeExtractor.findInputsOutputs(ce_inputs, ce_outputs);
  ValueSet::iterator vsit, vsend;

  ForallFn = codeExtractor.extractCodeRegion();
  assert(ForallFn != 0 && "Failed to rip forall statement into a new function.");

  // SC_TODO: WARNING -- these function names are once again used as a special
  // case within the DoallToPTX transformation pass (in the LLVM source).  If
  // you change the name here you will need to also make the changes to the
  // pass...
  //
  // SC_TODO - we should move renderall logic into its own call path...
  //if (isa<RenderAllStmt>(S))
  //  ForallFn->setName("uniRenderallCellsFn");
  //else
  ForallFn->setName("uniForallCellsFn");


  // SC_TODO - this is hard to follow in the middle of all the other details.  We
  // should move GPU lowering details into its own (member) function.
  if (isGPU()) {

    std::string name = ForallFn->getName().str();
    assert(name.find(".") == std::string::npos && "Illegal PTX identifier (function name).\n");

    // Add metadata for scout kernel function.
    llvm::NamedMDNode *ScoutMetadata;
    ScoutMetadata = CGM.getModule().getOrInsertNamedMetadata("scout.kernels");

    SmallVector<llvm::Value *, 4> KMD; // Kernel (as in GPU kernel) metadata.
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ForallFn));
    // For each function argument, a bit to indicate whether it is a mesh member.
    SmallVector<llvm::Value*, 3> args;
    SmallVector<llvm::Value*, 3> signedArgs;
    SmallVector<llvm::Value*, 3> meshArgs;
    SmallVector<llvm::Value*, 3> typeArgs;
    typedef llvm::Function::arg_iterator ArgIterator;
    size_t pos = 0;
    llvm::Value* gs;

    for(ArgIterator it = ForallFn->arg_begin(), end = ForallFn->arg_end();
        it != end; ++it, ++pos) {

      bool isSigned = false;
      std::string typeStr;
      // All of our values from the mesh are prefixed with the
      // mesh name (we do this as we lower).
      if (it->getName().startswith(MeshName) && isMeshMember(it, isSigned, typeStr)) {

        // SC_TODO - need to fix this...  At present, I'm not sure why we
        // even need it...  It should reflect the current argument
        // is a signed value or not???
        isSigned = false;

        args.push_back(llvm::ConstantInt::get(Int32Ty, 1));

        if (isSigned) {
          signedArgs.push_back(llvm::ConstantInt::get(Int32Ty, 1));
        } else {
          signedArgs.push_back(llvm::ConstantInt::get(Int32Ty, 0));
        }

        // Convert mesh field arguments to the function which have
        // been uniqued by ExtractCodeRegion() back into mesh field
        // names.
        // SC_TODO - this code was, and still is, fundamentally flawed...
        // We can't simply strip numbers off the end of the name as the
        // programmer could have specified
        std::string ns = (*it).getName().str();
        while(!ns.empty()) {
          if (meshFieldMap.find(ns) != meshFieldMap.end()) {
	    gs = llvm::ConstantDataArray::getString(getLLVMContext(), ns);
            meshArgs.push_back(gs);
            break;
          }
          ns.erase(ns.length() - 1, 1);
        }

        assert(!ns.empty() && "failed to convert uniqued mesh field name");

        gs = llvm::ConstantDataArray::getString(getLLVMContext(), typeStr);
        typeArgs.push_back(gs);
      } else {
        args.push_back(llvm::ConstantInt::get(Int32Ty, 0));
        signedArgs.push_back(llvm::ConstantInt::get(Int32Ty, 0));
        gs = llvm::ConstantDataArray::getString(getLLVMContext(), (*it).getName());
        meshArgs.push_back(gs);

        // SC_TODO: these are now named MeshName.[width|height|depth]
        // (see code above).  We probably should find something better
        // here than string comparisons...
        std::string FieldWidthStr(MeshName.str() + std::string(".width"));
        std::string FieldHeightStr(MeshName.str() + std::string(".height"));
        std::string FieldDepthStr(MeshName.str() + std::string(".depth"));
        if (it->getName().startswith(FieldWidthStr)   ||
            it->getName().startswith(FieldHeightStr)  ||
            it->getName().startswith(FieldDepthStr)) {
          gs = llvm::ConstantDataArray::getString(getLLVMContext(), "uint*");
          typeArgs.push_back(gs);
        } else {
          bool found = false;
          for(llvm::DenseMap<const Decl*, llvm::Value*>::iterator
              itr = LocalDeclMap.begin(), itrEnd = LocalDeclMap.end();
              itr != itrEnd; ++itr) {

            if (const ValueDecl* vd = dyn_cast<ValueDecl>(itr->first)) {

              if (vd->getName() == it->getName()) {
                std::string ts = vd->getType().getAsString();
                size_t pos = ts.find(" [");
                if (pos != std::string::npos) {
                  ts = ts.substr(0, pos);
                }

                if (ts.find("*") == std::string::npos) {
                  ts += "*";
                }

                gs =  llvm::ConstantDataArray::getString(getLLVMContext(), ts);
                found = true;
                break;
              }
            }
          }

          if (found) {
            typeArgs.push_back(gs);
          }
        }
      }
    }
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value * >(args)));

    args.clear();

    // Add dimension information.
    llvm::Value *zero = llvm::ConstantInt::get(Int32Ty, 0);
    MeshType::MeshDimensions dims = MT->dimensions();
    for(unsigned i = 0; i < rank; ++i) {
      args.push_back(zero);
      args.push_back(TranslateExprToValue(dims[i]));
    }
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value * >(args)));

    args.clear();
    args.push_back(llvm::ConstantDataArray::getString(getLLVMContext(), MeshName));;

    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(args)));
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(meshArgs)));
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(signedArgs)));
    KMD.push_back(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(typeArgs)));

    ScoutMetadata->addOperand(llvm::MDNode::get(getLLVMContext(), ArrayRef<llvm::Value*>(KMD)));
  }

  if (isSequential() || isGPU()) {
    llvm::BasicBlock *cbb = ret->getParent();
    ret->eraseFromParent();
    Builder.SetInsertPoint(cbb);
    return;
  }

  // Remove function call to ForallFn.
  llvm::BasicBlock *CallBB = split->getTerminator()->getSuccessor(0);
  typedef llvm::BasicBlock::iterator InstIterator;
  InstIterator I = CallBB->begin(), IE = CallBB->end();
  for( ; I != IE; ++I) {
    if (llvm::CallInst *call = dyn_cast< llvm::CallInst >(I)) {
      call->eraseFromParent();
      break;
    }
  }

  llvm::BasicBlock *continueBB = ret->getParent();
  ret->eraseFromParent();

  Builder.SetInsertPoint(continueBB);

  typedef llvm::SetVector< llvm::Value * > Values;
  Values inputs;

  std::string TheName = CurFn->getName();
#ifdef USE_FORALL_BLOCK
  CGBlockInfo blockInfo(S.getBlock()->getBlockDecl(), TheName.c_str());

  llvm::Value *BlockFn = EmitScoutBlockLiteral(S.getBlock(),
                                               blockInfo,
                                               ScoutMeshSizes,
                                               inputs);

  // Generate a function call to BlockFn.
  EmitScoutBlockFnCall(BlockFn, blockInfo,
                       ScoutMeshSizes, inputs);
#endif
}

void CodeGenFunction::EmitForAllArrayStmt(const ForAllArrayStmt &S) {

  llvm::SmallVector< llvm::Value *, 3 > ranges;
  for(size_t i = 0; i < 3; ++i) {
    Expr* end = S.getEnd(i);

    if(!end){
      break;
    }

    llvm::Value* ri = Builder.CreateAlloca(Int32Ty);
    ranges.push_back(ri);
    Builder.CreateStore(TranslateExprToValue(end), ri);
  }

  llvm::BasicBlock* entry = createBasicBlock("faa.entry");
  EmitBlock(entry);

  llvm::BasicBlock* End[4] = {0,0,0,0};

  End[0] = createBasicBlock("faa.end");

  llvm::BasicBlock* body = createBasicBlock("faa.body");

  llvm::Value* Undef = llvm::UndefValue::get(Int32Ty);
  llvm::Instruction* ForallArrayAllocaInsertPt =
  new llvm::BitCastInst(Undef, Int32Ty, "", Builder.GetInsertBlock());
  ForallArrayAllocaInsertPt->setName("faa.allocapt");

  llvm::Instruction *savedAllocaInsertPt = AllocaInsertPt;
  AllocaInsertPt = ForallArrayAllocaInsertPt;

  ScoutIdxVars.clear();

  for(unsigned i = 0; i < 3; ++i){
    const IdentifierInfo* ii = S.getInductionVar(i);
    if(!ii){
      break;
    }

    llvm::Value* ivar = Builder.CreateAlloca(Int32Ty, 0, ii->getName());
    Builder.CreateStore(TranslateExprToValue(S.getStart(i)), ivar);
    ScoutIdxVars.push_back(ivar);
  }

  llvm::BasicBlock::iterator entryPt = Builder.GetInsertPoint();

  for(unsigned i = 0; i < ScoutIdxVars.size(); ++i){
    End[i+1] = createBasicBlock("faa.loopend");
    CurFn->getBasicBlockList().push_back(End[i+1]);
    Builder.SetInsertPoint(End[i+1]);

    if(i < ScoutIdxVars.size() - 1){
      Builder.CreateStore(TranslateExprToValue(S.getStart(i + 1)), ScoutIdxVars[i + 1]);
    }

    llvm::Value* ivar = ScoutIdxVars[i];

    llvm::Value* iv = Builder.CreateLoad(ivar);
    Builder.CreateStore(Builder.CreateAdd(iv, TranslateExprToValue(S.getStride(i))), ivar);
    llvm::Value* cond =
    Builder.CreateICmpSLT(Builder.CreateLoad(ivar), TranslateExprToValue(S.getEnd(i)), "faa.cmp");

    Builder.CreateCondBr(cond, body, End[i]);
  }

  Builder.SetInsertPoint(entry, entryPt);

  EmitBlock(body);

  DeclMapTy curLocalDeclMap = LocalDeclMap;
  CallsPrintf = callsPrintf(&cast< Stmt >(S));

  CurrentForAllArrayStmt = &S;
  EmitStmt(S.getBody());
  CurrentForAllArrayStmt = 0;

  Builder.CreateBr(End[ScoutIdxVars.size()]);

  EmitBlock(End[0]);

  llvm::Value *zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::ReturnInst *ret = llvm::ReturnInst::Create(getLLVMContext(), zero,
                                                   Builder.GetInsertBlock());

  LocalDeclMap = curLocalDeclMap;
  AllocaInsertPt = savedAllocaInsertPt;

  std::vector< llvm::BasicBlock * > region;

  typedef llvm::Function::iterator BBIterator;
  BBIterator BB = CurFn->begin(), BB_end = CurFn->end();

  llvm::BasicBlock *split;
  for( ; BB->getName() != entry->getName(); ++BB)
    split = BB;

  for( ; BB != BB_end; ++BB) {
    region.push_back(BB);
  }

  llvm::DominatorTree DT;
  DT.runOnFunction(*CurFn);

  llvm::CodeExtractor codeExtractor(region, &DT, false);

  llvm::Function *ForallArrayFn = codeExtractor.extractCodeRegion();

  ForallArrayFn->setName("forall_array");

  if (isSequential() || isGPU()) {
    llvm::BasicBlock *cbb = ret->getParent();
    ret->eraseFromParent();

    Builder.SetInsertPoint(cbb);
    return;
  }

  // Remove function call to ForallFn.
  llvm::BasicBlock *CallBB = split->getTerminator()->getSuccessor(0);
  typedef llvm::BasicBlock::iterator InstIterator;
  InstIterator I = CallBB->begin(), IE = CallBB->end();
  for( ; I != IE; ++I) {
    if (llvm::CallInst *call = dyn_cast< llvm::CallInst >(I)) {
      call->eraseFromParent();
      break;
    }
  }

  llvm::BasicBlock *continueBB = ret->getParent();
  ret->eraseFromParent();

  Builder.SetInsertPoint(continueBB);

  typedef llvm::SetVector< llvm::Value * > Values;
  Values inputs;

  std::string TheName = CurFn->getName();
#ifdef USE_FORALL_BLOCK
  CGBlockInfo blockInfo(S.getBlock()->getBlockDecl(), TheName.c_str());

  CurrentForAllArrayStmt = &S;

  llvm::Value *BlockFn = EmitScoutBlockLiteral(S.getBlock(), blockInfo,
                                               ranges, inputs);

  CurrentForAllArrayStmt = 0;

  // Generate a function call to BlockFn.
  EmitScoutBlockFnCall(BlockFn, blockInfo, ranges, inputs);
#endif
}

bool CodeGenFunction::hasCalledFn(llvm::Function *Fn, llvm::StringRef name) {
  typedef llvm::Function::iterator BBIterator;
  BBIterator BB = Fn->begin(), BB_end = Fn->end();
  typedef llvm::BasicBlock::iterator InstIterator;
  for( ; BB != BB_end; ++BB) {
    InstIterator Inst = BB->begin(), Inst_end = BB->end();
    for( ; Inst != Inst_end; ++Inst) {
      if (isCalledFn(Inst, name)) return true;
    }
  }
  return false;
}

bool CodeGenFunction::isCalledFn(llvm::Instruction *Instn, llvm::StringRef name) {
  if (isa< llvm::CallInst >(Instn)) {
    llvm::CallInst *call = cast< llvm::CallInst >(Instn);
    llvm::Function *Fn = call->getCalledFunction();
    return Fn->getName() == name || hasCalledFn(Fn, name);
  }
  return false;
}

llvm::Value *CodeGenFunction::TranslateExprToValue(const Expr *E) {
  switch(E->getStmtClass()) {
    case Expr::IntegerLiteralClass:
    case Expr::BinaryOperatorClass:
      return EmitScalarExpr(E);
    default:
      return Builder.CreateLoad(EmitLValue(E).getAddress());
 }
}

void CodeGenFunction::EmitForAllStmt(const ForAllStmt &S) {
  DEBUG_OUT("EmitForAllStmt");

  // Forall will initially behave exactly like a for loop.
  RunCleanupsScope Forallscope(*this);

  llvm::StringRef name = "indvar";
  llvm::Value *zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value *one = llvm::ConstantInt::get(Int32Ty, 1);

  // Use the mesh's name to identify which mesh variable to use whem implicitly defined.
  const IdentifierInfo *MeshII = S.getMesh();
  llvm::StringRef meshName = MeshII->getName();

  // Get the number and size of the mesh's dimensions.
  const MeshType *MT = S.getMeshType();
  MeshType::MeshDimensions dims = MT->dimensions();

  typedef std::vector< unsigned > Vector;
  typedef Vector::iterator VecIterator;
  typedef Vector::reverse_iterator VecRevIterator;

  ForallTripCount = one;
  std::vector< llvm::Value * > start, end, diff;

  unsigned int rank = 0;
  for(unsigned i = 0; i < dims.size(); ++i) {
    if (dims[i] != 0)
      rank++;
  }

  for(unsigned i = 0; i < rank; ++i) {
    //start.push_back(TranslateExprToValue(S.getStart(i)));
    //end.push_back(TranslateExprToValue(S.getEnd(i)));

    //diff.push_back(Builder.CreateSub(end.back(), start.back()));
    //ForallTripCount = Builder.CreateMul(ForallTripCount, diff.back());

    llvm::Value* msi;
    sprintf(IRNameStr, "%s.%s", meshName.str().c_str(), DimNames[i]);

    switch(i) {

      case 0:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], IRNameStr);
        break;

      case 1:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], IRNameStr);
        break;

      case 2:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], IRNameStr);
        break;

      default:
        assert(false && "Dimension case not handled in EmitForAllStmt");
    }

    // SC_TODO -- we always start at zero for each rank...
    // Why build a list to store all zeros?  It is also
    // not clear why we need to do this with the width, height,
    // and depth -- we already have them...  ????
    start.push_back(zero);
    end.push_back(msi);
    diff.push_back(msi);
    ForallTripCount = Builder.CreateMul(ForallTripCount, msi);
  }

  llvm::Value *indVar = Builder.CreateAlloca(Int32Ty, 0, name);
  // SC_TODO - This is likely confusing -- both sequential and
  // cpu appear to imply the same thing...
  if(isSequential() || isMultiCPU())
    Builder.CreateStore(zero, indVar);

  ForallIndVar = indVar;
  ScoutIdxVars.clear(); // Clear the list of stale ScoutIdxVars.

  // Initialize the index variables.
  for(unsigned i = 0; i < rank; ++i) {
    llvm::Value *lval = 0;

    switch(i) {

      case 0:
        lval = Builder.CreateAlloca(Int32Ty, 0, "indvar.x");
        break;

      case 1:
        lval = Builder.CreateAlloca(Int32Ty, 0, "indvar.y");
        break;

      case 2:
        lval = Builder.CreateAlloca(Int32Ty, 0, "indvar.z");
        break;

      default:
        assert(false && "Case not handled for ForAll indvar");
    }

    Builder.CreateStore(start[i], lval); // SC_TODO - per above, this is always zero...
    ScoutIdxVars.push_back(lval);
  }

  llvm::Value *lval = 0;
  llvm::Value *cond = 0;
  llvm::BasicBlock *CondBlock;

  // SC_TODO - Again, this could be confusing -- both sequential and
  // cpu appear to be imply the same thing...
  if (isSequential() || isMultiCPU()) {

    // Start the loop with a block that tests the condition.
    JumpDest Continue = getJumpDestInCurrentScope("forall.cond");
    CondBlock = Continue.getBlock();
    EmitBlock(CondBlock);

    // Generate loop condition.
    lval = getGlobalIdx();
    cond = Builder.CreateICmpSLT(lval, ForallTripCount, "cmptmp");
  }

  llvm::BasicBlock *ForallBody = createBasicBlock("forall.body");

  llvm::BasicBlock *ExitBlock;
  // SC_TODO -- this is probably not as clear as it should be...
  // Both sequential and CPU appear to imply the same thing...
  if (isSequential() || isMultiCPU()) {
    ExitBlock = createBasicBlock("forall.end");
    Builder.SetInsertPoint(CondBlock);
    Builder.CreateCondBr(cond, ForallBody, ExitBlock);
  }

  // As long as the condition is true, iterate the loop.
  EmitBlock(ForallBody);
  Builder.SetInsertPoint(ForallBody);

  // Set each dimension's index variable from induction variable.
  for(unsigned i = 0; i < rank; ++i) {
    lval = getGlobalIdx();
    llvm::Value *val;
    if (i > 0) {
      if(i == 1)
        val = diff[i - 1];
      else
        val = Builder.CreateMul(diff[i-1], diff[i - 2]);
      lval = Builder.CreateUDiv(lval, val);
    }

    lval = Builder.CreateURem(lval, diff[i]);
    if (start[i] != zero)
      lval = Builder.CreateAdd(lval, start[i]);
    Builder.CreateStore(lval, ScoutIdxVars[i]);
  }

  // Generate the statements in the body of the forall.
  EmitStmt(S.getBody());

  if(isSequential() || isMultiCPU()) {
    // Increment the induction variables.
    lval = getGlobalIdx();
    Builder.CreateStore(Builder.CreateAdd(lval, one), ForallIndVar);
    Builder.CreateBr(CondBlock);

    EmitBlock(ExitBlock);
  }
}

void CodeGenFunction::EmitRenderAllStmt(const RenderAllStmt &S) {

  recomment this out...
  //---- *
  DEBUG_OUT("EmitRenderAllStmt");

  llvm::Type *fltTy = llvm::Type::getFloatTy(getLLVMContext());
  llvm::Type *Ty = llvm::PointerType::get(llvm::VectorType::get(fltTy, 4), 0);

  const MeshType *MT = S.getMeshType();
  MeshDecl::MeshDimensionVec dims = MT->dimensions();

  unsigned dim = 1;
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    dim *= dims[i]->EvaluateAsInt(getContext()).getSExtValue();
  }

  llvm::AttrListPtr namPAL;
  llvm::SmallVector< llvm::AttributeWithIndex, 4 > Attrs;
  llvm::AttributeWithIndex PAWI;
  PAWI.Index = 0u; PAWI.Attrs = 0 | llvm::Attribute::NoAlias;
  Attrs.push_back(PAWI);
  namPAL = llvm::AttrListPtr::get(Attrs.begin(), Attrs.end());

  if(!CGM.getModule().getFunction(SC_MANGLED_NEW)) {
    llvm::FunctionType *FTy = llvm::FunctionType::get(Int8PtrTy, Int64Ty, isVarArg=false);
    llvm::Function *namF = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                                  SC_MANGLED_NEW, &CGM.getModule());
    namF->setAttributes(namPAL);
  }

  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  Builder.SetInsertPoint(&*AllocaInsertPt);

  llvm::Constant *nam = CGM.getModule().getFunction(SC_MANGLED_NEW);

  llvm::CallInst *call = Builder.CreateCall(nam, llvm::ConstantInt::get(Int64Ty, 16 * dim));
  call->setAttributes(namPAL);
  llvm::Value *val = Builder.CreateBitCast(call, Ty);
  llvm::Value *alloca = Builder.CreateAlloca(Ty, 0, "color");
  val = Builder.CreateStore(val, alloca);

  Builder.SetInsertPoint(BB);
  ScoutColor = alloca;
  // --- * /
  recomment this out...

  // scout - skip the above, at least for now, because we are writing to colors
  // which is a preallocated pixel buffer that exists at the time the
  // renderall loop is started - we write to an offset corresponding
  // to the induction variable - done in EmitForAllStmt()

  RenderAll = 1;
  EmitForAllStmtWrapper(cast<ForAllStmt>(S));
  RenderAll = 0;
}

void CodeGenFunction::EmitVolumeRenderAllStmt(const VolumeRenderAllStmt &S)
{
  DEBUG_OUT("EmitVolumeRenderallStmt");
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),S.getLBracLoc(),
                                "LLVM IR generation of volume renderall statement ('{}')");

  CGDebugInfo *DI = getDebugInfo();
  if (DI)
    DI->EmitLexicalBlockStart(Builder, S.getLBracLoc());

  // Keep track of the current cleanup stack depth.
  RunCleanupsScope Scope(*this);

  // Clear stale mesh elements.
  MeshMembers.clear();
  const IdentifierInfo *MeshII = S.getMesh();
  llvm::StringRef meshName = MeshII->getName();
  (void)meshName; //supress warning

  const MeshType *MT = S.getMeshType();
  MeshType::MeshDimensions dims = MT->dimensions();
  const MeshDecl *MD = MT->getDecl();

  typedef std::map<std::string, bool> MeshFieldMap;
  MeshFieldMap meshFieldMap;
  const VarDecl* MVD = S.getMeshVarDecl();

  llvm::Value* baseAddr = LocalDeclMap[MVD];

  if(MVD->getType().getTypePtr()->isReferenceType()){
    baseAddr = Builder.CreateLoad(baseAddr);
  }

  ScoutMeshSizes.clear();
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    llvm::Value *lval = Builder.CreateConstInBoundsGEP2_32(baseAddr, 0, i);
    ScoutMeshSizes.push_back(lval);
  }

  llvm::Function *addVolFunc = CGM.getModule().getFunction("__scrt_renderall_add_volume");

  if (!addVolFunc) {
    llvm::PointerType* p1 = llvm::PointerType::get(llvm::Type::getFloatTy(getLLVMContext()), 0);
    llvm::Type* p2 = llvm::Type::getInt32Ty(getLLVMContext());
    std::vector<llvm::Type*> args;
    args.push_back(p1);
    args.push_back(p2);


    llvm::FunctionType *FTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                              args, false);

    addVolFunc = llvm::Function::Create(FTy,
                                        llvm::Function::ExternalLinkage,
                                        "__scrt_renderall_add_volume",
                                        &CGM.getModule());
  }


  size_t fieldcount = 0;
  typedef MeshDecl::field_iterator MeshFieldIterator;
  MeshFieldIterator it = MD->field_begin(), it_end = MD->field_end();

  for(unsigned i = 0; it != it_end; ++it, ++i) {

    MeshFieldDecl *FD = dyn_cast<MeshFieldDecl>(*it);
    llvm::StringRef name = FD->getName();
    meshFieldMap[name.str()] = true;

    QualType Ty = FD->getType();

    if (! FD->isImplicit()) {

      llvm::Value *addr;
      addr = Builder.CreateStructGEP(baseAddr, i+4, name); //SC_TODO: why i+4??
      addr = Builder.CreateLoad(addr);

      llvm::Value *var = Builder.CreateAlloca(addr->getType(), 0, name);
      Builder.CreateStore(addr, var);
      MeshMembers[name] = std::make_pair(Builder.CreateLoad(var) , Ty);
      MeshMembers[name].first->setName(var->getName());

      // the Value* var holding the addr where the mesh member is
      llvm::Value* meshField = MeshMembers[name].first;  // SC_TODO -- isn't this 'var' from above???

      // the Value* for the volume number
      llvm::ConstantInt* volumeNum;
      volumeNum = llvm::ConstantInt::get(Int32Ty, fieldcount);

      // emit the call
      llvm::CallInst* CI =
      Builder.CreateCall2(addVolFunc, meshField, volumeNum);
      (void)CI; //suppress warning
      ++fieldcount;
    }
  }

  std::vector<llvm::Value*> Args;

  llvm::Function *beginRendFunc = CGM.getModule().getFunction("__scrt_renderall_begin");

  if(!beginRendFunc){

    std::vector<llvm::Type*> args;

    llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                            args, false);

    beginRendFunc = llvm::Function::Create(FTy,
                                        llvm::Function::ExternalLinkage,
                                        "__scrt_renderall_begin",
                                        &CGM.getModule());
  }
  Builder.CreateCall(beginRendFunc, Args);

  llvm::Function *endRendFunc = CGM.getModule().getFunction("__scrt_renderall_end");

  if(!endRendFunc){

    std::vector<llvm::Type*> args;

    llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                            args, false);

    endRendFunc = llvm::Function::Create(FTy,
                                           llvm::Function::ExternalLinkage,
                                           "__scrt_renderall_end",
                                           &CGM.getModule());
  }

  Builder.CreateCall(endRendFunc, Args);
  recomment this out / *
  llvm::Function *delRendFunc = CGM.getModule().getFunction("__scrt_renderall_delete");
  if(!delRendFunc) {
    std::vector<llvm::Type*> args;
    llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                            args, false);

    delRendFunc = llvm::Function::Create(FTy,
                                           llvm::Function::ExternalLinkage,
                                           "__scrt_renderall_delete",
                                           &CGM.getModule());
  }
  Builder.CreateCall(delRendFunc, Args);
  recomment this out * /

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getRBracLoc());
}
#endif
