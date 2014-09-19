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
#include <cassert>
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "CGBlocks.h"

#include "Scout/CGScoutRuntime.h"
#include "Scout/CGLegionRuntime.h"
#include "Scout/ASTVisitors.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"

#include "legion/lsci.h"

using namespace clang;
using namespace CodeGen;

static const char *DimNames[]   = { "width", "height", "depth" };
static const char *IndexNames[] = { "x", "y", "z"};

static const uint8_t FIELD_CELL = 0;
static const uint8_t FIELD_VERTEX = 1;
static const uint8_t FIELD_EDGE = 2;
static const uint8_t FIELD_FACE = 3;

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

//from VarDecl get base addr of mesh
void CodeGenFunction::GetMeshBaseAddr(const VarDecl *MeshVarDecl, llvm::Value*& BaseAddr) {
  // is a global. SC_TODO why not MeshVarDecl->hasGlobalStorage()?
  if ((MeshVarDecl->hasLinkage() || MeshVarDecl->isStaticDataMember())
      && MeshVarDecl->getTLSKind() != VarDecl::TLS_Dynamic) {

    BaseAddr = CGM.GetAddrOfGlobalVar(MeshVarDecl);

    // If BaseAddr is an external global then it is assumed that we are within LLDB
    // and we need to load the mesh base address because it is passed as a global
    // reference.
    bool shouldLoad = false;
    if(llvm::PointerType* PT = dyn_cast<llvm::PointerType>(BaseAddr->getType())){
      llvm::Type* ET = PT->getElementType();
      shouldLoad = ET->isPointerTy();
    }

    if(shouldLoad) {
      llvm::Value* V = LocalDeclMap.lookup(MeshVarDecl);
      if(V){
        BaseAddr = V;
        return;
      }

      BaseAddr = Builder.CreateLoad(BaseAddr);
      LocalDeclMap[MeshVarDecl] = BaseAddr;
    } else {
      EmitGlobalMeshAllocaIfMissing(BaseAddr, *MeshVarDecl);
    }

  } else {
    if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(MeshVarDecl)){
      BaseAddr = LocalDeclMap[IP->getMeshVarDecl()];
    } else {
      BaseAddr = LocalDeclMap[MeshVarDecl];
    }

    // If Mesh ptr then load
    const Type *T = MeshVarDecl->getType().getTypePtr();
    if(T->isAnyPointerType() || T->isReferenceType()) {
      BaseAddr = Builder.CreateLoad(BaseAddr);
    }
  }
}

//from Stmt get base addr of mesh
void CodeGenFunction::GetMeshBaseAddr(const Stmt &S, llvm::Value*& BaseAddr) {

  const VarDecl *MeshVarDecl;

  if(const ForallMeshStmt *FA = dyn_cast<ForallMeshStmt>(&S)) {
    MeshVarDecl = FA->getMeshVarDecl();
  } else if (const RenderallMeshStmt *RA = dyn_cast<RenderallMeshStmt>(&S)) {
    MeshVarDecl = RA->getMeshVarDecl();
  } else {
    assert(false && "expected ForallMeshStmt or RenderallMeshStmt");
  }

  GetMeshBaseAddr(MeshVarDecl, BaseAddr);
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

void CodeGenFunction::EmitForallCellsVertices(const ForallMeshStmt &S){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.vertices.entry");
  (void)EntryBlock; //suppress warning

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  llvm::Value* vertexPosPtr = InnerInductionVar;
  VertexIndex = InnerIndex;

  Builder.CreateStore(Zero, vertexPosPtr);

  llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");
  llvm::Value* height;
  llvm::Value* indVar =  Builder.CreateLoad(InductionVar[3], "indVar");

  llvm::Value* width1;
  llvm::Value* height1;
  llvm::Value* idvw;
  llvm::Value* wh1;

  if(rank > 1){
    width1 =  Builder.CreateAdd(width, One);
    idvw = Builder.CreateUDiv(indVar, width, "idvw");
  }

  if(rank > 2){
    height = Builder.CreateLoad(LoopBounds[1], "height");
    height1 =  Builder.CreateAdd(height, One);
    wh1 = Builder.CreateMul(width1, height1);
  }

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.vertices.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  llvm::Value* vertexPos = Builder.CreateLoad(vertexPosPtr, "vertex.pos");

  if(rank == 3){
    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
    llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");
    llvm::Value* k = Builder.CreateLoad(InductionVar[2], "k");
    llvm::Value* v1 = Builder.CreateMul(j, width1);
    llvm::Value* v2 = Builder.CreateMul(k, wh1);
    llvm::Value* xyz = Builder.CreateAdd(i, Builder.CreateAdd(v1, v2));

    llvm::Value* pd4 = Builder.CreateUDiv(vertexPos, Four);
    llvm::Value* pm4 = Builder.CreateURem(vertexPos, Four);
    llvm::Value* pd2 = Builder.CreateUDiv(pm4, Two);
    llvm::Value* pm2 = Builder.CreateURem(pm4, Two);

    llvm::Value* v3 = Builder.CreateMul(pd2, width1);
    llvm::Value* v4 = Builder.CreateMul(pd4, wh1);

    llvm::Value* newVertexIndex =
        Builder.CreateAdd(Builder.CreateAdd(xyz, v3),
                          Builder.CreateAdd(pm2, v4));

    Builder.CreateStore(newVertexIndex, VertexIndex);
  }
  else if(rank == 2){
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

  if(rank == 3){
    Cond = Builder.CreateICmpSLT(vertexPos, Seven, "cond");
  }
  else if(rank == 2){
    Cond = Builder.CreateICmpSLT(vertexPos, Three, "cond");
  }
  else{
    Cond = Builder.CreateICmpSLT(vertexPos, One, "cond");
  }

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.vertices.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);
}

void CodeGenFunction::EmitForallVerticesCells(const ForallMeshStmt &S){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.cells.entry");
  (void)EntryBlock; //suppress warning

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  llvm::Value* cellPosPtr = InnerInductionVar;
  CellIndex = InnerIndex;

  Builder.CreateStore(Zero, cellPosPtr);

  llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");

  llvm::Value* height;
  llvm::Value* depth;
  if(rank > 1){
    height = Builder.CreateLoad(LoopBounds[1], "height");
  }

  if(rank > 2){
    depth = Builder.CreateLoad(LoopBounds[2], "depth");
  }

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.cells.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  llvm::Value* cellPos = Builder.CreateLoad(cellPosPtr, "cell.pos");

  if(rank == 3){
    llvm::Value* pd4 = Builder.CreateUDiv(cellPos, Four);
    llvm::Value* pm4 = Builder.CreateURem(cellPos, Four);
    llvm::Value* pd2 = Builder.CreateUDiv(pm4, Two);
    llvm::Value* pm2 = Builder.CreateURem(pm4, Two);

    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
    llvm::Value* x = Builder.CreateSub(i, pd2, "x");

    llvm::Value* cx1 = Builder.CreateICmpSLT(x, Zero);
    llvm::Value* cx2 = Builder.CreateICmpSGE(x, width);
    llvm::Value* vx2 = Builder.CreateAdd(x, width);
    llvm::Value* vx3 = Builder.CreateURem(x, width);
    x = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));

    llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");
    llvm::Value* y = Builder.CreateSub(j, pm2, "y");

    llvm::Value* cy1 = Builder.CreateICmpSLT(y, Zero);
    llvm::Value* cy2 = Builder.CreateICmpSGE(y, height);
    llvm::Value* vy2 = Builder.CreateAdd(y, height);
    llvm::Value* vy3 = Builder.CreateURem(y, height);
    y = Builder.CreateSelect(cy1, vy2, Builder.CreateSelect(cy2, vy3, y));

    llvm::Value* k = Builder.CreateLoad(InductionVar[2], "k");
    llvm::Value* z = Builder.CreateSub(k, pd4, "z");

    llvm::Value* cz1 = Builder.CreateICmpSLT(z, Zero);
    llvm::Value* cz2 = Builder.CreateICmpSGE(z, depth);
    llvm::Value* vz2 = Builder.CreateAdd(z, depth);
    llvm::Value* vz3 = Builder.CreateURem(z, depth);
    z = Builder.CreateSelect(cz1, vz2, Builder.CreateSelect(cz2, vz3, z));

    llvm::Value* v1 = Builder.CreateMul(j, width);
    llvm::Value* v2 = Builder.CreateMul(k, Builder.CreateMul(width, height));

    llvm::Value* newCellIndex = Builder.CreateAdd(i, Builder.CreateAdd(v1, v2));
    Builder.CreateStore(newCellIndex, CellIndex);
  }
  else if(rank == 2){
    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");

    llvm::Value* vx1 = Builder.CreateUDiv(cellPos, Two);
    llvm::Value* x = Builder.CreateSub(i, vx1, "x");

    llvm::Value* cx1 = Builder.CreateICmpSLT(x, Zero);
    llvm::Value* cx2 = Builder.CreateICmpSGE(x, width);
    llvm::Value* vx2 = Builder.CreateAdd(x, width);
    llvm::Value* vx3 = Builder.CreateURem(x, width);
    x = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));

    llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");
    llvm::Value* vy1 = Builder.CreateURem(cellPos, Two);
    llvm::Value* y = Builder.CreateSub(j, vy1, "y");

    llvm::Value* cy1 = Builder.CreateICmpSLT(y, Zero);
    llvm::Value* cy2 = Builder.CreateICmpSGE(y, height);
    llvm::Value* vy2 = Builder.CreateAdd(y, height);
    llvm::Value* vy3 = Builder.CreateURem(y, height);
    y = Builder.CreateSelect(cy1, vy2, Builder.CreateSelect(cy2, vy3, y));

    llvm::Value* newCellIndex = Builder.CreateAdd(Builder.CreateMul(y, width), x);
    Builder.CreateStore(newCellIndex, CellIndex);
  }
  else{
    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
    llvm::Value* vx1 = Builder.CreateURem(cellPos, Two);
    llvm::Value* x = Builder.CreateSub(i, vx1, "x");

    llvm::Value* cx1 = Builder.CreateICmpSLT(x, Zero);
    llvm::Value* cx2 = Builder.CreateICmpSGE(x, width);
    llvm::Value* vx2 = Builder.CreateAdd(x, width);
    llvm::Value* vx3 = Builder.CreateURem(x, width);
    x = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));

    Builder.CreateStore(x, CellIndex);
  }

  llvm::Value* newCellPos = Builder.CreateAdd(cellPos, One);
  Builder.CreateStore(newCellPos, cellPosPtr);

  EmitStmt(S.getBody());

  CellIndex = 0;

  llvm::Value* Cond;

  switch(rank){
  case 3:
    Cond = Builder.CreateICmpSLT(cellPos, Seven, "cond");
    break;
  case 2:
    Cond = Builder.CreateICmpSLT(cellPos, Three, "cond");
    break;
  case 1:
    Cond = Builder.CreateICmpSLT(cellPos, One, "cond");
    break;
  default:
    assert(false && "invalid rank");
  }

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.cells.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);
  return;
}

void CodeGenFunction::EmitForallCellsEdges(const ForallMeshStmt &S){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.edges.entry");
  (void)EntryBlock; //suppress warning

  if(rank == 1){
    EdgeIndex = InnerIndex;
    Builder.CreateStore(InductionVar[0], EdgeIndex);
    EmitStmt(S.getBody());
    EdgeIndex = 0;
    return;
  }

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  EdgeIndex = InnerIndex;
  llvm::Value* edgePosPtr = InnerInductionVar;
  Builder.CreateStore(Zero, edgePosPtr);

  if(rank == 3){
    llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");
    llvm::Value* width1 = Builder.CreateAdd(width, One, "width1");
    llvm::Value* height = Builder.CreateLoad(LoopBounds[1], "height");
    llvm::Value* height1 = Builder.CreateAdd(height, One, "height1");
    llvm::Value* depth = Builder.CreateLoad(LoopBounds[2], "depth");
    llvm::Value* depth1 = Builder.CreateAdd(depth, One, "depth1");

    llvm::Value* w1h = Builder.CreateMul(width1, height, "w1h");
    llvm::Value* h1w = Builder.CreateMul(height1, width, "h1w");

    llvm::Value* c = Builder.CreateAdd(h1w, w1h);
    llvm::Value* a = Builder.CreateMul(depth1, c);
    llvm::Value* b = Builder.CreateMul(width1, height1);

    llvm::Value* i = Builder.CreateLoad(InductionVar[0]);
    llvm::Value* j = Builder.CreateLoad(InductionVar[1]);
    llvm::Value* k = Builder.CreateLoad(InductionVar[2]);

    llvm::BasicBlock *LoopBlock = createBasicBlock("forall.edges.loop");
    Builder.CreateBr(LoopBlock);

    EmitBlock(LoopBlock);

    llvm::Value* edgePos = Builder.CreateLoad(edgePosPtr, "edge.pos");

    llvm::Value* pm4 = Builder.CreateURem(edgePos, Four);
    llvm::Value* pm2 = Builder.CreateURem(edgePos, Two);
    llvm::Value* c1 = Builder.CreateICmpEQ(pm4, Two);
    llvm::Value* c2 = Builder.CreateICmpEQ(pm4, Three);
    llvm::Value* c3 = Builder.CreateICmpUGT(edgePos, Three);
    llvm::Value* x = Builder.CreateSelect(c1, Builder.CreateAdd(i, One), i);
    llvm::Value* y = Builder.CreateSelect(c2, Builder.CreateAdd(j, One), j);
    llvm::Value* z = Builder.CreateSelect(c3, Builder.CreateAdd(k, One), k);

    llvm::Value* e1 = Builder.CreateAdd(x, Builder.CreateAdd(w1h, Builder.CreateMul(y, width)));
    llvm::Value* e2 = Builder.CreateAdd(x, Builder.CreateMul(y, width1));
    llvm::Value* c4 = Builder.CreateICmpEQ(pm2, One);

    llvm::Value* newEdgeIndex =
        Builder.CreateAdd(Builder.CreateMul(z, c), Builder.CreateSelect(c4, e1, e2));

    Builder.CreateStore(newEdgeIndex, EdgeIndex);

    llvm::Value* newEdgePos = Builder.CreateAdd(edgePos, One);
    Builder.CreateStore(newEdgePos, edgePosPtr);

    EmitStmt(S.getBody());

    EdgeIndex = 0;

    llvm::Value* Cond = Builder.CreateICmpSLT(edgePos, Seven, "cond");

    llvm::BasicBlock *ExitBlock = createBasicBlock("forall.edges.exit");
    Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
    EmitBlock(ExitBlock);

    Builder.CreateStore(Zero, edgePosPtr);

    llvm::BasicBlock *LoopBlock2 = createBasicBlock("forall.edges.loop2");
    Builder.CreateBr(LoopBlock2);

    EmitBlock(LoopBlock2);

    edgePos = Builder.CreateLoad(edgePosPtr, "edge.pos");

    pm2 = Builder.CreateURem(edgePos, Two);
    llvm::Value* pd2 = Builder.CreateUDiv(edgePos, Two);

    x = Builder.CreateAdd(i, pm2);
    y = Builder.CreateAdd(j, pd2);

    llvm::Value* v1 = Builder.CreateAdd(a, Builder.CreateMul(b, k));
    llvm::Value* v2 = Builder.CreateAdd(x, Builder.CreateMul(width1, y));

    newEdgeIndex = Builder.CreateAdd(v1, v2);

    Builder.CreateStore(newEdgeIndex, EdgeIndex);

    newEdgePos = Builder.CreateAdd(edgePos, One);
    Builder.CreateStore(newEdgePos, edgePosPtr);

    EmitStmt(S.getBody());

    EdgeIndex = 0;

    Cond = Builder.CreateICmpSLT(edgePos, Three, "cond");

    llvm::BasicBlock *ExitBlock2 = createBasicBlock("forall.edges.exit2");
    Builder.CreateCondBr(Cond, LoopBlock2, ExitBlock2);
    EmitBlock(ExitBlock2);
  }
  else if(rank == 2){
    llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");
    llvm::Value* height = Builder.CreateLoad(LoopBounds[1], "height");
    llvm::Value* width1 = Builder.CreateAdd(width, One, "width1");
    llvm::Value* w1h = Builder.CreateMul(width1, height, "w1h");
    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
    llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");

    llvm::BasicBlock *LoopBlock = createBasicBlock("forall.edges.loop");
    Builder.CreateBr(LoopBlock);

    EmitBlock(LoopBlock);

    llvm::Value* edgePos = Builder.CreateLoad(edgePosPtr, "edge.pos");

    llvm::Value* c1 = Builder.CreateICmpEQ(edgePos, Two);
    llvm::Value* c2 = Builder.CreateICmpEQ(edgePos, Three);

    llvm::Value* x = Builder.CreateSelect(c1, Builder.CreateAdd(i, One), i);
    llvm::Value* y = Builder.CreateSelect(c2, Builder.CreateAdd(j, One), j);

    llvm::Value* v1 = Builder.CreateMul(y, width);
    llvm::Value* e1 = Builder.CreateAdd(w1h, Builder.CreateAdd(v1, x));
    llvm::Value* e2 = Builder.CreateAdd(Builder.CreateMul(y, width1), x);

    llvm::Value* h = Builder.CreateURem(edgePos, Two);

    llvm::Value* newEdgeIndex =
        Builder.CreateSelect(Builder.CreateICmpEQ(h, One), e1, e2);

    Builder.CreateStore(newEdgeIndex, EdgeIndex);

    EmitStmt(S.getBody());

    EdgeIndex = 0;

    llvm::Value* newEdgePos = Builder.CreateAdd(edgePos, One);
    Builder.CreateStore(newEdgePos, edgePosPtr);

    llvm::Value* Cond = Builder.CreateICmpSLT(edgePos, Three, "cond");

    llvm::BasicBlock *ExitBlock = createBasicBlock("forall.edges.exit");
    Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
    EmitBlock(ExitBlock);
  }
  else{
    assert(false && "invalid rank");
  }
}

void CodeGenFunction::EmitForallCellsFaces(const ForallMeshStmt &S){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.faces.entry");
  (void)EntryBlock; //suppress warning

  if(rank == 1){
    FaceIndex = InnerIndex;
    Builder.CreateStore(InductionVar[0], FaceIndex);
    EmitStmt(S.getBody());
    FaceIndex = 0;
    return;
  }

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  //llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  //llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  EdgeIndex = InnerIndex;
  llvm::Value* facePosPtr = InnerInductionVar;
  Builder.CreateStore(Zero, facePosPtr);

  if(rank == 3){
    assert(false && "rank 3 forall faces unimplemented");
  }
  else if(rank == 2){
    llvm::Value* width = Builder.CreateLoad(LoopBounds[0], "width");
    llvm::Value* height = Builder.CreateLoad(LoopBounds[1], "height");
    llvm::Value* width1 = Builder.CreateAdd(width, One, "width1");
    llvm::Value* w1h = Builder.CreateMul(width1, height, "w1h");
    llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
    llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");

    llvm::BasicBlock *LoopBlock = createBasicBlock("forall.faces.loop");
    Builder.CreateBr(LoopBlock);

    EmitBlock(LoopBlock);

    llvm::Value* facePos = Builder.CreateLoad(facePosPtr, "edge.pos");

    llvm::Value* c1 = Builder.CreateICmpEQ(facePos, Two);
    llvm::Value* c2 = Builder.CreateICmpEQ(facePos, Three);

    llvm::Value* x = Builder.CreateSelect(c1, Builder.CreateAdd(i, One), i);
    llvm::Value* y = Builder.CreateSelect(c2, Builder.CreateAdd(j, One), j);

    llvm::Value* v1 = Builder.CreateMul(y, width);
    llvm::Value* e1 = Builder.CreateAdd(w1h, Builder.CreateAdd(v1, x));
    llvm::Value* e2 = Builder.CreateAdd(Builder.CreateMul(y, width1), x);

    llvm::Value* h = Builder.CreateURem(facePos, Two);

    llvm::Value* newFaceIndex =
        Builder.CreateSelect(Builder.CreateICmpEQ(h, One), e1, e2);

    Builder.CreateStore(newFaceIndex, FaceIndex);

    EmitStmt(S.getBody());

    FaceIndex = 0;

    llvm::Value* newFacePos = Builder.CreateAdd(facePos, One);
    Builder.CreateStore(newFacePos, facePosPtr);

    llvm::Value* Cond = Builder.CreateICmpSLT(facePos, Three, "cond");

    llvm::BasicBlock *ExitBlock = createBasicBlock("forall.faces.exit");
    Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
    EmitBlock(ExitBlock);
  }
  else{
    assert(false && "invalid rank");
  }
}

void CodeGenFunction::EmitForallEdgesCells(const ForallMeshStmt &S){
  unsigned int rank = S.getMeshType()->rankOf();
  if(rank <= 2){
    EmitForallEdgesOrFacesCellsLowD(S, EdgeIndex);
  }
  else{
    assert(false && "forall case unimplemented");
  }
}

void
CodeGenFunction::EmitForallEdgesOrFacesCellsLowD(const ForallMeshStmt &S,
                                                 llvm::Value* OuterIndex){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  llvm::Value* Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.cells.entry");
  (void)EntryBlock; //suppress warning

  if(rank == 1){
    CellIndex = InnerIndex;
    Builder.CreateStore(Builder.CreateTrunc(OuterIndex, Int32Ty), CellIndex);

    EmitStmt(S.getBody());
    CellIndex = 0;
  }
  if(rank == 2){
    llvm::Value* w = Builder.CreateLoad(LoopBounds[0], "w");
    w = Builder.CreateZExt(w, Int64Ty, "w");
    llvm::Value* w1 = Builder.CreateAdd(w, One, "w1");

    llvm::Value* h = Builder.CreateLoad(LoopBounds[1], "h");
    h = Builder.CreateZExt(h, Int64Ty, "h");

    llvm::Value* wm1 = Builder.CreateSub(w, One, "wm1");
    llvm::Value* hm1 = Builder.CreateSub(h, One, "hm1");
    llvm::Value* w1h = Builder.CreateMul(w1, h, "w1h");

    llvm::Value* k = Builder.CreateLoad(OuterIndex, "k");
    k = Builder.CreateZExt(k, Int64Ty, "k");

    llvm::Value* c1 = Builder.CreateICmpUGE(k, w1h, "c1");
    llvm::Value* km = Builder.CreateSub(k, w1h, "km");

    llvm::Value* x =
        Builder.CreateSelect(c1, Builder.CreateURem(km, w),
                             Builder.CreateURem(k, w1), "x");

    llvm::Value* xm1 = Builder.CreateSub(x, One, "xm1");

    llvm::Value* y =
        Builder.CreateSelect(c1, Builder.CreateUDiv(km, w),
                             Builder.CreateUDiv(k, w1), "y");

    llvm::Value* ym1 = Builder.CreateSub(y, One, "ym1");

    llvm::Value* c2 = Builder.CreateICmpEQ(x, Zero, "c2");
    llvm::Value* x1 =
        Builder.CreateSelect(c1, x, Builder.CreateSelect(c2, wm1, xm1), "x1");

    llvm::Value* c3 = Builder.CreateICmpEQ(y, Zero, "c3");
    llvm::Value* y1 =
        Builder.CreateSelect(c1, Builder.CreateSelect(c3, hm1, ym1), y, "y1");

    llvm::Value* cellIndex =
        Builder.CreateAdd(Builder.CreateMul(y1, w), x1, "cellIndex.1");

    CellIndex = InnerIndex;
    Builder.CreateStore(Builder.CreateTrunc(cellIndex, Int32Ty), CellIndex);

    EmitStmt(S.getBody());

    llvm::Value* c4 = Builder.CreateICmpEQ(x, w, "c4");
    llvm::Value* x2 =
        Builder.CreateSelect(c1, x, Builder.CreateSelect(c4, Zero, x), "x2");

    llvm::Value* c5 = Builder.CreateICmpEQ(y, h, "c5");
    llvm::Value* y2 =
        Builder.CreateSelect(c1, y, Builder.CreateSelect(c5, Zero, y), "y2");

    cellIndex =
        Builder.CreateAdd(Builder.CreateMul(y2, w), x2, "cellIndex.2");

    Builder.CreateStore(Builder.CreateTrunc(cellIndex, Int32Ty), CellIndex);

    EmitStmt(S.getBody());

    CellIndex = 0;
  }
  else{
    assert(false && "invalid rank");
  }
}

void CodeGenFunction::EmitForallFacesCells(const ForallMeshStmt &S){
  unsigned int rank = S.getMeshType()->rankOf();
  if(rank <= 2){
    EmitForallEdgesOrFacesCellsLowD(S, FaceIndex);
  }
  else{
    assert(false && "forall case unimplemented");
  }
}

void
CodeGenFunction::EmitForallEdgesOrFacesVerticesLowD(const ForallMeshStmt &S,
                                                    llvm::Value* OuterIndex){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  //llvm::Value* Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.edges.entry");
  (void)EntryBlock; //suppress warning

  if(rank == 1){
    llvm::Value* One32 = llvm::ConstantInt::get(Int32Ty, 1);

    llvm::Value* k = Builder.CreateLoad(OuterIndex, "k");
    k = Builder.CreateZExt(k, Int64Ty, "k");

    VertexIndex = InnerIndex;
    llvm::Value* vertexIndex = Builder.CreateTrunc(k, Int32Ty);
    Builder.CreateStore(vertexIndex, VertexIndex);

    EmitStmt(S.getBody());

    vertexIndex = Builder.CreateAdd(vertexIndex, One32);
    Builder.CreateStore(vertexIndex, VertexIndex);

    EmitStmt(S.getBody());

    VertexIndex = 0;
  }
  else if(rank == 2){
    llvm::Value* w = Builder.CreateLoad(LoopBounds[0], "w");
    w = Builder.CreateZExt(w, Int64Ty, "w");
    llvm::Value* w1 = Builder.CreateAdd(w, One, "w1");

    llvm::Value* h = Builder.CreateLoad(LoopBounds[1], "h");
    h = Builder.CreateZExt(h, Int64Ty, "h");

    llvm::Value* w1h = Builder.CreateMul(w1, h, "w1h");

    llvm::Value* k = Builder.CreateLoad(OuterIndex, "k");
    k = Builder.CreateZExt(k, Int64Ty, "k");
    
    llvm::Value* c1 = Builder.CreateICmpUGE(k, w1h, "c1");
    llvm::Value* km = Builder.CreateSub(k, w1h, "km");

    llvm::Value* x1 =
        Builder.CreateSelect(c1, Builder.CreateURem(km, w),
                             Builder.CreateURem(k, w1), "x1");

    llvm::Value* y1 =
        Builder.CreateSelect(c1, Builder.CreateUDiv(km, w),
                             Builder.CreateUDiv(k, w1), "y1");
    
    llvm::Value* vertexIndex =
        Builder.CreateAdd(Builder.CreateMul(y1, w1), x1, "vertexIndex.1");

    VertexIndex = InnerIndex;
    Builder.CreateStore(Builder.CreateTrunc(vertexIndex, Int32Ty), VertexIndex);

    EmitStmt(S.getBody());

    llvm::Value* x2 = Builder.CreateSelect(c1, Builder.CreateAdd(x1, One), x1);
    llvm::Value* y2 = Builder.CreateSelect(c1, y1, Builder.CreateAdd(y1, One));

    vertexIndex =
        Builder.CreateAdd(Builder.CreateMul(y2, w1), x2, "vertexIndex.2");

    Builder.CreateStore(Builder.CreateTrunc(vertexIndex, Int32Ty), VertexIndex);

    EmitStmt(S.getBody());

    VertexIndex = 0;
  }
  else{
    assert(false && "forall case unimplemented");
  }
}

void CodeGenFunction::EmitForallEdgesVertices(const ForallMeshStmt &S){
  unsigned int rank = S.getMeshType()->rankOf();
  if(rank <= 2){
    EmitForallEdgesOrFacesVerticesLowD(S, EdgeIndex);
  }
  else{
    assert(false && "forall case unimplemented");
  }
}

void CodeGenFunction::EmitForallFacesVertices(const ForallMeshStmt &S){
  unsigned int rank = S.getMeshType()->rankOf();
  if(rank <= 2){
    EmitForallEdgesOrFacesVerticesLowD(S, FaceIndex);
  }
  else{
    assert(false && "forall case unimplemented");
  }
}

void CodeGenFunction::EmitForallEdges(const ForallMeshStmt &S){
  if(isGPU()){
    llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");
    
    EmitGPUPreamble(S);
    
    llvm::BasicBlock* condBlock = createBasicBlock("forall.cond");
    EmitBlock(condBlock);
    
    llvm::Value* threadId = Builder.CreateLoad(GPUThreadId, "threadId");
    
    llvm::Value* cond = Builder.CreateICmpULT(threadId, GPUNumThreads);
    
    llvm::BasicBlock* bodyBlock = createBasicBlock("forall.body");
    llvm::BasicBlock* exitBlock = createBasicBlock("forall.exit");
    
    Builder.CreateCondBr(cond, bodyBlock, exitBlock);
    
    EmitBlock(bodyBlock);
    
    EdgeIndex = GPUThreadId;
    EmitStmt(S.getBody());
    EdgeIndex = 0;
    
    threadId = Builder.CreateAdd(threadId, GPUThreadInc);
    Builder.CreateStore(threadId, GPUThreadId);
    
    Builder.CreateBr(condBlock);
    
    EmitBlock(exitBlock);
    
    llvm::Function* f = ExtractRegion(entry, exitBlock, "ForallMeshFunction");
    
    AddScoutKernel(f, S);
    
    return;
  }
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.edges.entry");
  (void)EntryBlock; //suppress warning

  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.edges_idx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(Zero, InductionVar[3]);
  InnerIndex = Builder.CreateAlloca(Int32Ty, 0, "forall.inneridx.ptr");

  SmallVector<llvm::Value*, 3> Dimensions;
  GetMeshDimensions(S.getMeshType(), Dimensions);
  llvm::Value* numEdges;
  GetNumMeshItems(Dimensions, 0, 0, &numEdges, 0);

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.edges.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  EdgeIndex = InductionVar[3];
  EmitStmt(S.getBody());
  EdgeIndex = 0;

  llvm::Value* k = Builder.CreateLoad(InductionVar[3], "forall.edges_idx");
  k = Builder.CreateAdd(k, One);
  Builder.CreateStore(k, InductionVar[3]);
  k = Builder.CreateZExt(k, Int64Ty, "k");

  llvm::Value* Cond = Builder.CreateICmpSLT(k, numEdges, "cond");

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.edges.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);

  InnerIndex = 0;
}

void CodeGenFunction::EmitForallFaces(const ForallMeshStmt &S){
  if(isGPU()){
    llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");
    
    EmitGPUPreamble(S);
    
    llvm::BasicBlock* condBlock = createBasicBlock("forall.cond");
    EmitBlock(condBlock);
    
    llvm::Value* threadId = Builder.CreateLoad(GPUThreadId, "threadId");
    
    llvm::Value* cond = Builder.CreateICmpULT(threadId, GPUNumThreads);
    
    llvm::BasicBlock* bodyBlock = createBasicBlock("forall.body");
    llvm::BasicBlock* exitBlock = createBasicBlock("forall.exit");
    
    Builder.CreateCondBr(cond, bodyBlock, exitBlock);
    
    EmitBlock(bodyBlock);
    
    FaceIndex = GPUThreadId;
    EmitStmt(S.getBody());
    FaceIndex = 0;
    
    threadId = Builder.CreateAdd(threadId, GPUThreadInc);
    Builder.CreateStore(threadId, GPUThreadId);
    
    Builder.CreateBr(condBlock);
    
    EmitBlock(exitBlock);
    
    llvm::Function* f = ExtractRegion(entry, exitBlock, "ForallMeshFunction");
    
    AddScoutKernel(f, S);
    
    return;
  }
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

  llvm::BasicBlock *EntryBlock = EmitMarkerBlock("forall.faces.entry");
  (void)EntryBlock; //suppress warning

  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.faces_idx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(Zero, InductionVar[3]);

  SmallVector<llvm::Value*, 3> Dimensions;
  GetMeshDimensions(S.getMeshType(), Dimensions);
  llvm::Value* numFaces;
  GetNumMeshItems(Dimensions, 0, 0, 0, &numFaces);

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.faces.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  FaceIndex = InductionVar[3];
  EmitStmt(S.getBody());
  FaceIndex = 0;

  llvm::Value* k = Builder.CreateLoad(InductionVar[3], "forall.faces_idx");
  k = Builder.CreateAdd(k, One);
  Builder.CreateStore(k, InductionVar[3]);
  k = Builder.CreateZExt(k, Int64Ty, "k");

  llvm::Value* Cond = Builder.CreateICmpSLT(k, numFaces, "cond");

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.faces.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);
}

void CodeGenFunction::AddScoutKernel(llvm::Function* f,
                                     const ForallMeshStmt &S){
  llvm::NamedMDNode* kernels =
      CGM.getModule().getOrInsertNamedMetadata("scout.kernels");

  llvm::SmallVector<llvm::Value*, 3> kernelData;
  kernelData.push_back(f);
  
  const MeshType* mt = S.getMeshType();
  kernelData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                           mt->getName()));
  
  MeshDecl* md = mt->getDecl();
  
  llvm::SmallVector<llvm::Value*, 16> meshFields;
  
  for (MeshDecl::field_iterator itr = md->field_begin(),
       itrEnd = md->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* fd = *itr;
    
    llvm::SmallVector<llvm::Value*, 3> fieldData;
    fieldData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                            fd->getName()));
    if(fd->isCellLocated()){
      fieldData.push_back(llvm::ConstantInt::get(Int8Ty, FIELD_CELL));
    }
    else if(fd->isVertexLocated()){
      fieldData.push_back(llvm::ConstantInt::get(Int8Ty, FIELD_VERTEX));
    }
    else if(fd->isEdgeLocated()){
      fieldData.push_back(llvm::ConstantInt::get(Int8Ty, FIELD_EDGE));
    }
    else if(fd->isFaceLocated()){
      fieldData.push_back(llvm::ConstantInt::get(Int8Ty, FIELD_FACE));
    }
    
    llvm::Value* fieldDataMD =
    llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Value*>(fieldData));
    
    meshFields.push_back(fieldDataMD);
  }
  
  llvm::Value* fieldsMD =
  llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Value*>(meshFields));
  
  kernelData.push_back(fieldsMD);
  
  kernels->addOperand(llvm::MDNode::get(CGM.getLLVMContext(), kernelData));
}

void CodeGenFunction::EmitGPUPreamble(const ForallMeshStmt& S){
  assert(isGPU());

  const VarDecl* VD = S.getMeshVarDecl();
  llvm::Value* V = LocalDeclMap.lookup(VD);
  llvm::Value* Addr = Builder.CreateAlloca(V->getType(), 0, "TheMesh_addr");
  Builder.CreateStore(V, Addr);

  SmallVector<llvm::Value*, 3> Dimensions;
  GetMeshDimensions(S.getMeshType(), Dimensions);
  
  ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();

  Builder.CreateLoad(LoopBounds[0], "TheMesh.width");
  Builder.CreateLoad(LoopBounds[1], "TheMesh.height");
  Builder.CreateLoad(LoopBounds[2], "TheMesh.depth");
  
  llvm::Value* numItems;
  
  switch(FET){
    case ForallMeshStmt::Cells:
      GetNumMeshItems(Dimensions, &numItems, 0, 0, 0);
      break;
    case ForallMeshStmt::Vertices:
      GetNumMeshItems(Dimensions, 0, &numItems, 0, 0);
      break;
    case ForallMeshStmt::Edges:
      GetNumMeshItems(Dimensions, 0, 0, &numItems, 0);
      break;
    case ForallMeshStmt::Faces:
      GetNumMeshItems(Dimensions, 0, 0, 0, &numItems);
      break;
    default:
      assert(false && "unrecognized forall type");
  }
  
  GPUNumThreads = Builder.CreateIntCast(numItems, Int32Ty, false);
  
  llvm::Value* ptr = Builder.CreateAlloca(Int32Ty, 0, "tid.x.ptr");
  llvm::Value* tid = Builder.CreateLoad(ptr, "tid.x");

  ptr = Builder.CreateAlloca(Int32Ty, 0, "ntid.x.ptr");
  llvm::Value* ntid = Builder.CreateLoad(ptr, "ntid.x");
  
  ptr = Builder.CreateAlloca(Int32Ty, 0, "ctaid.x.ptr");
  llvm::Value* ctaid = Builder.CreateLoad(ptr, "ctaid.x");

  ptr = Builder.CreateAlloca(Int32Ty, 0, "nctaid.x.ptr");
  llvm::Value* nctaid = Builder.CreateLoad(ptr, "nctaid.x");
  
  GPUThreadId = Builder.CreateAlloca(Int32Ty, 0, "threadId.ptr");
  llvm::Value* threadId = Builder.CreateAdd(tid, Builder.CreateMul(ctaid, ntid));
  Builder.CreateStore(threadId, GPUThreadId);
  
  GPUThreadInc = Builder.CreateMul(ntid, nctaid, "threadInc");
}

void CodeGenFunction::EmitLegionTask(const FunctionDecl* FD,
                                     llvm::Function* TF){

  assert(FD && TF);

  funcDecl = FD;
  taskFunc = TF;
  taskId = CGM.NextLegionTaskId++;

  // Make sure all other relevant member vars are zero'ed out 
  // so we cleanly emit the legion task.
  // Would like to encapsulate these vars in a struct or class eventually.
  legionTaskInitFunc = NULL;
  legionTaskFunc = NULL;
  legionContext = NULL;
  legionRuntime = NULL;
  meshTaskArgs = NULL;
  taskArgs = NULL;
  indexLauncher = NULL; 
  argMap = NULL;
  regions = NULL;
  subgridBounds = NULL;
  taskDeclVisitor = NULL;
  meshPos = 0;
  meshDecl = NULL;
  meshPtr = NULL;
  meshType = NULL;
  fields = {};
  taskFuncArgs = {};
  
  auto aitr = taskFunc->arg_begin();
 
  // do some checking of arguments 
  llvm::PointerType* meshPtrType = dyn_cast<llvm::PointerType>(aitr->getType());
  assert(meshPtrType && "Expected a mesh ptr");
  
  meshType = dyn_cast<llvm::StructType>(meshPtrType->getElementType());
  assert(meshType && "Expected a mesh");

  auto& B = Builder;
  llvm::BasicBlock* prevBlock = B.GetInsertBlock();
  llvm::BasicBlock::iterator prevPoint = B.GetInsertPoint();

  EmitLegionTaskInitFunction();

  EmitLegionTaskFunction();

  B.SetInsertPoint(prevBlock, prevPoint);
 
  assert(legionTaskFunc); 
  CGM.regTaskInLsciMainFunction(taskId, legionTaskFunc);

}

typedef std::vector<llvm::Value*> ValueVec;
typedef std::vector<llvm::Type*> TypeVec;

void CodeGenFunction::EmitLegionTaskInitFunction() {

  // Create the function we will be adding to
  EmitLegionTaskInitFunctionStart();

  assert(legionTaskInitFunc);

  // get, meshPtr, legionContext and legionRuntime  
  auto aitr = legionTaskInitFunc->arg_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      meshPtr = aitr;
    }
    ++aitr;
  }
  
  legionContext = aitr++;
  legionRuntime = aitr;
 
  auto& B = Builder;

  // emit entry block 
  llvm::LLVMContext& context = CGM.getLLVMContext();

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", legionTaskInitFunc);
  B.SetInsertPoint(entry);

  taskDeclVisitor = new TaskDeclVisitor(funcDecl);

  EmitUnimeshGetVecByNameFuncCalls();

  EmitArgumentMapCreateFuncCall();

  llvm::Value* Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);
  
  llvm::Value* iPtr = B.CreateAlloca(Int64Ty, 0, "i.ptr");
  B.CreateStore(Zero, iPtr);

  meshTaskArgs = CreateMeshTaskArgs();

  llvm::BasicBlock* cond = llvm::BasicBlock::Create(context, "cond", legionTaskInitFunc);
  llvm::BasicBlock* loop = llvm::BasicBlock::Create(context, "loop", legionTaskInitFunc);
  llvm::BasicBlock* merge = llvm::BasicBlock::Create(context, "merge", legionTaskInitFunc);
  
  B.CreateBr(cond);
  B.SetInsertPoint(cond);
  llvm::Value* i = B.CreateLoad(iPtr);
  
  assert(fields.size() > 0);
  llvm::Value* launchDmn =
  B.CreateStructGEP(fields[0], LSCI_VECTOR_LAUNCH_DOMAIN, "launchDomain.ptr");
  llvm::Value* volume =
  B.CreateLoad(B.CreateStructGEP(launchDmn, LSCI_DOMAIN_VOLUME), "volume");

  llvm::Value* cmp = B.CreateICmpULT(i, volume);
  B.CreateCondBr(cmp, loop, merge);
  
  B.SetInsertPoint(loop);
  
  EmitSubgridBoundsAtSetFuncCall(i);
  
  EmitArgumentMapSetPointFuncCall(i); 

  B.CreateStore(B.CreateAdd(i, One), iPtr);
  B.CreateBr(cond);
  
  B.SetInsertPoint(merge);
  
  EmitIndexLauncherCreateFuncCall();
 
  EmitAddMeshRegionReqAndFieldFuncCalls(); 

  // done with taskDeclVisitor 
  delete taskDeclVisitor;  
  taskDeclVisitor = NULL;

  EmitAddVectorRegionReqAndFieldFuncCalls();  


  EmitExecuteIndexSpaceFuncCall();

  B.CreateRetVoid();
}

 
void CodeGenFunction::EmitLegionTaskInitFunctionStart()
{
 
  assert(funcDecl && taskFunc); 

  CGLegionRuntime& R = CGM.getLegionRuntime();
  
  llvm::LLVMContext& context = CGM.getLLVMContext();
 
  // Create LegionTaskInitFunction(lsci_unimesh_t *mesh, lsci_context_t context, lsci_runtime_t runtime); 
  // This will create the task launcher, add region requirements, add fields, and execute the index space.
  // Note: context and runtime are needed for lsci_execute_index_space(runtime, context, indexLauncher).
  TypeVec params;
  
  size_t idx = 0;
  const UniformMeshType* mt = 0;
  auto aitr = taskFunc->arg_begin();
  for(auto itr = funcDecl->param_begin(), itrEnd = funcDecl->param_end();
      itr != itrEnd; ++itr){
    ParmVarDecl* pd = *itr;
    const Type* t = pd->getType().getTypePtr();
    if(t->isPointerType() || t->isReferenceType()){
      if(const UniformMeshType* ct = dyn_cast<UniformMeshType>(t->getPointeeType())){
        assert(!mt && "more than one mesh param found");
        mt = ct;
        meshPos = idx;
        params.push_back(R.PointerTy(R.UnimeshTy));
      }
      else{
        params.push_back(aitr->getType());
      }
    }
    else{
      params.push_back(aitr->getType());
    }
    ++idx;
    ++aitr;
  }
  
  assert(mt && "expected a mesh param");
  
  meshDecl = mt->getDecl();
  
  params.push_back(R.ContextTy);
  params.push_back(R.RuntimeTy);
  llvm::FunctionType* ft = llvm::FunctionType::get(VoidTy, params, false);
  
  legionTaskInitFunc =
  llvm::Function::Create(ft,
                   llvm::Function::ExternalLinkage,
                   "LegionTaskInitFunction",
                   &CGM.getModule());

  // adding metadata 
  llvm::NamedMDNode* tasks =
  CGM.getModule().getOrInsertNamedMetadata("scout.tasks");
  
  
  SmallVector<llvm::Value*, 3> taskInfo;
  taskInfo.push_back(llvm::ConstantInt::get(Int32Ty, taskId));
  taskInfo.push_back(taskFunc);
  taskInfo.push_back(legionTaskInitFunc);
 
  tasks->addOperand(llvm::MDNode::get(context, taskInfo));
}

void CodeGenFunction::EmitUnimeshGetVecByNameFuncCalls()
{
  assert(funcDecl && meshPtr && legionContext && legionRuntime);

  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  // visit the function to determine read and write uses of the mesh fields 
  taskDeclVisitor->VisitStmt(funcDecl->getBody());
  
  const MeshFieldMap& LHS = taskDeclVisitor->getLHSmap();
  const MeshFieldMap& RHS = taskDeclVisitor->getRHSmap();
  
  const MeshNameMap& MN = taskDeclVisitor->getMeshNamemap();
  assert(MN.size() == 1 && "expected one mesh");
  
  const std::string& meshName = MN.begin()->first;

  llvm::Value* firstField = 0;
  ValueVec args;

  // generate calls to lsci_unimesh_get_vec_by_name() for each field used

  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* fd = *itr;

    std::string fieldName = meshName + "." + fd->getName().str();

    bool read = RHS.find(fieldName) != RHS.end();
    bool write = LHS.find(fieldName) != LHS.end();

    if(read || write) {
      llvm::Value* field = B.CreateAlloca(R.VectorTy, 0, fd->getName() + ".ptr");
      fields.push_back(field);
      
      args = {meshPtr, B.CreateGlobalStringPtr(fd->getName()),
        field, legionContext, legionRuntime};
      
      B.CreateCall(R.UnimeshGetVecByNameFunc(), args);
      
      if(!firstField){
        firstField = field;
      }
    }
    else{
      fields.push_back(0);
    }
  }
  assert(firstField && "no mesh fields accessed");
}

 
void CodeGenFunction::EmitArgumentMapCreateFuncCall() {

  CGLegionRuntime& R = CGM.getLegionRuntime();
  auto& B = Builder;
  argMap = Builder.CreateAlloca(R.ArgumentMapTy, 0, "argMap.ptr");
  ValueVec args;
  args = {argMap};
  llvm::Function* f = R.ArgumentMapCreateFunc();
  B.CreateCall(f, args);
}

llvm::Value* CodeGenFunction::CreateMeshTaskArgs() {

  assert((fields.size() > 0) && meshPtr);
 
  llvm::Value* firstField = fields[0];
   
  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  llvm::Value* len =
  B.CreateLoad(B.CreateStructGEP(firstField, LSCI_VECTOR_SUBGRID_BOUNDS_LEN), "len");
  
  llvm::Value* rank =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_DIMS), "rank");
  
  llvm::Value* width =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_WIDTH), "width");
  
  llvm::Value* height =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_HEIGHT), "height");
  
  llvm::Value* depth =
  B.CreateLoad(B.CreateStructGEP(meshPtr, LSCI_UNIMESH_DEPTH), "depth");
  
  llvm::Value* mtargs =
  B.CreateAlloca(R.MeshTaskArgsTy, 0, "meshTaskArgs.ptr");

  B.CreateStore(rank, B.CreateStructGEP(mtargs, LSCI_MTARGS_RANK));
  B.CreateStore(width, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_WIDTH));
  B.CreateStore(height, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_HEIGHT));
  B.CreateStore(depth, B.CreateStructGEP(mtargs, LSCI_MTARGS_GLOBAL_DEPTH));
  B.CreateStore(len, B.CreateStructGEP(mtargs, LSCI_MTARGS_SUBGRID_LEN));
  
  return mtargs;
}

void CodeGenFunction::EmitSubgridBoundsAtSetFuncCall(llvm::Value* i) {

  assert((fields.size() > 0) && meshTaskArgs && i);
 
  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  llvm::Value* bounds =
  B.CreateLoad(B.CreateStructGEP(fields[0], LSCI_VECTOR_SUBGRID_BOUNDS), "bounds");
 
  llvm::Value* rect1dStoragePtr = B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_SUBGRID_BOUNDS);
  ValueVec args;
  args = {bounds, i, rect1dStoragePtr};
  B.CreateCall(R.SubgridBoundsAtSetFunc(), args);
}

void CodeGenFunction::EmitArgumentMapSetPointFuncCall(llvm::Value* i) {  

  assert(argMap && meshTaskArgs && i);

  CGLegionRuntime& R = CGM.getLegionRuntime();
  auto& B = Builder;

  ValueVec args;

  args = {argMap, i, B.CreateBitCast(meshTaskArgs, R.VoidPtrTy),
    llvm::ConstantInt::get(Int64Ty, sizeof(lsci_mesh_task_args_t))};
  
  B.CreateCall(R.ArgumentMapSetPointFunc(), args);
}

void CodeGenFunction::EmitIndexLauncherCreateFuncCall() {

  assert((fields.size() > 0) && argMap );

  CGLegionRuntime& R = CGM.getLegionRuntime();
  auto& B = Builder;

  indexLauncher =
  B.CreateAlloca(R.IndexLauncherTy, 0, "indexLauncher");
  
  llvm::Value* launchDomain =
  B.CreateStructGEP(fields[0], LSCI_VECTOR_LAUNCH_DOMAIN, "launchDomain.ptr");

  llvm::Value* TaskId = llvm::ConstantInt::get(Int32Ty, taskId);

  llvm::Value* ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);

  ValueVec args;
  args = {indexLauncher, TaskId, launchDomain,
    R.GetNull(Int8Ty), ConstantZero, argMap};

  B.CreateCall(R.IndexLauncherCreateFunc(), args);

}

void CodeGenFunction::EmitAddMeshRegionReqAndFieldFuncCalls() { 

  assert(funcDecl && meshDecl && (fields.size() > 0) && indexLauncher && taskDeclVisitor);

  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  // get mesh name
  const MeshNameMap& MN = taskDeclVisitor->getMeshNamemap();
  assert(MN.size() == 1 && "expected one mesh");
  const std::string& meshName = MN.begin()->first;

  const MeshFieldMap& LHS = taskDeclVisitor->getLHSmap();
  const MeshFieldMap& RHS = taskDeclVisitor->getRHSmap();

  ValueVec args;

  uint32_t j = 0;
  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    
    MeshFieldDecl* fd = *itr;
    llvm::Value* field = fields[j];
    
    if(field) {
      
      assert(fd);
      std::string fieldName = meshName + "." + fd->getName().str();
      bool read = RHS.find(fieldName) != RHS.end();
      bool write = LHS.find(fieldName) != LHS.end();
     
      llvm::Value* mode = read ? (write ? R.ReadWriteVal : R.ReadOnlyVal) : R.WriteDiscardVal;
      
      llvm::Value* fieldId =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
      
      llvm::Value* logicalPartition =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
      
      llvm::Value* logicalRegion =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
      
      args =
      {indexLauncher, logicalPartition, llvm::ConstantInt::get(Int32Ty, 0),
        mode, R.ExclusiveVal, logicalRegion};
      
      B.CreateCall(R.AddRegionRequirementFunc(), args);
      
      args =
      {indexLauncher, llvm::ConstantInt::get(Int32Ty, j), fieldId};
      B.CreateCall(R.AddFieldFunc(), args);
    }
    
    ++j;
  }
}

void CodeGenFunction::EmitAddVectorRegionReqAndFieldFuncCalls() { 

  assert(legionTaskInitFunc && funcDecl && indexLauncher);

  ValueVec args;
  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

  uint32_t j = fields.size();
  auto aitr = legionTaskInitFunc->arg_begin();
  auto pitr = funcDecl->param_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      ++aitr;
      ++pitr;
      continue;
    }
    
    const llvm::Type* t = aitr->getType();
    llvm::Value* mode;
    
    if((*pitr)->getType().isConstQualified()){
      mode = R.ReadOnlyVal;
    }
    else{
      mode = R.ReadWriteVal;
    }

    if(const llvm::PointerType* pt = dyn_cast<llvm::PointerType>(t)){
      if(const llvm::StructType* st = dyn_cast<llvm::StructType>(pt->getElementType())){
        for(auto eitr = st->element_begin(), eitrEnd = st->element_end();
            eitr != eitrEnd; ++eitr){
          const llvm::Type* et = *eitr;

          llvm::Value* len;
          
          if(const llvm::ArrayType* at = dyn_cast<llvm::ArrayType>(et)){
            len = llvm::ConstantInt::get(Int64Ty, at->getNumElements());
            et = at->getElementType();
          }
          else{
            len = One;
          }
          
          llvm::Value* field = B.CreateAlloca(R.VectorTy, 0);
          
          if(et->isFloatTy()){
            args = {field, len, R.TypeFloatVal, legionContext, legionRuntime};
          }
          else if(et->isDoubleTy()){
            args = {field, len, R.TypeDoubleVal, legionContext, legionRuntime};
          }
          else if(et->isIntegerTy(32)){
            args = {field, len, R.TypeInt32Val, legionContext, legionRuntime};
          }
          else if(et->isIntegerTy(64)){
            args = {field, len, R.TypeInt64Val, legionContext, legionRuntime};
          }
          else{
            assert(false && "invalid task scalar param type");
          }
          
          B.CreateCall(R.VectorCreateFunc(), args);
          
          llvm::Value* fieldId =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
          
          llvm::Value* logicalPartition =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
          
          llvm::Value* logicalRegion =
          B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
          
          args =
          {indexLauncher, logicalPartition, llvm::ConstantInt::get(Int32Ty, 0),
            mode, R.ExclusiveVal, logicalRegion};
          
          B.CreateCall(R.AddRegionRequirementFunc(), args);
          
          args =
          {indexLauncher, llvm::ConstantInt::get(Int32Ty, j++), fieldId};
          B.CreateCall(R.AddFieldFunc(), args);
          
        }
      }
      else{
        assert(false && "invalid pointer type");
      }
    }
    else{
      llvm::Value* field = B.CreateAlloca(R.VectorTy, 0, aitr->getName());
      
      if(t->isFloatTy()){
        args = {field, One, R.TypeFloatVal, legionContext, legionRuntime};
      }
      else if(t->isDoubleTy()){
        args = {field, One, R.TypeDoubleVal, legionContext, legionRuntime};
      }
      else if(t->isIntegerTy(32)){
        args = {field, One, R.TypeInt32Val, legionContext, legionRuntime};
      }
      else if(t->isIntegerTy(64)){
        args = {field, One, R.TypeInt64Val, legionContext, legionRuntime};
      }
      else{
        assert(false && "invalid task scalar param type");
      }
      
      B.CreateCall(R.VectorCreateFunc(), args);
      
      llvm::Value* fieldId =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_FID), "fieldId");
      
      llvm::Value* logicalPartition =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_PARTITION), "logicalPartition");
      
      llvm::Value* logicalRegion =
      B.CreateLoad(B.CreateStructGEP(field, LSCI_VECTOR_LOGICAL_REGION), "logicalRegion");
      
      args =
      {indexLauncher, logicalPartition, llvm::ConstantInt::get(Int32Ty, 0),
        mode, R.ExclusiveVal, logicalRegion};
      
      B.CreateCall(R.AddRegionRequirementFunc(), args);
      
      args =
      {indexLauncher, llvm::ConstantInt::get(Int32Ty, j++), fieldId};
      B.CreateCall(R.AddFieldFunc(), args);
    }
    
    ++aitr;
    ++pitr;
  }
} 

void CodeGenFunction::EmitExecuteIndexSpaceFuncCall(){

  assert(legionRuntime && legionContext && indexLauncher);

  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  ValueVec args;
  args = {legionRuntime, legionContext, indexLauncher};
  B.CreateCall(R.ExecuteIndexSpaceFunc(), args);
}
  

void CodeGenFunction::EmitLegionTaskFunction() { 


  EmitLegionTaskFunctionStart();

  auto& B = Builder;
  llvm::LLVMContext& context = CGM.getLLVMContext();

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", legionTaskFunc);
  B.SetInsertPoint(entry);

  EmitMeshRawRectPtr1dFuncCalls(); 
  EmitVectorRawRectPtr1dFuncCalls(); 
  EmitTaskFuncCall();

  B.CreateRetVoid();
}


void CodeGenFunction::EmitLegionTaskFunctionStart() { 

  llvm::LLVMContext& context = CGM.getLLVMContext();
  CGLegionRuntime& R = CGM.getLegionRuntime();


   //fetch or create type for lsci_task_args_t
   llvm::StructType *TaskArgsTy = CGM.getModule().getTypeByName("struct.lsci_task_args_t");
   if (!TaskArgsTy) {
     std::vector<llvm::Type*> structMembers = {
       R.ContextTy,
       R.RuntimeTy,
       R.Int32Ty,
       R.Int64Ty,
       R.PhysicalRegionsTy,
       R.VoidPtrTy};
     TaskArgsTy = llvm::StructType::create(context, structMembers, "struct.lsci_task_args_t");
   }

   llvm::PointerType* TaskArgsPtrTy = llvm::PointerType::get(TaskArgsTy, 0);

   // use lsci_task_args_t to create main_task function type and function
   TypeVec params = {TaskArgsPtrTy};

  llvm::FunctionType* funcType = llvm::FunctionType::get(VoidTy, params, false);
  
  legionTaskFunc = llvm::Function::Create(funcType,
                                    llvm::Function::ExternalLinkage,
                                    "LegionTaskFunction",
                                    &CGM.getModule());
  
  auto aitr = legionTaskFunc->arg_begin();
  taskArgs = aitr;
  taskArgs->setName("task_args_ptr");
}
 
void CodeGenFunction::EmitMeshRawRectPtr1dFuncCalls() {

  assert(taskArgs && meshType && meshDecl && (fields.size() > 0) && legionContext && legionRuntime);

  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();


  // load the taskArgs pointer
  llvm::Value* taskArgsAddr = B.CreateAlloca(R.PointerTy(R.TaskArgsTy), 0, "task_args.addr");
  B.CreateAlignedStore(taskArgs, taskArgsAddr, 8);
  llvm::LoadInst* loadTaskArgsPtr = B.CreateAlignedLoad(taskArgsAddr, 8, "task_args_loaded.ptr");

  // load the task
  
  llvm::Value* task = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_TASK), "task");
  legionContext = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_CONTEXT), "task");
  legionRuntime = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_RUNTIME), "task");
  //llvm::Value* taskAddr = B.CreateAlloca(R.PointerTy(R.TaskTy), 0, "task.addr");
  //llvm::Value* taskPtr = B.CreateStructGEP(taskAddr, LSCI_TARGS_TASK); 
  //task = B.CreateAlignedLoad(taskPtr, 8, "task.loaded");

  // load the mesh task args ptr from taskArgs local_argsp field
  llvm::Value* meshTaskArgsAddr = B.CreateAlloca(R.PointerTy(R.MeshTaskArgsTy), 0, "mtargs.addr");
  llvm::Value* localArgsp = B.CreateStructGEP(loadTaskArgsPtr, LSCI_TARGS_LOCAL_ARGSP); 
  llvm::LoadInst* loadedLocalArgsp = B.CreateAlignedLoad(localArgsp, 8, "local_argsp.loaded");

  // must cast to a lsci_mesh_task_args_t* , since it is a void*
  llvm::Value* loadedMtargsp = B.CreateBitCast(loadedLocalArgsp, R.PointerTy(R.MeshTaskArgsTy), "mtargsp.loaded"); 
  B.CreateAlignedStore(loadedMtargsp, meshTaskArgsAddr, 8);
  meshTaskArgs = B.CreateAlignedLoad(meshTaskArgsAddr, 8, "mesh_task_args.ptr");

  // cast the *lsci_rect_1d_storage_t to lsci_rect_1d (which is a void pointer)
  subgridBounds = 
  B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_SUBGRID_BOUNDS), R.Rect1dTy);
 
  mesh = B.CreateAlloca(meshType, 0, "mesh.ptr");
  regions = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_REGIONS), "regions");
 
  uint32_t j = 0;
  ValueVec args;

  for(MeshDecl::field_iterator itr = meshDecl->field_begin(),
      itrEnd = meshDecl->field_end(); itr != itrEnd; ++itr){
    
    llvm::Value* field = fields[j];

    llvm::Value* mf = B.CreateStructGEP(mesh, j);
    
    if(field){
      MeshFieldDecl* fd = *itr;
      llvm::Type* ft = ConvertType(fd->getType());
      
      args = {regions};
      
      if(ft->isFloatTy()){
        args.push_back(R.TypeFloatVal);
      }
      else if(ft->isDoubleTy()){
        args.push_back(R.TypeDoubleVal);
      }
      else if(ft->isIntegerTy(32)){
        args.push_back(R.TypeInt32Val);
      }
      else if(ft->isIntegerTy(64)){
        args.push_back(R.TypeInt64Val);
      }
      else{
        assert(false && "unhandled mesh field type");
      }
      
      args.push_back(llvm::ConstantInt::get(Int64Ty, j));
      args.push_back(llvm::ConstantInt::get(Int32Ty, 0));
      
      args.push_back(task); 
      args.push_back(legionContext); 
      args.push_back(legionRuntime); 

      llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
      llvm::Value* cv = B.CreateBitCast(fp, meshType->getTypeAtIndex(j));
      
      B.CreateStore(cv, mf);
    }
    else{
      llvm::PointerType* pt = dyn_cast<llvm::PointerType>(mf->getType());
      assert(pt && "expected a pointer type");
      
      llvm::PointerType* et = dyn_cast<llvm::PointerType>(pt->getElementType());
      assert(et && "expected a pointer element type");
      
      B.CreateStore(llvm::ConstantPointerNull::get(et), mf);
    }
    
    ++j;
  }
}
 
void CodeGenFunction::EmitVectorRawRectPtr1dFuncCalls() {

  assert(taskArgs && legionTaskInitFunc && funcDecl && mesh && regions && subgridBounds);

  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();

  ValueVec args;
  uint32_t j = fields.size();
  llvm::Value* task = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_TASK), "task");
  legionContext = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_CONTEXT), "task");
  legionRuntime = B.CreateLoad(B.CreateStructGEP(taskArgs, LSCI_TARGS_RUNTIME), "task");
 
  // create calls to  lsci_raw_rect_ptr_1d
  uint32_t k = j;
  auto aitr = legionTaskInitFunc->arg_begin();
  auto pitr = funcDecl->param_begin();
  for(size_t i = 0; i < legionTaskInitFunc->arg_size() - 2; ++i){
    if(i == meshPos){
      taskFuncArgs.push_back(mesh);
      ++aitr;
      ++pitr;
      continue;
    }
    
    llvm::Type* t = aitr->getType();
    
    if(llvm::PointerType* pt = dyn_cast<llvm::PointerType>(t)){
      if(llvm::StructType* st = dyn_cast<llvm::StructType>(pt->getElementType())){
        llvm::Value* arg = B.CreateAlloca(st, 0, aitr->getName());
        taskFuncArgs.push_back(arg);
        
        size_t ei = 0;
        for(auto eitr = st->element_begin(), eitrEnd = st->element_end();
            eitr != eitrEnd; ++eitr){
          llvm::Type* et = *eitr;
          
          bool array;
          if(const llvm::ArrayType* at = dyn_cast<llvm::ArrayType>(et)){
            et = at->getElementType();
            array = true;
          }
          else{
            array = false;
          }
          
          args = {regions};
          
          if(et->isFloatTy()){
            args.push_back(R.TypeFloatVal);
          }
          else if(et->isDoubleTy()){
            args.push_back(R.TypeDoubleVal);
          }
          else if(et->isIntegerTy(32)){
            args.push_back(R.TypeInt32Val);
          }
          else if(et->isIntegerTy(64)){
            args.push_back(R.TypeInt64Val);
          }
          else{
            assert(false && "invalid task scalar param type");
          }
          
          args.push_back(llvm::ConstantInt::get(Int64Ty, k++));
          args.push_back(llvm::ConstantInt::get(Int32Ty, 0));
          args.push_back(task); 
          args.push_back(legionContext); 
          args.push_back(legionRuntime); 
  
          llvm::Value* ep = B.CreateStructGEP(arg, ei++);
          llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
          llvm::Value* cv = B.CreateBitCast(fp, R.PointerTy(et));

          if(array){
            ep = B.CreateBitCast(ep, R.PointerTy(R.PointerTy(et)));
          }
          else{
            cv = B.CreateLoad(cv);
          }
          
          B.CreateStore(cv, ep);
        }
      }
      else{
        assert(false && "invalid pointer type");
      }
    }
    else{
      llvm::Value* arg = B.CreateAlloca(t, 0, aitr->getName());
      
      args = {regions};
      
      if(t->isFloatTy()){
        args.push_back(R.TypeFloatVal);
      }
      else if(t->isDoubleTy()){
        args.push_back(R.TypeDoubleVal);
      }
      else if(t->isIntegerTy(32)){
        args.push_back(R.TypeInt32Val);
      }
      else if(t->isIntegerTy(64)){
        args.push_back(R.TypeInt64Val);
      }
      else{
        assert(false && "invalid task scalar param type");
      }
      
      args.push_back(llvm::ConstantInt::get(Int64Ty, k++));
      args.push_back(llvm::ConstantInt::get(Int32Ty, 0));
      args.push_back(task); 
      args.push_back(legionContext); 
      args.push_back(legionRuntime); 
      
      llvm::Value* fp = B.CreateCall(R.RawRectPtr1dFunc(), args);
      llvm::Value* cv = B.CreateBitCast(fp, R.PointerTy(t));
      B.CreateStore(B.CreateLoad(cv), arg);
      
      taskFuncArgs.push_back(B.CreateLoad(arg));
    }
    
    ++aitr;
    ++pitr;
  }
}

// This emits the code that was in the original function preceded by the "task" indicator in 
// the Scout program
void CodeGenFunction::EmitTaskFuncCall() {

  assert(meshTaskArgs && mesh && taskFunc && (taskFuncArgs.size() > 0));
  
  auto& B = Builder;
  CGLegionRuntime& R = CGM.getLegionRuntime();
  
  // Must get dims and rank from mesh_task_args_t, since we
  // are parsing this function before the mesh has
  // been instantiated, so we don't know dimensions.
  // The "mesh" struct has pointers to the data for fields first, 
  //then width, height, depth, rank stored after the fields.

  int32_t j = fields.size();

  llvm::Value* TAwidthPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_WIDTH), R.PointerTy(R.Int32Ty));
  llvm::Value* TAwidth = B.CreateAlignedLoad(TAwidthPtr, 8, "task_arg_width");
  llvm::Value* widthPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAwidth, widthPtr, 4);

  llvm::Value* TAheightPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_HEIGHT), R.PointerTy(R.Int32Ty));
  llvm::Value* TAheight = B.CreateAlignedLoad(TAheightPtr, 8, "task_arg_height");
  llvm::Value* heightPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAheight, heightPtr, 4);

  llvm::Value* TAdepthPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_GLOBAL_DEPTH), R.PointerTy(R.Int32Ty));
  llvm::Value* TAdepth = B.CreateAlignedLoad(TAdepthPtr, 8, "task_arg_depth");
  llvm::Value* depthPtr = B.CreateStructGEP(mesh, j++);
  B.CreateAlignedStore(TAdepth, depthPtr, 4);

  llvm::Value* TArankPtr = 
      B.CreateBitCast(B.CreateStructGEP(meshTaskArgs, LSCI_MTARGS_RANK), R.PointerTy(R.Int32Ty));
  llvm::Value* TArank = B.CreateAlignedLoad(TArankPtr, 8, "task_arg_rank");
  llvm::Value* rankPtr = B.CreateStructGEP(mesh, j);
  B.CreateAlignedStore(TArank, rankPtr, 4);

  B.CreateCall(taskFunc, taskFuncArgs);
}

void CodeGenFunction::EmitForallMeshStmt(const ForallMeshStmt &S) {
  const VarDecl* VD = S.getMeshVarDecl();

  ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();

  // handle nested forall, e.g: forall vertices within a forall cells
  if(const ImplicitMeshParamDecl* IP = dyn_cast<ImplicitMeshParamDecl>(VD)){
    switch(IP->getElementType()){
    case ImplicitMeshParamDecl::Cells:
      switch(FET){
      case ForallMeshStmt::Vertices:
        EmitForallCellsVertices(S);
        return;
      case ForallMeshStmt::Edges:
        EmitForallCellsEdges(S);
        return;
      case ForallMeshStmt::Faces:
        EmitForallCellsFaces(S);
        return;
      default:
        assert(false && "invalid forall case");
        return;
      }
    case ImplicitMeshParamDecl::Vertices:
      switch(FET){
      case ForallMeshStmt::Cells:
        EmitForallVerticesCells(S);
        return;
      case ForallMeshStmt::Edges:
        assert(false && "unimplemented forall case");
        return;
      case ForallMeshStmt::Faces:
        assert(false && "unimplemented forall case");
        return;
      default:
        assert(false && "invalid forall case");
        return;
      }
    case ImplicitMeshParamDecl::Edges:
      switch(FET){
      case ForallMeshStmt::Cells:
        EmitForallEdgesCells(S);
        return;
      case ForallMeshStmt::Vertices:
        EmitForallEdgesVertices(S);
        return;
      case ForallMeshStmt::Faces:
        assert(false && "unimplemented forall case");
        return;
      default:
        assert(false && "invalid forall case");
        return;
      }
    case ImplicitMeshParamDecl::Faces:
      switch(FET){
      case ForallMeshStmt::Cells:
        EmitForallFacesCells(S);
        return;
      case ForallMeshStmt::Vertices:
        EmitForallFacesVertices(S);
        return;
      case ForallMeshStmt::Edges:
        assert(false && "unimplemented forall case");
        return;
      default:
        assert(false && "invalid forall case");
        return;
      }
      default:
        assert(false && "invalid forall case");
    }
  }

  llvm::Value* ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);

  ResetVars();

  //get mesh Base Addr
  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  llvm::StringRef MeshName = MeshBaseAddr->getName();

  // find number of fields
  MeshDecl* MD =  S.getMeshType()->getDecl();
  unsigned int nfields = MD->fields();

  // Extract width/height/depth from the mesh for this rank
  // note: width/height depth are stored after mesh fields
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    LoopBounds[i] = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+i, IRNameStr);
  }

  if(FET == ForallMeshStmt::Edges){
    EmitForallEdges(S);
    return;
  }
  else if(FET == ForallMeshStmt::Faces){
    EmitForallFaces(S);
    return;
  }

  //need a marker for start of Forall for CodeExtraction
  llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");

  // Track down the mesh meta data. 
  EmitForallMeshMDBlock(S);

  if(isGPU()){
    EmitGPUPreamble(S);

    llvm::BasicBlock* condBlock = createBasicBlock("forall.cond");
    EmitBlock(condBlock);
    
    llvm::Value* threadId = Builder.CreateLoad(GPUThreadId, "threadId");
    
    llvm::Value* cond = Builder.CreateICmpULT(threadId, GPUNumThreads);
    
    llvm::BasicBlock* bodyBlock = createBasicBlock("forall.body");
    llvm::BasicBlock* exitBlock = createBasicBlock("forall.exit");
    
    Builder.CreateCondBr(cond, bodyBlock, exitBlock);
    
    EmitBlock(bodyBlock);
    
    CellIndex = GPUThreadId;
    EmitStmt(S.getBody());
    CellIndex = 0;
    
    threadId = Builder.CreateAdd(threadId, GPUThreadInc);
    Builder.CreateStore(threadId, GPUThreadId);
    
    Builder.CreateBr(condBlock);

    EmitBlock(exitBlock);
    
    llvm::Function* f = ExtractRegion(entry, exitBlock, "ForallMeshFunction");
    
    AddScoutKernel(f, S);

    return;
  }

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

  InnerInductionVar =
      Builder.CreateAlloca(Int32Ty, 0, "forall.inneridx_ind.ptr");

  InnerIndex = Builder.CreateAlloca(Int32Ty, 0, "forall.inneridx.ptr");

  // extract rank from mesh stored after width/height/depth
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  Rank = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+3, IRNameStr);

  EmitForallMeshLoop(S, 3);

  // reset Loopbounds, Rank and induction var
  // so width/height etc can't be called after forall
  ResetVars();
  Rank = 0;

  //need a marker for end of Forall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("forall.exit");

  // Extract Blocks to function and replace w/ call to function
  if(!inLLDB()){
    ExtractRegion(entry, exit, "ForallMeshFunction");
  }
}


//generate one of the nested loops
void CodeGenFunction::EmitForallMeshLoop(const ForallMeshStmt &S, unsigned r) {
  unsigned int rank = S.getMeshType()->rankOf();

  RegionCounter Cnt = getPGORegionCounter(&S);
  (void)Cnt; //suppress warning 
 
  llvm::StringRef MeshName = S.getMeshVarDecl()->getName();

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

  // Extract the loop bounds from the mesh for this rank
  sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), DimNames[r-1]);
  llvm::Value *LoopBound  = Builder.CreateLoad(LoopBounds[r-1], IRNameStr);

  ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();
  if(FET == ForallMeshStmt::Vertices){
    if(r <= rank){
      LoopBound = Builder.CreateAdd(LoopBound, ConstantOne);
    }
    VertexIndex = InductionVar[3];
  }

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

  VertexIndex = 0;
  CellIndex = 0;
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


void CodeGenFunction::EmitForallMeshMDBlock(const ForallMeshStmt &S) {

  llvm::NamedMDNode *MeshMD = CGM.getModule().getNamedMetadata("scout.meshmd");
  assert(MeshMD != 0 && "unable to find module-level mesh metadata!");

  llvm::BasicBlock *entry = createBasicBlock("forall.md");
  llvm::BranchInst *BI = Builder.CreateBr(entry);

  // find meta data for mesh used in this forall
  StringRef MeshName = S.getMeshVarDecl()->getName();
  StringRef MeshTypeName =  S.getMeshType()->getName();
  for (llvm::NamedMDNode::op_iterator II = MeshMD->op_begin(), IE = MeshMD->op_end();
      II != IE; ++II) {
    if((*II)->getOperand(0)->getName() == MeshTypeName) {
      BI->setMetadata(MeshName, *II);
    }
  }

  ForallMeshStmt *SP = const_cast<ForallMeshStmt *>(&S);
  ForallVisitor v(SP);
  v.Visit(SP);

  //find fields used on LHS and add to metadata
  const MeshFieldMap LHS = v.getLHSmap();
  EmitMeshFieldsUsedMD(LHS, "LHS", BI);

  //find fields used on RHS and add to metadata
  const MeshFieldMap RHS = v.getRHSmap();
  EmitMeshFieldsUsedMD(RHS, "RHS", BI);

  EmitBlock(entry);

}

// Emit a branch and block. used as markers for code extraction
llvm::BasicBlock *CodeGenFunction::EmitMarkerBlock(const std::string name) {
  llvm::BasicBlock *entry = createBasicBlock(name);
  Builder.CreateBr(entry);
  EmitBlock(entry);
  return entry;
}



// Extract blocks to function and replace w/ call to function
llvm::Function* CodeGenFunction:: ExtractRegion(llvm::BasicBlock *entry, llvm::BasicBlock *exit, const std::string name) {
  std::vector< llvm::BasicBlock * > Blocks;

  llvm::Function::iterator BB = CurFn->begin();

  //SC_TODO: is there a better way rather than using name?
  // find start marker
  for( ; BB->getName() != entry->getName(); ++BB) { }

  // collect forall basic blocks up to exit
  for( ; BB->getName() != exit->getName(); ++BB) {

    // look for function local metadata
    for (llvm::BasicBlock::const_iterator II = BB->begin(), IE = BB->end(); II != IE; ++II) {
      for(unsigned i = 0, e = II->getNumOperands(); i!=e; ++i){
        if(llvm::MDNode *N = dyn_cast_or_null<llvm::MDNode>(II->getOperand(i))){
          if (N->isFunctionLocal()) {
            // just remove function local metadata
            // see http://lists.cs.uiuc.edu/pipermail/llvmdev/2013-November/068205.html
            N->replaceOperandWith(i, 0);
          }
        } 
      }
    }
    Blocks.push_back(BB);
  }

  //SC_TODO: should we be using a DominatorTree?
  //llvm::DominatorTree DT;

  llvm::CodeExtractor codeExtractor(Blocks, 0/*&DT*/, false);

  llvm::Function *ForallFn = codeExtractor.extractCodeRegion();

  ForallFn->setName(name);
  return ForallFn;
}


void CodeGenFunction::EmitForallArrayStmt(const ForallArrayStmt &S) {

  //need a marker for start of Forall for CodeExtraction
  llvm::BasicBlock *entry = EmitMarkerBlock("forall.entry");

  EmitForallArrayLoop(S, S.getDims());

  //need a marker for end of Forall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("forall.exit");

  // Extract Blocks to function and replace w/ call to function

  if(!inLLDB()){
  	ExtractRegion(entry, exit, "ForallArrayFunction");
  }
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

  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  llvm::StringRef MeshName = MeshBaseAddr->getName();

  // find number of fields
  MeshDecl* MD =  S.getMeshType()->getDecl();
  unsigned int nfields = MD->fields();

  ResetVars();

  // Add render target argument
  
  const VarDecl* RTVD = S.getRenderTargetVarDecl();

  llvm::Value* RTAlloc;
  
  if ((RTVD->hasLinkage() || RTVD->isStaticDataMember())
      && RTVD->getTLSKind() != VarDecl::TLS_Dynamic) {
    RTAlloc = CGM.GetAddrOfGlobalVar(RTVD);
  }
  else{
    RTAlloc = LocalDeclMap.lookup(RTVD);
  }

  // Check if it's a window or image type
  // cuz we don't handle images yet.
  const clang::Type &Ty = *getContext().getCanonicalType(RTVD->getType()).getTypePtr();
  
  llvm::SmallVector< llvm::Value *, 4 > Args;
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
  
  if(Ty.getTypeClass() != Type::Window){
  	RTAlloc = Builder.CreateLoad(RTAlloc);
  }

  // cast scout.window_t** to void**
  llvm::Value* int8PtrPtrRTAlloc = Builder.CreateBitCast(RTAlloc, Int8PtrPtrTy, "");

  // dereference the void** 
  llvm::Value* int8PtrRTAlloc = Builder.CreateLoad(int8PtrPtrRTAlloc, "derefwin");

  // put the window on the arg list
  Args.push_back(int8PtrRTAlloc);

  // create linear loop index as 4th element and zero-initialize
  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "renderall.linearidx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[3]);

  RenderallMeshStmt::MeshElementType ET = S.getMeshElementRef();

  SmallVector<llvm::Value*, 3> Dimensions;
  GetMeshDimensions(S.getMeshType(), Dimensions);

  // make quad renderable and add to the window and return color pointer
  // use same args as for RenderallUniformBeginFunction
  // in the future, this will be a mesh renderable

  llvm::Function *WinQuadRendFunc;
  llvm::Function *WinPaintFunc = CGM.getScoutRuntime().CreateWindowPaintFunction();
  
  bool cellLoop = false;
  
  switch(ET){
    case ForallMeshStmt::Cells:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableColorsFunction();
      cellLoop = true;
      break;
    case ForallMeshStmt::Vertices:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableVertexColorsFunction();
      break;
    case ForallMeshStmt::Edges:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableEdgeColorsFunction();
      break;
    case ForallMeshStmt::Faces:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableEdgeColorsFunction();
      break;
    default:
      assert(false && "unrecognized renderall type");
  }

  // %1 = call <4 x float>* @__scrt_window_quad_renderable_colors(i32 %HeatMeshType.width.ptr14, i32 %HeatMeshType.height.ptr16, i32 %HeatMeshType.depth.ptr18, i8* %derefwin)
  Color = Builder.CreateCall(WinQuadRendFunc, ArrayRef<llvm::Value *>(Args), "localcolor.ptr");

  // extract rank from mesh stored after width/height/depth
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  Rank = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+3, IRNameStr);

  if(cellLoop){
    // renderall loops + body
    EmitRenderallMeshLoop(S, 3);
  }
  else{
    EmitRenderallVerticesEdgesFaces(S);
  }
  
  // paint window (draws all renderables) (does clear beforehand, and swap buffers after)
  if(S.isLast()){
    Args.clear();
    Args.push_back(int8PtrRTAlloc);
    Builder.CreateCall(WinPaintFunc, ArrayRef<llvm::Value *>(Args));
  }

  // reset Loopbounds, Rank, induction var
  // so width/height etc can't be called after renderall
  ResetVars();
  Rank = 0;

  //need a marker for end of Renderall for CodeExtraction
  llvm::BasicBlock *exit = EmitMarkerBlock("renderall.exit");

  if(!inLLDB()){
  	ExtractRegion(entry, exit, "RenderallFunction");
  }
}

void CodeGenFunction::EmitRenderallVerticesEdgesFaces(const RenderallMeshStmt &S){
	llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
	llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

	llvm::BasicBlock *EntryBlock = EmitMarkerBlock("renderall.entry");
	(void)EntryBlock; //suppress warning

	InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "renderall.idx.ptr");
	//zero-initialize induction var
	Builder.CreateStore(Zero, InductionVar[3]);
	InnerIndex = Builder.CreateAlloca(Int32Ty, 0, "renderall.inneridx.ptr");

	SmallVector<llvm::Value*, 3> Dimensions;
	GetMeshDimensions(S.getMeshType(), Dimensions);

	RenderallMeshStmt::MeshElementType ET = S.getMeshElementRef();

	//llvm::Function *WinQuadRendFunc;
	//llvm::Function *WinPaintFunc;

	llvm::BasicBlock *LoopBlock = createBasicBlock("renderall.loop");
	Builder.CreateBr(LoopBlock);

	EmitBlock(LoopBlock);

  llvm::Value** IndexPtr;

  llvm::Value* numItems;
  
  switch(ET){
    case ForallMeshStmt::Vertices:
      IndexPtr = &VertexIndex;
      GetNumMeshItems(Dimensions, 0, &numItems, 0, 0);
      break;
    case ForallMeshStmt::Edges:
      IndexPtr = &EdgeIndex;
      GetNumMeshItems(Dimensions, 0, 0, &numItems, 0);
      break;
    case ForallMeshStmt::Faces:
      IndexPtr = &FaceIndex;
      GetNumMeshItems(Dimensions, 0, 0, 0, &numItems);
      break;
    case ForallMeshStmt::Cells:
    case ForallMeshStmt::Undefined:
      break;
  }
  
	*IndexPtr = InductionVar[3];
	EmitStmt(S.getBody());
	*IndexPtr = 0;

	llvm::Value* k = Builder.CreateLoad(InductionVar[3], "renderall.idx");
	k = Builder.CreateAdd(k, One);
	Builder.CreateStore(k, InductionVar[3]);
	k = Builder.CreateZExt(k, Int64Ty, "k");

	llvm::Value* Cond = Builder.CreateICmpSLT(k, numItems, "cond");

	llvm::BasicBlock *ExitBlock = createBasicBlock("renderall.exit");
	Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
	EmitBlock(ExitBlock);

	InnerIndex = 0;
}

//generate one of the nested loops
void CodeGenFunction::EmitRenderallMeshLoop(const RenderallMeshStmt &S, unsigned r) {
  RegionCounter Cnt = getPGORegionCounter(&S);
  (void)Cnt; //suppress warning

  llvm::StringRef MeshName = S.getMeshVarDecl()->getName();

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

  
