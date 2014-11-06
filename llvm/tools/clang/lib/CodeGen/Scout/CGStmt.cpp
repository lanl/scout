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
#include "Scout/ASTVisitors.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include "Scout/CGLegionTask.h"

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
    if(inLLDB()){
      llvm::Value* V = LocalDeclMap.lookup(MeshVarDecl);
      if(V){
        BaseAddr = V;
        return;
      }
      
      while(llvm::PointerType* PT =
            dyn_cast<llvm::PointerType>(BaseAddr->getType())){
        
        if(!PT->getElementType()->isPointerTy()){
          break;
        }
        
        BaseAddr = Builder.CreateLoad(BaseAddr);
      }
      
      LocalDeclMap[MeshVarDecl] = BaseAddr;
      return;
    }
    
    EmitGlobalMeshAllocaIfMissing(BaseAddr, *MeshVarDecl);
    
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

// find number of fields
unsigned int GetMeshNFields(const Stmt &S) {
  MeshDecl* MD;
  if (const ForallMeshStmt *FAMS = dyn_cast<ForallMeshStmt>(&S)) {
    MD = FAMS->getMeshType()->getDecl();
  }  else if (const RenderallMeshStmt *RAMS = dyn_cast<RenderallMeshStmt>(&S)) {
    MD =  RAMS->getMeshType()->getDecl();
  } else {
    assert(false && "non-mesh stmt in GetMeshNFields()");
  }
  return MD->fields();
}


// modifies meshDims, MeshDimsP1, LoopBoundsCells, Rank
void CodeGenFunction::SetMeshBounds(const Stmt &S) {
  llvm::Value *ConstantOne = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);

  //get mesh Base Addr
  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  llvm::StringRef MeshName = MeshBaseAddr->getName();

  // find number of fields
  unsigned int nfields = GetMeshNFields(S);

  // extract rank from mesh stored after width/height/depth
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  MeshRank = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+MeshParameterOffset::RankOffset, IRNameStr);

  // Extract width/height/depth from the mesh
  // note: width/height depth are stored after mesh fields
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    MeshDims[i] =  Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, nfields+i, IRNameStr);

    // dimensions + 1 are used in many places so cache them
    MeshDimsP1[i] =  Builder.CreateAlloca(Int32Ty, 0, "meshdimsp1.ptr");
    llvm::Value *incr = Builder.CreateAdd(Builder.CreateLoad(MeshDims[i]), ConstantOne);
    Builder.CreateStore(incr, MeshDimsP1[i]);

    // if LoopBoundCells == 0 then set it to 1 (for cells)
    LoopBoundsCells[i] = Builder.CreateAlloca(Int32Ty, 0, "meshdims.ptr");
    llvm::Value *dim = Builder.CreateLoad(MeshDims[i]);
    llvm::Value *Check = Builder.CreateICmpEQ(dim, ConstantZero);
    llvm::Value *x = Builder.CreateSelect(Check, ConstantOne, dim);
    Builder.CreateStore(x, LoopBoundsCells[i]);
  }
}



// generate code to return d1 if rank = 1, d2 if rank = 2, d3 if rank = 3;
llvm::Value *CodeGenFunction::GetNumLocalMeshItems(llvm::Value *d1, llvm::Value *d2, llvm::Value *d3) {

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  llvm::Value *check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
  llvm::Value *check2 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 2));
  return Builder.CreateSelect(check3, d3, Builder.CreateSelect(check2, d2, d1));

}

void CodeGenFunction::EmitForallCellsVertices(const ForallMeshStmt &S){
  EmitMarkerBlock("forall.vertices.entry");

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  llvm::Value* vertexPosPtr = InnerInductionVar;
  VertexIndex = InnerIndex;

  Builder.CreateStore(Zero, vertexPosPtr);

  llvm::Value* width = Builder.CreateLoad(MeshDims[0], "width");
  llvm::Value* indVar =  Builder.CreateLoad(InductionVar[3], "indVar");

  llvm::Value* width1 = Builder.CreateLoad(MeshDimsP1[0], "widthp1");
  llvm::Value* height1 = Builder.CreateLoad(MeshDimsP1[1], "height1");
  llvm::Value* idvw = Builder.CreateUDiv(indVar, width, "idvw");
  llvm::Value* wh1 = Builder.CreateMul(width1, height1);
  llvm::Value *v1, *v2, *v3, *v4, *v5;
  llvm::Value *Rank = Builder.CreateLoad(MeshRank);

  llvm::Value *PNVert = GetNumLocalMeshItems(One, Three, Seven);

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.vertices.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  llvm::Value* vertexPos = Builder.CreateLoad(vertexPosPtr, "vertex.pos");

  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Merge = createBasicBlock("rank.merge");

  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, Three);
  Builder.CreateCondBr(Check3, Then3, Else3);

  // rank = 3
  EmitBlock(Then3);
  llvm::Value* i = Builder.CreateLoad(InductionVar[0], "i");
  llvm::Value* j = Builder.CreateLoad(InductionVar[1], "j");
  llvm::Value* k = Builder.CreateLoad(InductionVar[2], "k");
  v1 = Builder.CreateMul(j, width1);
  v2 = Builder.CreateMul(k, wh1);
  llvm::Value* xyz = Builder.CreateAdd(i, Builder.CreateAdd(v1, v2));

  llvm::Value* pd4 = Builder.CreateUDiv(vertexPos, Four);
  llvm::Value* pm4 = Builder.CreateURem(vertexPos, Four);
  llvm::Value* pd2 = Builder.CreateUDiv(pm4, Two);
  llvm::Value* pm2 = Builder.CreateURem(pm4, Two);

  v3 = Builder.CreateMul(pd2, width1);
  v4 = Builder.CreateMul(pd4, wh1);
  llvm::Value* newVertexIndex3 =
      Builder.CreateAdd(Builder.CreateAdd(xyz, v3),
          Builder.CreateAdd(pm2, v4));
  Builder.CreateBr(Merge);

  // rank != 3
  EmitBlock(Else3);

  // rank = 2
  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);
  EmitBlock(Then2);
  v1 = Builder.CreateUDiv(vertexPos, Two);
  v2 = Builder.CreateMul(v1, width1);
  v3 = Builder.CreateURem(vertexPos, Two);
  v4 = Builder.CreateAdd(v2, v3);
  v5 = Builder.CreateAdd(v4, indVar);
  llvm::Value* newVertexIndex2 = Builder.CreateAdd(v5, idvw, "vertex.index.new");
  Builder.CreateBr(Merge);

  // rank !=2 (rank = 1)
  EmitBlock(Else2);
  llvm::Value* newVertexIndex1 = Builder.CreateAdd(vertexPos, indVar, "vertex.index.new");
  Builder.CreateBr(Merge);

  // Merge Block
  EmitBlock(Merge);
  llvm::PHINode *PNVI = Builder.CreatePHI(Int32Ty, 3, "rank.phi");
  PNVI->addIncoming(newVertexIndex3, Then3);
  PNVI->addIncoming(newVertexIndex2, Then2);
  PNVI->addIncoming(newVertexIndex1, Else2);
  Builder.CreateStore(PNVI, VertexIndex);

  llvm::Value* newVertexPos = Builder.CreateAdd(vertexPos, One);
  Builder.CreateStore(newVertexPos, vertexPosPtr);

  EmitStmt(S.getBody());

  VertexIndex = 0;

  llvm::Value* Cond = Builder.CreateICmpSLT(vertexPos, PNVert, "cond");

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.vertices.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);
}

//SC_TODO: Vertices has "regular" boundary while Cells is circular.
void CodeGenFunction::EmitForallVerticesCells(const ForallMeshStmt &S){

  llvm::Value *cx1, *cx2, *vx2, *vx3, *cy1, *cy2, *vy2, *vy3;
  llvm::Value *vx1, *vy1;
  llvm::Value *x, *y, *z, *i, *j, *k;

  EmitMarkerBlock("forall.cells.entry");

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  llvm::Value* cellPosPtr = InnerInductionVar;
  CellIndex = InnerIndex;

  Builder.CreateStore(Zero, cellPosPtr);

  llvm::Value* width = Builder.CreateLoad(MeshDims[0], "width");
  llvm::Value* height = Builder.CreateLoad(MeshDims[1], "height");
  llvm::Value* depth = Builder.CreateLoad(MeshDims[2], "depth");

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);

  llvm::Value *PNCell = GetNumLocalMeshItems(One, Three, Seven);

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.cells.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  llvm::Value* cellPos = Builder.CreateLoad(cellPosPtr, "cell.pos");

  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Merge = createBasicBlock("rank.merge");

  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, Three);
  Builder.CreateCondBr(Check3, Then3, Else3);

  // rank = 3
  EmitBlock(Then3);
  llvm::Value* pd4 = Builder.CreateUDiv(cellPos, Four);
  llvm::Value* pm4 = Builder.CreateURem(cellPos, Four);
  llvm::Value* pd2 = Builder.CreateUDiv(pm4, Two);
  llvm::Value* pm2 = Builder.CreateURem(pm4, Two);

  i = Builder.CreateLoad(InductionVar[0], "i");
  x = Builder.CreateSub(i, pd2, "x");

  cx1 = Builder.CreateICmpSLT(x, Zero);
  cx2 = Builder.CreateICmpSGE(x, width);
  vx2 = Builder.CreateAdd(x, width);
  vx3 = Builder.CreateURem(x, width);
  x = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));

  j = Builder.CreateLoad(InductionVar[1], "j");
  y = Builder.CreateSub(j, pm2, "y");

  cy1 = Builder.CreateICmpSLT(y, Zero);
  cy2 = Builder.CreateICmpSGE(y, height);
  vy2 = Builder.CreateAdd(y, height);
  vy3 = Builder.CreateURem(y, height);
  y = Builder.CreateSelect(cy1, vy2, Builder.CreateSelect(cy2, vy3, y));

  k = Builder.CreateLoad(InductionVar[2], "k");
  z = Builder.CreateSub(k, pd4, "z");

  llvm::Value* cz1 = Builder.CreateICmpSLT(z, Zero);
  llvm::Value* cz2 = Builder.CreateICmpSGE(z, depth);
  llvm::Value* vz2 = Builder.CreateAdd(z, depth);
  llvm::Value* vz3 = Builder.CreateURem(z, depth);
  z = Builder.CreateSelect(cz1, vz2, Builder.CreateSelect(cz2, vz3, z));

  llvm::Value* v1 = Builder.CreateMul(y, width);
  llvm::Value* v2 = Builder.CreateMul(z, Builder.CreateMul(width, height));

  llvm::Value* newCellIndex3 = Builder.CreateAdd(x, Builder.CreateAdd(v1, v2));
  Builder.CreateBr(Merge);

  // rank != 3
  EmitBlock(Else3);

  // rank = 2
  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);
  EmitBlock(Then2);

  i = Builder.CreateLoad(InductionVar[0], "i");

  vx1 = Builder.CreateUDiv(cellPos, Two);
  x = Builder.CreateSub(i, vx1, "x");

  cx1 = Builder.CreateICmpSLT(x, Zero);
  cx2 = Builder.CreateICmpSGE(x, width);
  vx2 = Builder.CreateAdd(x, width);
  vx3 = Builder.CreateURem(x, width);
  x = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));

  j = Builder.CreateLoad(InductionVar[1], "j");
  vy1 = Builder.CreateURem(cellPos, Two);
  y = Builder.CreateSub(j, vy1, "y");

  cy1 = Builder.CreateICmpSLT(y, Zero);
  cy2 = Builder.CreateICmpSGE(y, height);
  vy2 = Builder.CreateAdd(y, height);
  vy3 = Builder.CreateURem(y, height);
  y = Builder.CreateSelect(cy1, vy2, Builder.CreateSelect(cy2, vy3, y));

  llvm::Value* newCellIndex2 = Builder.CreateAdd(Builder.CreateMul(y, width), x);
  Builder.CreateBr(Merge);

  // rank !=2 (rank = 1)
  EmitBlock(Else2);

  i = Builder.CreateLoad(InductionVar[0], "i");
  vx1 = Builder.CreateURem(cellPos, Two);
  x = Builder.CreateSub(i, vx1, "x");

  cx1 = Builder.CreateICmpSLT(x, Zero);
  cx2 = Builder.CreateICmpSGE(x, width);
  vx2 = Builder.CreateAdd(x, width);
  vx3 = Builder.CreateURem(x, width);
  llvm::Value* newCellIndex1 = Builder.CreateSelect(cx1, vx2, Builder.CreateSelect(cx2, vx3, x));
  Builder.CreateBr(Merge);
  Else2 = Builder.GetInsertBlock();

  // Merge Block
  EmitBlock(Merge);
  llvm::PHINode *PNVI = Builder.CreatePHI(Int32Ty, 3, "rank.phi");
  PNVI->addIncoming(newCellIndex3, Then3);
  PNVI->addIncoming(newCellIndex2, Then2);
  PNVI->addIncoming(newCellIndex1, Else2);
  Builder.CreateStore(PNVI, CellIndex);

  llvm::Value* newCellPos = Builder.CreateAdd(cellPos, One);
  Builder.CreateStore(newCellPos, cellPosPtr);

  EmitStmt(S.getBody());

  CellIndex = 0;

  llvm::Value* Cond = Builder.CreateICmpSLT(cellPos, PNCell, "cond");

  llvm::BasicBlock *ExitBlock = createBasicBlock("forall.cells.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);
  return;
}

void CodeGenFunction::EmitForallCellsEdges(const ForallMeshStmt &S){

  EmitMarkerBlock("forall.edges.entry");

  llvm::Value *width, *height, *width1, *height1, *w1h, *i, *j, *k ,*a, *b, *c;
  llvm::Value *edgePos, *Cond, *x, *y, *z, *c1, *c2, *e1, *e2, *v1;
  llvm::Value *newEdgePos, *newEdgeIndex, *edgePosPtr;

  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
  llvm::Value* Four = llvm::ConstantInt::get(Int32Ty, 4);
  llvm::Value* Seven = llvm::ConstantInt::get(Int32Ty, 7);

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);

  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Done = createBasicBlock("rank.done");

  //rank = 3
  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, Three);
  Builder.CreateCondBr(Check3, Then3, Else3);
  EmitBlock(Then3);

  EdgeIndex = InnerIndex;
  edgePosPtr = InnerInductionVar;
  Builder.CreateStore(Zero, edgePosPtr);

  width = Builder.CreateLoad(MeshDims[0], "width");
  width1 = Builder.CreateLoad(MeshDimsP1[0], "width1");
  height = Builder.CreateLoad(MeshDims[1], "height");
  height1 = Builder.CreateLoad(MeshDimsP1[1], "height1");

  llvm::Value* depth1 = Builder.CreateLoad(MeshDimsP1[2], "depth1");

  w1h = Builder.CreateMul(width1, height, "w1h");
  llvm::Value* h1w = Builder.CreateMul(height1, width, "h1w");

  c = Builder.CreateAdd(h1w, w1h);
  a = Builder.CreateMul(depth1, c);
  b = Builder.CreateMul(width1, height1);

  i = Builder.CreateLoad(InductionVar[0]);
  j = Builder.CreateLoad(InductionVar[1]);
  k = Builder.CreateLoad(InductionVar[2]);

  llvm::BasicBlock *LoopBlock = createBasicBlock("forall.edges.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);

  edgePos = Builder.CreateLoad(edgePosPtr, "edge.pos");

  llvm::Value* pm4 = Builder.CreateURem(edgePos, Four);
  llvm::Value* pm2 = Builder.CreateURem(edgePos, Two);
  c1 = Builder.CreateICmpEQ(pm4, Two);
  c2 = Builder.CreateICmpEQ(pm4, Three);
  llvm::Value* c3 = Builder.CreateICmpUGT(edgePos, Three);
  x = Builder.CreateSelect(c1, Builder.CreateAdd(i, One), i);
  y = Builder.CreateSelect(c2, Builder.CreateAdd(j, One), j);
  z = Builder.CreateSelect(c3, Builder.CreateAdd(k, One), k);

  e1 = Builder.CreateAdd(x, Builder.CreateAdd(w1h, Builder.CreateMul(y, width)));
  e2 = Builder.CreateAdd(x, Builder.CreateMul(y, width1));
  llvm::Value* c4 = Builder.CreateICmpEQ(pm2, One);

  newEdgeIndex =
      Builder.CreateAdd(Builder.CreateMul(z, c), Builder.CreateSelect(c4, e1, e2));

  Builder.CreateStore(newEdgeIndex, EdgeIndex);

  newEdgePos = Builder.CreateAdd(edgePos, One);
  Builder.CreateStore(newEdgePos, edgePosPtr);

  EmitStmt(S.getBody());

  Cond = Builder.CreateICmpSLT(edgePos, Seven, "cond");

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

  v1 = Builder.CreateAdd(a, Builder.CreateMul(b, k));
  llvm::Value* v2 = Builder.CreateAdd(x, Builder.CreateMul(width1, y));

  newEdgeIndex = Builder.CreateAdd(v1, v2);

  Builder.CreateStore(newEdgeIndex, EdgeIndex);

  newEdgePos = Builder.CreateAdd(edgePos, One);
  Builder.CreateStore(newEdgePos, edgePosPtr);

  EmitStmt(S.getBody());

  Cond = Builder.CreateICmpSLT(edgePos, Three, "cond");

  llvm::BasicBlock *ExitBlock2 = createBasicBlock("forall.edges.exit2");
  Builder.CreateCondBr(Cond, LoopBlock2, ExitBlock2);
  EmitBlock(ExitBlock2);

  EdgeIndex = 0;
  Builder.CreateBr(Done);

  // rank !=3
  EmitBlock(Else3);

  // rank = 2
  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);
  EmitBlock(Then2);

  EdgeIndex = InnerIndex;
  edgePosPtr = InnerInductionVar;
  Builder.CreateStore(Zero, edgePosPtr);

  width = Builder.CreateLoad(MeshDims[0], "width");
  height = Builder.CreateLoad(MeshDims[1], "height");
  width1 =  Builder.CreateLoad(MeshDimsP1[0], "width1");
  w1h = Builder.CreateMul(width1, height, "w1h");
  i = Builder.CreateLoad(InductionVar[0], "i");
  j = Builder.CreateLoad(InductionVar[1], "j");

  llvm::BasicBlock *LoopBlock3 = createBasicBlock("forall.edges.loop");
  Builder.CreateBr(LoopBlock3);

  EmitBlock(LoopBlock3);

  edgePos = Builder.CreateLoad(edgePosPtr, "edge.pos");

  c1 = Builder.CreateICmpEQ(edgePos, Two);
  c2 = Builder.CreateICmpEQ(edgePos, Three);

  x = Builder.CreateSelect(c1, Builder.CreateAdd(i, One), i);
  y = Builder.CreateSelect(c2, Builder.CreateAdd(j, One), j);

  v1 = Builder.CreateMul(y, width);
  e1 = Builder.CreateAdd(w1h, Builder.CreateAdd(v1, x));
  e2 = Builder.CreateAdd(Builder.CreateMul(y, width1), x);

  llvm::Value* h = Builder.CreateURem(edgePos, Two);

  newEdgeIndex =
      Builder.CreateSelect(Builder.CreateICmpEQ(h, One), e1, e2);

  Builder.CreateStore(newEdgeIndex, EdgeIndex);

  EmitStmt(S.getBody());

  newEdgePos = Builder.CreateAdd(edgePos, One);
  Builder.CreateStore(newEdgePos, edgePosPtr);

  Cond = Builder.CreateICmpSLT(edgePos, Three, "cond");

  llvm::BasicBlock *ExitBlock3 = createBasicBlock("forall.edges.exit");
  Builder.CreateCondBr(Cond, LoopBlock3, ExitBlock3);
  EmitBlock(ExitBlock3);
  EdgeIndex = 0;
  Builder.CreateBr(Done);

  // rank !=2 (rank = 1)
  EmitBlock(Else2);
  EdgeIndex = InnerIndex;
  Builder.CreateStore(Builder.CreateLoad(InductionVar[0]), EdgeIndex);
  EmitStmt(S.getBody());
  EdgeIndex = 0;
  Builder.CreateBr(Done);

  EmitBlock(Done);
}

void CodeGenFunction::EmitForallCellsFaces(const ForallMeshStmt &S){
  //SC_TODO: this will not work inside a function
  unsigned int rank = S.getMeshType()->rankOf();

  EmitMarkerBlock("forall.faces.entry");

  if(rank == 1){
    FaceIndex = InnerIndex;
    Builder.CreateStore(Builder.CreateLoad(InductionVar[0]), FaceIndex);
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
    llvm::Value* width = Builder.CreateLoad(MeshDims[0], "width");
    llvm::Value* height = Builder.CreateLoad(MeshDims[1], "height");
    llvm::Value* width1 =  Builder.CreateLoad(MeshDimsP1[0], "width1");
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
  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Done3 = createBasicBlock("done3.else");
  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
  Builder.CreateCondBr(Check3, Then3, Else3);

  EmitBlock(Then3);
  //SC_TODO: 3D case
  Builder.CreateBr(Done3);

  EmitBlock(Else3);
  EmitForallEdgesOrFacesCellsLowD(S, EdgeIndex);
  Builder.CreateBr(Done3);

  EmitBlock(Done3);
}

void
CodeGenFunction::EmitForallEdgesOrFacesCellsLowD(const ForallMeshStmt &S,
                                                 llvm::Value* OuterIndex){

  llvm::Value* Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);

  EmitMarkerBlock("forall.cells.entry");

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);

  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Done2 = createBasicBlock("done2.else");

  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);

  // rank 2
  EmitBlock(Then2);
  llvm::Value* w = Builder.CreateLoad(MeshDims[0], "w");
  w = Builder.CreateZExt(w, Int64Ty, "w");
  llvm::Value* w1 = Builder.CreateAdd(w, One, "w1");

  llvm::Value* h = Builder.CreateLoad(MeshDims[1], "h");
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
      Builder.CreateSelect(c1, Builder.CreateSelect(c5, Zero, y), y, "y2");

  cellIndex =
      Builder.CreateAdd(Builder.CreateMul(y2, w), x2, "cellIndex.2");

  Builder.CreateStore(Builder.CreateTrunc(cellIndex, Int32Ty), CellIndex);

  EmitStmt(S.getBody());

  CellIndex = 0;
  Builder.CreateBr(Done2);

  // rank 1
  EmitBlock(Else2);
  CellIndex = InnerIndex;
  Builder.CreateStore(Builder.CreateTrunc(Builder.CreateLoad(OuterIndex), Int32Ty), CellIndex);

  EmitStmt(S.getBody());
  CellIndex = 0;

  EmitBlock(Done2);

}

void CodeGenFunction::EmitForallFacesCells(const ForallMeshStmt &S){
  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
   llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
   llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
   llvm::BasicBlock *Done3 = createBasicBlock("done3.else");
   llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
   Builder.CreateCondBr(Check3, Then3, Else3);

   EmitBlock(Then3);
   //SC_TODO: 3D case
   Builder.CreateBr(Done3);

   EmitBlock(Else3);
   EmitForallEdgesOrFacesCellsLowD(S, FaceIndex);
   Builder.CreateBr(Done3);

   EmitBlock(Done3);
}

void
CodeGenFunction::EmitForallEdgesOrFacesVerticesLowD(const ForallMeshStmt &S,
                                                    llvm::Value* OuterIndex){

  llvm::Value *k, *vertexIndex;
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);
  llvm::Value* Two = llvm::ConstantInt::get(Int32Ty, 2);

  EmitMarkerBlock("forall.edges.entry");

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);

  llvm::BasicBlock *Then2 = createBasicBlock("rank2.then");
  llvm::BasicBlock *Else2 = createBasicBlock("rank2.else");
  llvm::BasicBlock *Done2 = createBasicBlock("done2.else");

  llvm::Value *Check2 = Builder.CreateICmpEQ(Rank, Two);
  Builder.CreateCondBr(Check2, Then2, Else2);

  // rank 2
  EmitBlock(Then2);

  llvm::Value* w = Builder.CreateLoad(MeshDims[0], "w");
  w = Builder.CreateZExt(w, Int64Ty, "w");
  llvm::Value* w1 = Builder.CreateAdd(w, One, "w1");

  llvm::Value* h = Builder.CreateLoad(MeshDims[1], "h");
  h = Builder.CreateZExt(h, Int64Ty, "h");

  llvm::Value* w1h = Builder.CreateMul(w1, h, "w1h");

  k = Builder.CreateLoad(OuterIndex, "k");
  k = Builder.CreateZExt(k, Int64Ty, "k");

  llvm::Value* c1 = Builder.CreateICmpUGE(k, w1h, "c1");
  llvm::Value* km = Builder.CreateSub(k, w1h, "km");

  llvm::Value* x1 =
      Builder.CreateSelect(c1, Builder.CreateURem(km, w),
          Builder.CreateURem(k, w1), "x1");

  llvm::Value* y1 =
      Builder.CreateSelect(c1, Builder.CreateUDiv(km, w),
          Builder.CreateUDiv(k, w1), "y1");

  vertexIndex =
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
  Builder.CreateBr(Done2);

  // rank 1
  EmitBlock(Else2);


  llvm::Value* One32 = llvm::ConstantInt::get(Int32Ty, 1);

  k = Builder.CreateLoad(OuterIndex, "k");
  k = Builder.CreateZExt(k, Int64Ty, "k");

  VertexIndex = InnerIndex;
  vertexIndex = Builder.CreateTrunc(k, Int32Ty);
  Builder.CreateStore(vertexIndex, VertexIndex);

  EmitStmt(S.getBody());

  vertexIndex = Builder.CreateAdd(vertexIndex, One32);
  Builder.CreateStore(vertexIndex, VertexIndex);

  EmitStmt(S.getBody());

  VertexIndex = 0;

  EmitBlock(Done2);
}

void CodeGenFunction::EmitForallEdgesVertices(const ForallMeshStmt &S){
  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Done3 = createBasicBlock("done3.else");
  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
  Builder.CreateCondBr(Check3, Then3, Else3);

  EmitBlock(Then3);
  //SC_TODO: 3D case
  Builder.CreateBr(Done3);

  EmitBlock(Else3);
  EmitForallEdgesOrFacesVerticesLowD(S, EdgeIndex);
  Builder.CreateBr(Done3);

  EmitBlock(Done3);
}

void CodeGenFunction::EmitForallFacesVertices(const ForallMeshStmt &S){
  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  llvm::BasicBlock *Then3 = createBasicBlock("rank3.then");
  llvm::BasicBlock *Else3 = createBasicBlock("rank3.else");
  llvm::BasicBlock *Done3 = createBasicBlock("done3.else");
  llvm::Value *Check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
  Builder.CreateCondBr(Check3, Then3, Else3);

  EmitBlock(Then3);
  //SC_TODO: 3D case
  Builder.CreateBr(Done3);

  EmitBlock(Else3);
  EmitForallEdgesOrFacesVerticesLowD(S, FaceIndex);
  Builder.CreateBr(Done3);

  EmitBlock(Done3);
}

void CodeGenFunction::EmitForallEdges(const ForallMeshStmt &S){
  if(isGPU()){
    EmitGPUForall(S, EdgeIndex);
    return;
  }
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

  EmitMarkerBlock("forall.edges.entry");

  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.edges_idx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(Zero, InductionVar[3]);
  InnerIndex = Builder.CreateAlloca(Int32Ty, 0, "forall.inneridx.ptr");

  // find number of edges
  llvm::Value* width = Builder.CreateLoad(MeshDims[0], "width");
  llvm::Value* width1 = Builder.CreateLoad(MeshDimsP1[0], "width1");
  llvm::Value* height = Builder.CreateLoad(MeshDims[1], "height");
  llvm::Value* height1 = Builder.CreateLoad(MeshDimsP1[1], "height1");
  llvm::Value* depth = Builder.CreateLoad(MeshDims[2], "depth");
  llvm::Value* depth1 = Builder.CreateLoad(MeshDimsP1[2], "depth1");
  llvm::Value *p1 = Builder.CreateMul(width, Builder.CreateMul(height1, depth1));
  llvm::Value *p2 = Builder.CreateMul(width1, Builder.CreateMul(height, depth1));
  llvm::Value *p3 = Builder.CreateMul(width1, Builder.CreateMul(height1, depth));
  llvm::Value* numEdges = Builder.CreateAdd(p1, Builder.CreateAdd(p2, p3), "numEdges32");
  numEdges = Builder.CreateZExt(numEdges, Int64Ty, "numEdges");

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
    EmitGPUForall(S, FaceIndex);
    return;
  }
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);

  EmitMarkerBlock("forall.faces.entry");

  InductionVar[3] = Builder.CreateAlloca(Int32Ty, 0, "forall.faces_idx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(Zero, InductionVar[3]);

  // find number of faces (if rank !=3 this is the same as the edges case)
  llvm::Value* width = Builder.CreateLoad(MeshDims[0], "width");
  llvm::Value* width1 = Builder.CreateLoad(MeshDimsP1[0], "width1");
  llvm::Value* height = Builder.CreateLoad(MeshDims[1], "height");
  llvm::Value* height1 = Builder.CreateLoad(MeshDimsP1[1], "height1");
  llvm::Value* depth = Builder.CreateLoad(MeshDims[2], "depth");
  llvm::Value* depth1 = Builder.CreateLoad(MeshDimsP1[2], "depth1");


  llvm::BasicBlock *Then = createBasicBlock("numfaces.then");
  llvm::BasicBlock *Else = createBasicBlock("numfaces.else");
  llvm::BasicBlock *Merge = createBasicBlock("numfaces.merge");


  llvm::Value *Check = Builder.CreateICmpEQ(Builder.CreateLoad(MeshRank), Three);
  Builder.CreateCondBr(Check, Then, Else);

  //then block (rank == 3 case)
  EmitBlock(Then);
  llvm::Value *p1 = Builder.CreateMul(width1, Builder.CreateMul(height, depth));
  llvm::Value *p2 = Builder.CreateMul(width, Builder.CreateMul(height1, depth));
  llvm::Value *p3 = Builder.CreateMul(width, Builder.CreateMul(height, depth1));
  llvm::Value *V1 = Builder.CreateAdd(p1, Builder.CreateAdd(p2, p3), "numfaces3");
  Builder.CreateBr(Merge);

  // else block (rank !=3 case)
  EmitBlock(Else);
  llvm::Value *V2 = Builder.CreateAdd(Builder.CreateMul(width, height1),
      Builder.CreateMul(width1, height), "numfaces12");
  Builder.CreateBr(Merge);

  //Merge Block
  EmitBlock(Merge);
  llvm::PHINode *PN = Builder.CreatePHI(Int32Ty, 2, "numfaces.phi");
  PN->addIncoming(V1, Then);
  PN->addIncoming(V2, Else);
  llvm::Value* numFaces = Builder.CreateZExt(PN, Int64Ty, "numFaces");


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

void CodeGenFunction::EmitGPUForall(const ForallMeshStmt& S, llvm::Value *&Index) {
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

  Index = GPUThreadId;
  EmitStmt(S.getBody());
  Index = 0;

  threadId = Builder.CreateAdd(threadId, GPUThreadInc);
  Builder.CreateStore(threadId, GPUThreadId);

  Builder.CreateBr(condBlock);

  EmitBlock(exitBlock);

  llvm::Function* f = ExtractRegion(entry, exitBlock, "ForallMeshFunction");

  AddScoutKernel(f, S);
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

  Builder.CreateLoad(LoopBoundsCells[0], "TheMesh.width");
  Builder.CreateLoad(LoopBoundsCells[1], "TheMesh.height");
  Builder.CreateLoad(LoopBoundsCells[2], "TheMesh.depth");
  
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

  CGLegionTask cGLegionTask(FD, TF, CGM, Builder, this);
  cGLegionTask.EmitLegionTask();
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

  ResetMeshBounds();
  SetMeshBounds(S);

  switch(FET) {
   case ForallMeshStmt::Cells:
   case ForallMeshStmt::Vertices:
     EmitForallCellsOrVertices(S);
     return;
   case ForallMeshStmt::Edges:
     EmitForallEdges(S);
     return;
   case ForallMeshStmt::Faces:
     EmitForallFaces(S);
     return;
   default:
     assert(false && "invalid forall case");
   }
}

// ----- EmitforallCellsOrVertices
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

void CodeGenFunction::EmitForallCellsOrVertices(const ForallMeshStmt &S) {

  ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();
  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);

  // Track down the mesh meta data. 
  EmitForallMeshMDBlock(S);

  if(isGPU()) {
    if(FET == ForallMeshStmt::Vertices) {
      EmitGPUForall(S, VertexIndex);
    } else if (FET == ForallMeshStmt::Cells) {
      EmitGPUForall(S, CellIndex);
    }
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

  EmitForallMeshLoop(S, 3);

  // reset MeshDims, Rank and induction var
  // so width/height etc can't be called after forall
  ResetMeshBounds();
}


//generate one of the nested loops
void CodeGenFunction::EmitForallMeshLoop(const ForallMeshStmt &S, unsigned r) {
 
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
  llvm::Value *LoopBound = 0;

  ForallMeshStmt::MeshElementType FET = S.getMeshElementRef();
  if(FET == ForallMeshStmt::Cells) {
    LoopBound  = Builder.CreateLoad(LoopBoundsCells[r-1], IRNameStr);
  } else if(FET == ForallMeshStmt::Vertices) {
    LoopBound  = Builder.CreateLoad(MeshDimsP1[r-1], IRNameStr);
    VertexIndex = InductionVar[3];
  } else {
    assert(false && "non cells/vertices loop in EmitForallMeshLoop()");
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


// reset Loopbounds, mesh dimensions, rank and induction var
void CodeGenFunction::ResetMeshBounds(void) {

    LoopBoundsCells.clear();
    MeshDims.clear();
    MeshDimsP1.clear();
    InductionVar.clear();
    for(unsigned int i = 0; i < 3; i++) {
       LoopBoundsCells.push_back(0);
       MeshDims.push_back(0);
       MeshDimsP1.push_back(0);
       InductionVar.push_back(0);
    }
    // create linear loop index as 4th element
    InductionVar.push_back(0);
    MeshRank = 0;
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

  //SC_TODO: is there a betterf way rather than using name?
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

  EmitForallArrayLoop(S, S.getDims());

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


  ResetMeshBounds();
  SetMeshBounds(S);

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
     Args[i] = Builder.CreateLoad(LoopBoundsCells[i], IRNameStr);
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
  ResetMeshBounds();

}

void CodeGenFunction::EmitRenderallVerticesEdgesFaces(const RenderallMeshStmt &S){
	llvm::Value* Zero = llvm::ConstantInt::get(Int32Ty, 0);
	llvm::Value* One = llvm::ConstantInt::get(Int32Ty, 1);

	EmitMarkerBlock("renderall.entry");

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

  llvm::Value* numItems = 0;
  
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
      IndexPtr = &CellIndex;
      break;
    case ForallMeshStmt::Undefined:
      assert(false && "Undefined MeshElementType");
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
  llvm::LoadInst *LoopBound  = Builder.CreateLoad(LoopBoundsCells[r-1], IRNameStr);


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

  
