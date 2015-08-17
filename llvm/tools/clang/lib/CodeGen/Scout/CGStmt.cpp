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
#include <limits>
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "CGBlocks.h"

#include "Scout/CGScoutRuntime.h"
#include "Scout/ASTVisitors.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include "Scout/CGLegionTask.h"
#include "Scout/CGPlotRuntime.h"

using namespace clang;
using namespace CodeGen;

static const char *DimNames[]   = { "width", "height", "depth" };
static const char *StartNames[]   = { "xstart", "ystart", "zstart" };
static const char *SizeNames[]   = { "xsize", "ysize", "zsize" };

static const char *IndexNames[] = { "x", "y", "z"};

static const uint8_t FIELD_CELL = 0;
static const uint8_t FIELD_VERTEX = 1;
static const uint8_t FIELD_EDGE = 2;
static const uint8_t FIELD_FACE = 3;

namespace{

const uint32_t FLAG_VAR_CONSTANT = 0x00000001;
const uint32_t FLAG_VAR_POSITION = 0x00000002;

const uint32_t nullVarId = std::numeric_limits<uint32_t>::max();
  
} // namespace
  
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

void CodeGenFunction::GetFrameBaseAddr(const VarDecl *FrameVarDecl, llvm::Value*& BaseAddr) {
  // is a global. SC_TODO why not MeshVarDecl->hasGlobalStorage()?
  if ((FrameVarDecl->hasLinkage() || FrameVarDecl->isStaticDataMember())
      && FrameVarDecl->getTLSKind() != VarDecl::TLS_Dynamic) {
    
    BaseAddr = CGM.GetAddrOfGlobalVar(FrameVarDecl);
    
    // If BaseAddr is an external global then it is assumed that we are within LLDB
    // and we need to load the Frame base address because it is passed as a global
    // reference.
    if(inLLDB()){
      llvm::Value* V = LocalDeclMap.lookup(FrameVarDecl);
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
      
      LocalDeclMap[FrameVarDecl] = BaseAddr;
      return;
    }
  } else {
    BaseAddr = LocalDeclMap[FrameVarDecl];
    BaseAddr = Builder.CreateLoad(BaseAddr);
    
    // If Frame ptr then load
    const Type *T = FrameVarDecl->getType().getTypePtr();
    if(T->isAnyPointerType() || T->isReferenceType()) {
      BaseAddr = Builder.CreateLoad(BaseAddr);
    }
  }
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


void CodeGenFunction::SetMeshBounds(const Stmt &S) {
  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  const MeshType* mt;

  if (const ForallMeshStmt *FAMS = dyn_cast<ForallMeshStmt>(&S)) {
    mt = FAMS->getMeshType();
    SetMeshBounds(FAMS->getMeshElementRef(), MeshBaseAddr, mt);
  }  else if (const RenderallMeshStmt *RAMS = dyn_cast<RenderallMeshStmt>(&S)) {
    mt = RAMS->getMeshType();
    SetMeshBounds(RAMS->getMeshElementRef(), MeshBaseAddr, mt, false);
  } else {
    assert(false && "non-mesh stmt in SetMeshBounds()");
  }
}


void CodeGenFunction::SetMeshBounds(MeshElementType type, llvm::Value* MeshBaseAddr, const MeshType* mt, bool isForAll) {
  SetMeshBoundsImpl(isForAll, type, MeshBaseAddr, mt);
}

// deal w/ differences in Renderall/Forall cases
void CodeGenFunction::SetMeshBoundsImpl(bool isForall, int meshType, llvm::Value* MeshBaseAddr, const MeshType* mt) {

  MeshTy = mt;

  // find number of mesh fields (struct fields - the fixed ones like width/height/depth)
  unsigned int nfields = mt->getDecl()->fields();

  llvm::Value *ConstantOne = llvm::ConstantInt::get(Int64Ty, 1);
  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);
  
  llvm::StringRef MeshName = MeshBaseAddr->getName();
  
  // extract rank from mesh stored after width/height/depth
  sprintf(IRNameStr, "%s.rank.ptr", MeshName.str().c_str());
  MeshRank =
  Builder.CreateConstInBoundsGEP2_32(0,
                                     MeshBaseAddr, 0,
                                     nfields + MeshParameterOffset::RankOffset,
                                     IRNameStr);
  
  unsigned start = nfields + MeshParameterOffset::WidthOffset;
  
  // Extract width/height/depth from the mesh
  // note: width/height depth are stored after mesh fields
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    MeshDims[i] =
    Builder.CreateConstInBoundsGEP2_32(0,
                                       MeshBaseAddr, 0, start + i, IRNameStr);

  }

  start =  nfields + MeshParameterOffset::XStartOffset;
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), StartNames[i]);
    MeshStart[i] =
        Builder.CreateConstInBoundsGEP2_32(0,
                                           MeshBaseAddr, 0, start + i, IRNameStr);
  }

  start =  nfields + MeshParameterOffset::XSizeOffset;
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), SizeNames[i]);
    MeshSize[i] =
        Builder.CreateConstInBoundsGEP2_32(0,
                                           MeshBaseAddr, 0, start + i, IRNameStr);
  }

  for(unsigned int i = 0; i < 3; i++) {

     if(isForall == true) { // forall
       if  (meshType == Cells) {
         // if LoopBound == 0 then set it to 1 (for cells)
         LoopBounds[i] = CreateTempAlloca(Int64Ty, "loopbound.ptr");
         llvm::Value *dim = Builder.CreateLoad(MeshSize[i]);
         llvm::Value *Check = Builder.CreateICmpEQ(dim, ConstantZero);
         llvm::Value *x = Builder.CreateSelect(Check, ConstantOne, dim);
         Builder.CreateStore(x, LoopBounds[i]);
       } else if  (meshType == Vertices) {
         LoopBounds[i] = CreateTempAlloca(Int64Ty, "loopbounds.ptr");
         llvm::Value *incr = Builder.CreateAdd(Builder.CreateLoad(MeshSize[i]), ConstantOne);
         Builder.CreateStore(incr, LoopBounds[i]);
       }
     } else { //renderall
       // for renderall want full mesh w/ cell based bounds
       LoopBounds[i] = CreateTempAlloca(Int64Ty, "loopbound.ptr");
       llvm::Value *dim = Builder.CreateLoad(MeshDims[i]);
       llvm::Value *Check = Builder.CreateICmpEQ(dim, ConstantZero);
       llvm::Value *x = Builder.CreateSelect(Check, ConstantOne, dim);
       Builder.CreateStore(x, LoopBounds[i]);
     }
   }
}

// generate code to return d1 if rank = 1, d2 if rank = 2, d3 if rank = 3;
llvm::Value *CodeGenFunction::GetNumLocalMeshItems(llvm::Value *d1, llvm::Value *d2, llvm::Value *d3) {

  llvm::Value *Rank = Builder.CreateLoad(MeshRank);
  llvm::Value *check3 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 3));
  llvm::Value *check2 = Builder.CreateICmpEQ(Rank, llvm::ConstantInt::get(Int32Ty, 2));
  return Builder.CreateSelect(check3, d3, Builder.CreateSelect(check2, d2, d1));

}


void CodeGenFunction::GetNumMeshItems(llvm::Value** numCells, llvm::Value** numVertices,
    llvm::Value** numEdges, llvm::Value** numFaces) {

  llvm::Value *ConstantOne = llvm::ConstantInt::get(Int32Ty, 1);
  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int32Ty, 0);

  llvm::Value *width, *height, *depth, *n;
  llvm::Value *width1, *height1, *depth1;
  llvm::Value *p1, *p2, *p3;

  // find number of cells
  if(numCells) {
    llvm::Value *w = Builder.CreateLoad(MeshDims[0]);
    llvm::Value *Checkw = Builder.CreateICmpEQ(w, ConstantZero);
    width = Builder.CreateSelect(Checkw, ConstantOne, w);

    llvm::Value *h = Builder.CreateLoad(MeshDims[1]);
    llvm::Value *Checkh = Builder.CreateICmpEQ(h, ConstantZero);
    height = Builder.CreateSelect(Checkh, ConstantOne, h);

    llvm::Value *d = Builder.CreateLoad(MeshDims[2]);
    llvm::Value *Checkd = Builder.CreateICmpEQ(d, ConstantZero);
    depth = Builder.CreateSelect(Checkd, ConstantOne, d);

    n = Builder.CreateMul(depth, Builder.CreateMul(height, width));
    *numCells = Builder.CreateZExt(n, Int64Ty, "numCells");
  }

  // find number of vertices
  if(numVertices) {
    width = Builder.CreateAdd(Builder.CreateLoad(MeshDims[0]), ConstantOne, "width");
    height = Builder.CreateAdd(Builder.CreateLoad(MeshDims[1]), ConstantOne, "height");
    depth = Builder.CreateAdd(Builder.CreateLoad(MeshDims[2]), ConstantOne, "depth");
    n = Builder.CreateMul(depth, Builder.CreateMul(height, width));
    *numVertices = Builder.CreateZExt(n, Int64Ty, "numVertices");
  }

  // find number of edges
  if(numEdges) {
    width = Builder.CreateLoad(MeshDims[0], "width");
    width1 = Builder.CreateAdd(width, ConstantOne, "width1");
    height = Builder.CreateLoad(MeshDims[1], "height");
    height1 = Builder.CreateAdd(height, ConstantOne, "height1");
    depth = Builder.CreateLoad(MeshDims[2], "depth");
    depth1 = Builder.CreateAdd(depth, ConstantOne, "depth1");
    p1 = Builder.CreateMul(width, Builder.CreateMul(height1, depth1));
    p2 = Builder.CreateMul(width1, Builder.CreateMul(height, depth1));
    p3 = Builder.CreateMul(width1, Builder.CreateMul(height1, depth));
    n = Builder.CreateAdd(p1, Builder.CreateAdd(p2, p3), "numEdges32");
    *numEdges = Builder.CreateZExt(n, Int64Ty, "numEdges");
  }

  // find number of faces
  if(numFaces) {
    llvm::Value* Three = llvm::ConstantInt::get(Int32Ty, 3);
    width = Builder.CreateLoad(MeshDims[0], "width");
    width1 = Builder.CreateAdd(width, ConstantOne,"width1");
    height = Builder.CreateLoad(MeshDims[1], "height");
    height1 = Builder.CreateAdd(height, ConstantOne,"height1");
    depth = Builder.CreateLoad(MeshDims[2], "depth");
    depth1 = Builder.CreateAdd(depth, ConstantOne, "depth1");

    llvm::BasicBlock *Then = createBasicBlock("numfaces.then");
    llvm::BasicBlock *Else = createBasicBlock("numfaces.else");
    llvm::BasicBlock *Merge = createBasicBlock("numfaces.merge");

    llvm::Value *Check = Builder.CreateICmpEQ(Builder.CreateLoad(MeshRank), Three);
    Builder.CreateCondBr(Check, Then, Else);

    //then block (rank == 3 case)
    EmitBlock(Then);
    p1 = Builder.CreateMul(width1, Builder.CreateMul(height, depth));
    p2 = Builder.CreateMul(width, Builder.CreateMul(height1, depth));
    p3 = Builder.CreateMul(width, Builder.CreateMul(height, depth1));
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
    *numFaces = Builder.CreateZExt(PN, Int64Ty, "numFaces");
  }
}

void CodeGenFunction::AddScoutKernel(llvm::Function* f,
                                     const ForallMeshStmt &S){
  llvm::NamedMDNode* kernels =
      CGM.getModule().getOrInsertNamedMetadata("scout.kernels");

  llvm::SmallVector<llvm::Metadata*, 3> kernelData;
  kernelData.push_back(llvm::ValueAsMetadata::get(f));
  
  const MeshType* mt = S.getMeshType();
  kernelData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                           mt->getName()));
  
  MeshDecl* md = mt->getDecl();
  
  llvm::SmallVector<llvm::Metadata*, 16> meshFields;
  
  for (MeshDecl::field_iterator itr = md->field_begin(),
       itrEnd = md->field_end(); itr != itrEnd; ++itr){
    MeshFieldDecl* fd = *itr;
    
    llvm::SmallVector<llvm::Metadata*, 3> fieldData;
    fieldData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                            fd->getName()));
    if(fd->isCellLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_CELL)));
    }
    else if(fd->isVertexLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_VERTEX)));
    }
    else if(fd->isEdgeLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_EDGE)));
    }
    else if(fd->isFaceLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_FACE)));
    }
    
    llvm::Metadata* fieldDataMD =
    llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(fieldData));
    
    meshFields.push_back(fieldDataMD);
  }
  
  llvm::Metadata* fieldsMD =
  llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(meshFields));
  
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

  MeshElementType FET = S.getMeshElementRef();

  Builder.CreateLoad(LoopBounds[0], "TheMesh.width");
  Builder.CreateLoad(LoopBounds[1], "TheMesh.height");
  Builder.CreateLoad(LoopBounds[2], "TheMesh.depth");

  llvm::Value* numItems;
  
  switch (FET){
    case Cells:
      GetNumMeshItems(Dimensions, &numItems, 0, 0, 0);
      break;
    case Vertices:
      GetNumMeshItems(Dimensions, 0, &numItems, 0, 0);
      break;
    case Edges:
      GetNumMeshItems(Dimensions, 0, 0, &numItems, 0);
      break;
    case Faces:
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

void CodeGenFunction::EmitForallMeshStmt(const ForallMeshStmt &S){
  using namespace std;
  using namespace llvm;
  
  if(isGPU()){
    SetMeshBounds(S);
    InductionVar[3] = Builder.CreateAlloca(Int64Ty);
    EmitGPUForall(S, InductionVar[3]);
    return;
  }
  
  typedef vector<Value*> ValueVec;
  
  auto& B = Builder;
  auto R = CGM.getScoutRuntime();
  
  bool top = ForallStack.empty();
  
  Value* topology;
  Value* meshPtr;
  Value* rank;
  
  if(top){
    SetMeshBounds(S);
    
    NestedForallVisitor visitor;
    visitor.VisitStmt(const_cast<ForallMeshStmt*>(&S));
    auto fs = visitor.forallStmts();
    
    size_t i = 0;
    MeshElementType aboveElementType;
    Value* aboveRank;
    
    for(const ForallMeshStmt* s : fs){
      ForallData data;
      
      const VarDecl* mvd;
      if(const ImplicitMeshParamDecl* ip =
         dyn_cast<ImplicitMeshParamDecl>(s->getMeshVarDecl())){
        mvd = ip->getBaseVarDecl();
      }
      else{
        mvd = s->getMeshVarDecl();
      }
      
      data.meshVarDecl = mvd;
      
      QualType t = data.meshVarDecl->getType();
      const MeshType* mt;
      
      for(;;){
        if(const PointerType* pt = dyn_cast<PointerType>(t)){
          t = pt->getPointeeType();
        }
        else if(const ReferenceType* rt = dyn_cast<ReferenceType>(t)){
          t = rt->getPointeeType();
        }
        else{
          mt = dyn_cast<MeshType>(t);
          assert(mt && "expected a mesh type");
          break;
        }
      }
      
      const MeshDecl* md = mt->getDecl();

      GetMeshBaseAddr(mvd, meshPtr);
      
      if(i == 0){
        topology =
        B.CreateStructGEP(nullptr, meshPtr,
                          md->fields() + MeshParameterOffset::TopologyOffset);
        topology = B.CreateLoad(topology, "topology.ptr");
        data.topology = topology;
      }
      
      rank =
      B.CreateStructGEP(nullptr, meshPtr,
                        md->fields() + MeshParameterOffset::RankOffset);
      rank = B.CreateLoad(rank);
            
      data.elementType = s->getMeshElementRef();
      data.indexPtr = B.CreateAlloca(Int64Ty, nullptr, "index.ptr");
   
      for(size_t j = 0; j < 3; ++j){
        data.inductionVar[j] = Builder.CreateAlloca(Int64Ty, nullptr, IRNameStr);
      }
      
      Value* topologyDim = GetMeshTopologyDim(rank, data.elementType);
      
      if(i > 0){
        Value* aboveTopologyDim = GetMeshTopologyDim(aboveRank, aboveElementType);
        
        ValueVec args =
        {topology, aboveTopologyDim, topologyDim};
        
        if(i > 1){
          data.fromIndicesPtr =
          B.CreateCall(R.MeshGetFromIndicesFunc(), args, "from.indices.ptr");
        }
        else{
          data.fromIndicesPtr = nullptr;
        }
        
        data.startToIndicesPtr =
        B.CreateCall(R.MeshGetToIndicesFunc(), args, "to.indices.ptr");
        
        data.toIndicesPtr =
        B.CreateAlloca(data.startToIndicesPtr->getType(),
                       nullptr, "to.indices.ptr.ptr");
        
        B.CreateStore(data.startToIndicesPtr, data.toIndicesPtr);
      }
      else{
        data.fromIndicesPtr = nullptr;
        data.toIndicesPtr = nullptr;
      }
      
      ForallStack.emplace_back(move(data));
    
      aboveElementType = data.elementType;
      aboveRank = rank;
      
      ++i;
    }
  }
  
  const VarDecl* mvd;
  if(const ImplicitMeshParamDecl* ip =
     dyn_cast<ImplicitMeshParamDecl>(S.getMeshVarDecl())){
    mvd = ip->getBaseVarDecl();
  }
  else{
    mvd = S.getMeshVarDecl();
  }
  
  int i = FindForallData(S.getMeshElementRef());

  ForallStackIndex = i; // stash current state
  assert(i >= 0 && "error finding forall data");
  
  ForallData& topData = ForallStack[0];
  topology = topData.topology;
  
  ForallData& data = ForallStack[i];
  
  data.entryBlock = createBasicBlock("forall.entry");
  B.CreateBr(data.entryBlock);
  EmitBlock(data.entryBlock);
  
  if(data.fromIndicesPtr){
    ForallData& aboveData = ForallStack[i - 1];
    
    Value* fromIndex = B.CreateLoad(aboveData.indexPtr, "from.index");
    Value* toPos = B.CreateGEP(data.fromIndicesPtr, fromIndex);
    toPos = B.CreateLoad(toPos, "to.pos");
    Value* ptr = B.CreateGEP(data.startToIndicesPtr, toPos, "to.indices");
    B.CreateStore(ptr, data.toIndicesPtr);
  }
  
  Value* endIndex;
  
  Value* queryMask = nullptr;
  
  if(top){
    Value* topologyDim = GetMeshTopologyDim(rank, data.elementType);
    ValueVec args = {topology, topologyDim};
    endIndex = B.CreateCall(R.MeshNumEntitiesFunc(), args, "end.index");
    
    if(S.getQueryVarDecl()){
      queryMask = EmitForallQueryCall(S, endIndex);
    }
    
    B.CreateStore(ConstantInt::get(Int64Ty, 0), data.indexPtr);
  }
  
  BasicBlock* loopBlock = createBasicBlock("forall.loop");
  
  BasicBlock* condBlock = nullptr;
  
  if(queryMask){
    condBlock = createBasicBlock("query.cond");
    EmitBlock(condBlock);
    
    Value* index = B.CreateLoad(data.indexPtr, "index");
    
    Value* mask = B.CreateGEP(queryMask, index);
    mask = Builder.CreateLoad(mask);
    Value* cond = B.CreateICmpNE(mask, llvm::ConstantInt::get(Int8Ty, 0));
    BasicBlock* skipBlock = createBasicBlock("query.skip");
    B.CreateCondBr(cond, loopBlock, skipBlock);
    EmitBlock(skipBlock);
    index = B.CreateAdd(index, llvm::ConstantInt::get(Int64Ty, 1));
    B.CreateStore(index, data.indexPtr);
    B.CreateBr(condBlock);
  }
  else{
    B.CreateBr(loopBlock);
  }

  EmitBlock(loopBlock);
  
  Value* cond;
  
  if(top){
    EmitStmt(S.getBody());
    
    Value* index = B.CreateLoad(data.indexPtr, "index");
    Value* incIndex = B.CreateAdd(index, ConstantInt::get(Int64Ty, 1), "index.inc");
    B.CreateStore(incIndex, data.indexPtr);
    cond = B.CreateICmpSLT(incIndex, endIndex, "cond");
  }
  else{
    Value* toIndicesPtr = B.CreateLoad(data.toIndicesPtr, "to.indices.ptr");
    
    Value* rawIndex = B.CreateLoad(toIndicesPtr, "raw.index");
    Value* index = Builder.CreateAnd(rawIndex, ~(1UL << 63), "index");
    
    B.CreateStore(index, data.indexPtr);
    
    EmitStmt(S.getBody());
    
    toIndicesPtr = B.CreateConstGEP1_64(toIndicesPtr, 1, "next.to.indices.ptr");
    B.CreateStore(toIndicesPtr, data.toIndicesPtr);
    Value* done = B.CreateAnd(rawIndex, 1UL << 63, "done");
    cond = B.CreateICmpEQ(done, ConstantInt::get(Int64Ty, 0), "cond");
  }

  BasicBlock* exitBlock = createBasicBlock("forall.exit");
  
  if(condBlock){
    B.CreateCondBr(cond, condBlock, exitBlock);
  }
  else{
    B.CreateCondBr(cond, loopBlock, exitBlock);
  }

  EmitBlock(exitBlock);
  ForallStackIndex--; // update state.
  
  if(top){
    ForallStack.clear();
    ForallStackIndex = 0;
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

llvm::Value* CodeGenFunction::EmitForallQueryCall(const ForallMeshStmt& S,
                                                  llvm::Value* numItems){
  using namespace std;
  using namespace llvm;
  
  auto& B = Builder;
  
  typedef vector<llvm::Type*> TypeVec;
  typedef vector<Value*> ValueVec;
  
  VarDecl* qd = S.getQueryVarDecl();
 
  if(!qd){
    return 0;
  }
  
  const MeshType* cmt = S.getMeshType();
  
  StructType* mt =
  cast<StructType>(ConvertType(QualType(cmt, 0)));
  
  llvm::PointerType* mpt =
  llvm::PointerType::get(mt, 0);
  
  llvm::PointerType* outTy = llvm::PointerType::get(Int8Ty, 0);
  
  TypeVec params = {mpt, outTy, Int64Ty, Int64Ty};
  llvm::FunctionType* ft = llvm::FunctionType::get(VoidTy, params, false);
  
  Value* qp = LocalDeclMap[qd];
  
  Value* rawFuncPtr = B.CreateStructGEP(0, qp, 0);
  rawFuncPtr = B.CreateLoad(rawFuncPtr, "query.func.ptr");
  Value* funcPtr = B.CreateBitCast(rawFuncPtr, llvm::PointerType::get(ft, 0));
  
  Value* rawMeshPtr = B.CreateStructGEP(0, qp, 1, "query.mesh.ptr");
  rawMeshPtr = B.CreateLoad(rawMeshPtr);
  Value* meshPtr = B.CreateBitCast(rawMeshPtr, mpt);
  
  ValueVec args = {numItems};
  llvm::Function* allocFunc = CGM.getScoutRuntime().MemAllocFunction();
  Value* queryMask = B.CreateCall(allocFunc, args);
  queryMask = B.CreateBitCast(queryMask, outTy);
  
  Value* zero = llvm::ConstantInt::get(Int64Ty, 0);
  Value* one = llvm::ConstantInt::get(Int64Ty, 1);
  
  Value* end = B.CreateSub(numItems, one);
  ValueVec queryArgs = {meshPtr, queryMask, zero, end};
  B.CreateCall(funcPtr, queryArgs);
  
  return queryMask;
}

// reset Loopbounds, mesh dimensions, rank and induction var
void CodeGenFunction::ResetMeshBounds(void) {

    MeshTy = NULL;
    MeshDims.clear();
    MeshStart.clear();
    MeshSize.clear();
    LoopBounds.clear();
    for(unsigned int i = 0; i < 3; i++) {
       MeshDims.push_back(nullptr);
       MeshStart.push_back(nullptr);
       MeshSize.push_back(nullptr);
       LoopBounds.push_back(nullptr);
    }
    // create linear loop index as 4th element
    MeshRank = nullptr;
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
    if(cast<llvm::MDString>((*II)->getOperand(0))->getString() == MeshTypeName) {
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
  incrementProfileCounter(&S);
  
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
  const MeshType* mt = S.getMeshType();
  if (mt->dimensions().size() == 3) {
    EmitVolumeRenderallStmt(S);
    return;
  }
  
  llvm::Value *ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);

  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  llvm::StringRef MeshName = MeshBaseAddr->getName();

  // Add render target argument
  const VarDecl* RTVD = S.getRenderTargetVarDecl();

  llvm::Value* RTAlloc;
  
  if ((RTVD->hasLinkage() || RTVD->isStaticDataMember())
      && RTVD->getTLSKind() != VarDecl::TLS_Dynamic) {
    RTAlloc = CGM.GetAddrOfGlobalVar(RTVD);
  } else {
    RTAlloc = LocalDeclMap.lookup(RTVD);
  }

  // Check if it's a window or image type cuz we don't handle images
  // yet.
  const clang::Type &Ty = *getContext().getCanonicalType(RTVD->getType()).getTypePtr();
  
  llvm::SmallVector< llvm::Value *, 4 > Args;
  Args.clear();


  ResetMeshBounds();
  SetMeshBounds(S);

  // Create the induction variables for eack rank.
  for(unsigned int i = 0; i < 3; i++) {
    sprintf(IRNameStr, "renderall.induct.%s.ptr", IndexNames[i]);
    InductionVar[i] = CreateTempAlloca(Int64Ty, IRNameStr);
    //zero-initialize induction var
    Builder.CreateStore(ConstantZero, InductionVar[i]);
  }

  //build argument list for renderall setup runtime function
  for(unsigned int i = 0; i < 3; i++) {
    Args.push_back(0);
    sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), DimNames[i]);
    Args[i] =
    Builder.CreateTrunc(Builder.CreateLoad(LoopBounds[i], IRNameStr), Int32Ty);
  }
  
  if (Ty.getTypeClass() != Type::Window) {
    RTAlloc = Builder.CreateLoad(RTAlloc);
  }

  // cast scout.window_t** to void**
  llvm::Value* int8PtrPtrRTAlloc = Builder.CreateBitCast(RTAlloc, Int8PtrPtrTy, "");

  // dereference the void** 
  llvm::Value* int8PtrRTAlloc = Builder.CreateLoad(int8PtrPtrRTAlloc, "derefwin");

  // put the window on the arg list
  Args.push_back(int8PtrRTAlloc);

  // create linear loop index as 4th element and zero-initialize
  InductionVar[3] = CreateTempAlloca(Int64Ty, "renderall.linearidx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(ConstantZero, InductionVar[3]);

  MeshElementType ET = S.getMeshElementRef();

  // make quad renderable and add to the window and return color pointer
  // use same args as for RenderallUniformBeginFunction
  // in the future, this will be a mesh renderable

  llvm::Function *WinQuadRendFunc;
  llvm::Function *WinPaintFunc = CGM.getScoutRuntime().CreateWindowPaintFunction();
  
  bool cellLoop = false;
  
  switch(ET) {
    
    case Cells:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableColorsFunction();
      cellLoop = true;
      break;
      
    case Vertices:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableVertexColorsFunction();
      break;
      
    case Edges:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableEdgeColorsFunction();
      break;
      
    case Faces:
      WinQuadRendFunc = CGM.getScoutRuntime().CreateWindowQuadRenderableEdgeColorsFunction();
      break;
      
    default:
      assert(false && "unrecognized renderall type");
  }

  // %1 = call <4 x float>* @__scrt_window_quad_renderable_colors(i32 %HeatMeshType.width.ptr14, i32 %HeatMeshType.height.ptr16, i32 %HeatMeshType.depth.ptr18, i8* %derefwin)
  Color = Builder.CreateCall(WinQuadRendFunc, ArrayRef<llvm::Value *>(Args), "localcolor.ptr");

  if (cellLoop) {
    // renderall loops + body
    EmitRenderallMeshLoop(S, 3);
  } else {
    EmitRenderallVerticesEdgesFaces(S);
  }
  
  // paint window (draws all renderables) (does clear beforehand, and swap buffers after)
  if (S.isLast()) {
    Args.clear();
    Args.push_back(int8PtrRTAlloc);
    Builder.CreateCall(WinPaintFunc, ArrayRef<llvm::Value *>(Args));
  }

  // reset Loopbounds, Rank, induction var
  // so width/height etc can't be called after renderall
  ResetMeshBounds();
}

void CodeGenFunction::EmitVolumeRenderallStmt(const RenderallMeshStmt &S) {
  assert(false && "refactoring");
  /*
  using namespace llvm;
  using namespace std;
  
  typedef vector<Value*> ValueVec;
  typedef vector<llvm::Type*> TypeVec;
  
  auto& B = Builder;
  LLVMContext& C = getLLVMContext();
  auto R = CGM.getScoutRuntime();
  
  RenderallVisitor visitor(&S);
  visitor.VisitStmt(const_cast<Stmt*>(S.getBody()));
  
  auto& fs = visitor.getFieldSet();
  auto& vs = visitor.getVarSet();
  
  B.CreateGlobalStringPtr(CGM.getCodeGenOpts().ScoutPTXDir, "scout.ptx.dir");
  
  MeshDecl* MD = cast<MeshDecl>(S.getMeshType()->getDecl());
  
  Value* MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  
  StructType* meshStruct =
  dyn_cast<StructType>(MeshBaseAddr->getType()->getPointerElementType());
  
  Value* width =
  B.CreateLoad(B.CreateStructGEP(nullptr, MeshBaseAddr,
                                 meshStruct->getNumElements() - 1), "width");
  
  Value* height =
  B.CreateLoad(B.CreateStructGEP(nullptr, MeshBaseAddr,
                                 meshStruct->getNumElements() - 2), "height");
  
  Value* depth =
  B.CreateLoad(B.CreateStructGEP(nullptr, MeshBaseAddr,
                                 meshStruct->getNumElements() - 3), "depth");
  
  const VarDecl* target = S.getRenderTargetVarDecl();
  
  Value* targetPtr;
  
  if((target->hasLinkage() || target->isStaticDataMember())
     && target->getTLSKind() != VarDecl::TLS_Dynamic){
    targetPtr = Builder.CreateLoad(CGM.GetAddrOfGlobalVar(target));
  }
  else{
    targetPtr = LocalDeclMap.lookup(target);
  }
  
  assert(targetPtr);
  
  llvm::Type* ptrType = llvm::PointerType::get(VoidPtrTy, 0);
  targetPtr = Builder.CreateBitCast(targetPtr, ptrType);
  
  targetPtr = Builder.CreateLoad(targetPtr);
  
  BasicBlock* prevBlock = B.GetInsertBlock();
  BasicBlock::iterator prevPoint = B.GetInsertPoint();
  
  TypeVec params =
  {MeshBaseAddr->getType(), targetPtr->getType(),
    Int32Ty, Int32Ty, Int32Ty};
  
  for(VarDecl* vd : vs){
    Value* v = LocalDeclMap[vd];
    assert(v);
    params.push_back(v->getType());
  }
  
  llvm::Function* renderallFunc =
  llvm::Function::Create(llvm::FunctionType::get(VoidTy, params, false),
                         llvm::Function::ExternalLinkage,
                         "volume_renderall",
                         &CGM.getModule());
  
  auto aitr = renderallFunc->arg_begin();
  aitr->setName("mesh.ptr");
  aitr++;
  aitr->setName("window.ptr");
  aitr++;
  aitr->setName("width");
  aitr++;
  aitr->setName("height");
  aitr++;
  aitr->setName("depth");
  aitr++;
  
  ValueVec args = {MeshBaseAddr, targetPtr, width, height, depth};
  
  auto vitr = vs.begin();
  while(aitr != renderallFunc->arg_end()){
    aitr->setName((*vitr)->getName());
    Value* v = LocalDeclMap[*vitr];
    args.push_back(v);
    ++aitr;
  }
  
  B.CreateCall(renderallFunc, args);
  
  TypeVec fields;
  
  size_t index = 0;
  for(MeshFieldDecl* fd : fs){
    fields.push_back(llvm::PointerType::get(ConvertType(fd->getType()), 0));
    CurrentVolumeRenderallFieldMap.insert({fd, index++});
  }
  
  for(VarDecl* vd : vs){
    llvm::Type* t = ConvertType(vd->getType());
    
    //bool isScalar = false;
    if(t->isIntegerTy(32) ||
       t->isIntegerTy(64) ||
       t->isFloatTy() ||
       t->isDoubleTy()){
      //isScalar = true;
    }
    else{
      assert(false && "invalid var in volume renderall body");
    }
    
    fields.push_back(t);
  }
  
  StructType* meshType =
  StructType::create(C, fields, "mesh.struct");
  
  params = {VoidPtrTy, Int32Ty};
  
  llvm::FunctionType* transferFuncTy =
  llvm::FunctionType::get(llvm::VectorType::get(FloatTy, 4), params, false);
  
  llvm::Function* transferFunc =
  llvm::Function::Create(transferFuncTy,
                         llvm::Function::ExternalLinkage,
                         "volren_transfer",
                         &CGM.getModule());
  
  aitr = transferFunc->arg_begin();
  
  Value* mp = aitr;
  
  aitr->setName("mesh.void.ptr");
  aitr++;

  CurrentVolumeRenderallIndex = aitr;
  aitr->setName("index");
  aitr++;

  BasicBlock* entry = BasicBlock::Create(C, "entry", transferFunc);
  B.SetInsertPoint(entry);
  
  CurrentVolumeRenderallMeshPtr =
  B.CreateBitCast(mp, llvm::PointerType::get(meshType, 0));
  
  using RestoreMap = map<VarDecl*, Value*>;
  RestoreMap rm;
  
  size_t varOffset = fs.size();
  for(VarDecl* vd : vs){
    Value* v =
    B.CreateStructGEP(nullptr, CurrentVolumeRenderallMeshPtr,
                      varOffset, vd->getName());
    
    rm[vd] = LocalDeclMap[vd];
    LocalDeclMap[vd] = v;
    
    ++varOffset;
  }
  
  EmitStmt(S.getBody());
  
  for(auto& itr : rm){
    LocalDeclMap[itr.first] = itr.second;
  }
  
  Builder.CreateRet(B.CreateLoad(CurrentVolumeRenderallColor));
  
  CurrentVolumeRenderallMeshPtr = nullptr;
  CurrentVolumeRenderallIndex = nullptr;
  CurrentVolumeRenderallColor = nullptr;
  CurrentVolumeRenderallFieldMap.clear();
  
  //llvm::VectorType* Int3Ty = llvm::VectorType::get(Int32Ty, 3);
  
  params = {llvm::PointerType::get(Int32Ty, 0),
            llvm::PointerType::get(FloatTy, 0),
            Int32Ty, Int32Ty,
            Int32Ty, Int32Ty, Int32Ty, Int32Ty, Int32Ty, Int32Ty, FloatTy, FloatTy,
            FloatTy, FloatTy, VoidPtrTy};
  
  llvm::Function* wrapperFunc =
  llvm::Function::Create(llvm::FunctionType::get(VoidTy, params, false),
                         llvm::Function::ExternalLinkage,
                         "volume_render_wrapper",
                         &CGM.getModule());

  aitr = wrapperFunc->arg_begin();
  aitr->setName("output");
  aitr++;
  aitr->setName("invMat");
  aitr++;
  aitr->setName("imageW");
  aitr++;
  aitr->setName("imageH");
  aitr++;
  aitr->setName("startX");
  aitr++;
  aitr->setName("startY");
  aitr++;
  aitr->setName("startZ");
  aitr++;
  aitr->setName("width");
  aitr++;
  aitr->setName("height");
  aitr++;
  aitr->setName("depth");
  aitr++;
  aitr->setName("density");
  aitr++;
  aitr->setName("brightness");
  aitr++;
  aitr->setName("transferOffset");
  aitr++;
  aitr->setName("transferScale");
  aitr++;
  aitr->setName("meshPtr");
  aitr++;
   
  params.push_back(llvm::PointerType::get(transferFuncTy, 0));

  llvm::Function* renderFunc =
  llvm::Function::Create(llvm::FunctionType::get(VoidTy, params, false),
                         llvm::Function::ExternalLinkage,
                         "volume_render",
                         &CGM.getModule());
  
  aitr = renderFunc->arg_begin();
  aitr->setName("output");
  aitr++;
  aitr->setName("invMat");
  aitr++;
  aitr->setName("imageW");
  aitr++;
  aitr->setName("imageH");
  aitr++;
  aitr->setName("startX");
  aitr++;
  aitr->setName("startY");
  aitr++;
  aitr->setName("startZ");
  aitr++;
  aitr->setName("width");
  aitr++;
  aitr->setName("height");
  aitr++;
  aitr->setName("depth");
  aitr++;
  aitr->setName("density");
  aitr++;
  aitr->setName("brightness");
  aitr++;
  aitr->setName("transferOffset");
  aitr++;
  aitr->setName("transferScale");
  aitr++;
  aitr->setName("meshPtr");
  aitr++;
  aitr->setName("transferFunc");
  aitr++;
  
  entry = BasicBlock::Create(C, "entry", wrapperFunc);
  B.SetInsertPoint(entry);

  args.clear();
  aitr = wrapperFunc->arg_begin();
  while(aitr != wrapperFunc->arg_end()){
    args.push_back(aitr++);
  }
  
  args.push_back(transferFunc);
  
  B.CreateCall(renderFunc, args);
  B.CreateRetVoid();
  
  B.SetInsertPoint(prevBlock, prevPoint);
  
  llvm::NamedMDNode* volrens =
  CGM.getModule().getOrInsertNamedMetadata("scout.volren");
  
  llvm::SmallVector<llvm::Metadata*, 5> volrenData;
  volrenData.push_back(llvm::ValueAsMetadata::get(renderallFunc));
  volrenData.push_back(llvm::ValueAsMetadata::get(wrapperFunc));
  volrenData.push_back(llvm::ValueAsMetadata::get(transferFunc));
  
  const MeshType* mt = S.getMeshType();
  volrenData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                           mt->getName()));
  
  llvm::SmallVector<llvm::Metadata*, 16> meshFields;
  
  for(MeshFieldDecl* fd : fs){
    llvm::SmallVector<llvm::Metadata*, 3> fieldData;
    
    fieldData.push_back(llvm::MDString::get(CGM.getLLVMContext(),
                                            fd->getName()));
    
    bool found = false;
    uint32_t index = 0;
    for(auto fitr = MD->field_begin(), fitrEnd = MD->field_end();
        fitr != fitrEnd; ++fitr){
      if(*fitr == fd){
        fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, index)));
        found = true;
        break;
      }
      ++index;
    }
    assert(found);
    
    if(fd->isCellLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_CELL)));
    }
    else if(fd->isVertexLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_VERTEX)));
    }
    else if(fd->isEdgeLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_EDGE)));
    }
    else if(fd->isFaceLocated()){
      fieldData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int8Ty, FIELD_FACE)));
    }
    
    llvm::Metadata* fieldDataMD =
    llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(fieldData));
    
    meshFields.push_back(fieldDataMD);
  }
  
  llvm::Metadata* fieldsMD =
  llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(meshFields));
  
  volrenData.push_back(fieldsMD);
  
  llvm::SmallVector<llvm::Metadata*, 16> vars;
  for(VarDecl* vd : vs) {
    llvm::Type* t = ConvertType(vd->getType());
    
    size_t size = t->getPrimitiveSizeInBits() / 8;
    assert(size != 0 && "size of non-primitive type requested -- fix this to be flexible!");
    
    //
    //if (t->isIntegerTy(32) || t->isFloatTy()) {
    //  size = 4;
    //} else if(t->isIntegerTy(64) || t->isDoubleTy()) {
   //   size = 8;
   // } else {

   // }
    llvm::SmallVector<llvm::Metadata*, 1> varData;
    varData.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Int32Ty, size)));
    
    llvm::Metadata* varDataMD =
    llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(varData));
    
    vars.push_back(varDataMD);
  }
  
  llvm::Metadata* varsMD =
  llvm::MDNode::get(CGM.getLLVMContext(), ArrayRef<llvm::Metadata*>(vars));
  
  volrenData.push_back(varsMD);
  volrens->addOperand(llvm::MDNode::get(CGM.getLLVMContext(), volrenData));
*/
}

void CodeGenFunction::EmitRenderallVerticesEdgesFaces(const RenderallMeshStmt &S){
  auto R = CGM.getScoutRuntime();
  
  llvm::Value* Zero = llvm::ConstantInt::get(Int64Ty, 0);
  llvm::Value* One = llvm::ConstantInt::get(Int64Ty, 1);

  const MeshType* mt = S.getMeshType();
  
  const MeshDecl* md = mt->getDecl();
  
  llvm::Value *MeshBaseAddr;
  GetMeshBaseAddr(S, MeshBaseAddr);
  
  llvm::Value* topology = Builder.CreateStructGEP(nullptr, MeshBaseAddr, md->fields());
  topology = Builder.CreateLoad(topology, "topology.ptr");
  
  llvm::Value* rank =
  Builder.CreateStructGEP(nullptr, MeshBaseAddr,
                          md->fields() + MeshParameterOffset::RankOffset);
  

  llvm::Value* topologyDim = GetMeshTopologyDim(rank, S.getMeshElementRef());
  
  std::vector<llvm::Value*> args = {topology, topologyDim};
  llvm::Value* numItems = Builder.CreateCall(R.MeshNumEntitiesFunc(), args);
  
  EmitMarkerBlock("renderall.entry");

  InductionVar[3] = CreateTempAlloca(Int64Ty, "renderall.idx.ptr");
  //zero-initialize induction var
  Builder.CreateStore(Zero, InductionVar[3]);
  //InnerIndex = CreateTempAlloca(Int64Ty, "renderall.inneridx.ptr");

  llvm::BasicBlock *LoopBlock = createBasicBlock("renderall.loop");
  Builder.CreateBr(LoopBlock);

  EmitBlock(LoopBlock);
  
  EmitStmt(S.getBody());

  llvm::Value* k = Builder.CreateLoad(InductionVar[3], "renderall.idx");
  k = Builder.CreateAdd(k, One);
  Builder.CreateStore(k, InductionVar[3]);
  k = Builder.CreateZExt(k, Int64Ty, "k");

  llvm::Value* Cond = Builder.CreateICmpSLT(k, numItems, "cond");

  llvm::BasicBlock *ExitBlock = createBasicBlock("renderall.exit");
  Builder.CreateCondBr(Cond, LoopBlock, ExitBlock);
  EmitBlock(ExitBlock);

  //InnerIndex = 0;
}

//generate one of the nested loops
void CodeGenFunction::EmitRenderallMeshLoop(const RenderallMeshStmt &S, unsigned r) {
  llvm::StringRef MeshName = S.getMeshVarDecl()->getName();

  CGDebugInfo *DI = getDebugInfo();

  llvm::Value *ConstantZero  = 0;
  llvm::Value *ConstantOne   = 0;
  ConstantZero = llvm::ConstantInt::get(Int64Ty, 0);
  ConstantOne  = llvm::ConstantInt::get(Int64Ty, 1);

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
  llvm::Value *LoopBound  = Builder.CreateLoad(LoopBounds[r-1], IRNameStr);

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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
void CodeGenFunction::EmitScoutStmt(const ScoutStmt &S) {
  switch(S.kind()){
    case ScoutStmt::FrameCapture:
      EmitFrameCaptureStmt(static_cast<const FrameCaptureStmt&>(S));
      break;
    case ScoutStmt::Plot:
      EmitPlotStmt(static_cast<const PlotStmt&>(S));
      break;
    default:
      assert(false && "unhandled ScoutStmt");
  }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

void CodeGenFunction::EmitFrameCaptureStmt(const FrameCaptureStmt &S) {
  using namespace std;
  using namespace llvm;
  
  typedef vector<Value*> ValueVec;
  
  auto R = CGM.getPlotRuntime();
  
  const VarDecl* vd = S.getFrameVar();
  const FrameType* ft = dyn_cast<FrameType>(vd->getType().getTypePtr());
  const FrameDecl* fd = ft->getDecl();

  const SpecObjectExpr* spec = S.getSpec();
  
  Value* framePtr = LocalDeclMap.lookup(vd);
  assert(framePtr);
  framePtr = Builder.CreateLoad(framePtr, "frame.ptr");

  auto m = fd->getVarMap();
  auto mm = spec->memberMap();
  
  for(auto& itr : mm){
    const string& k = itr.first;
    
    auto mitr = m.find(k);
    assert(mitr != m.end());
    
    uint32_t varId = mitr->second.varId;
    VarDecl* varDecl = mitr->second.varDecl;
    
    llvm::Type* lt = ConvertType(QualType(varDecl->getType().getTypePtr(), 0));
    
    Value* val =
    EmitAnyExprToTemp(itr.second.second->toExpr()).getScalarVal();
    
    ValueVec args = {framePtr, Builder.getInt32(varId), val};
    
    if(lt->isIntegerTy(32)){
      Builder.CreateCall(R.FrameCaptureI32Func(), args);
    }
    else if(lt->isIntegerTy(64)){
      Builder.CreateCall(R.FrameCaptureI64Func(), args);
    }
    else if(lt->isFloatTy()){
      Builder.CreateCall(R.FrameCaptureFloatFunc(), args);
    }
    else if(lt->isDoubleTy()){
      Builder.CreateCall(R.FrameCaptureDoubleFunc(), args);
    }
    else if(lt->isPointerTy()){
      if(lt->getPointerElementType()->isIntegerTy(8)){
        Builder.CreateCall(R.FrameCaptureStringFunc(), args);
      }
      else{
        assert(false && "invalid field type");
      }
    }
    else{
      assert(false && "invalid field type");
    }
  }
}

llvm::Value* CodeGenFunction::EmitPlotExpr(const PlotStmt &S,
                                           llvm::Value* PlotPtr,
                                           SpecExpr* E,
                                           uint32_t flags,
                                           uint32_t useVarId){
  using namespace std;
  using namespace llvm;
  
  typedef vector<Value*> ValueVec;
  typedef vector<llvm::Type*> TypeVec;
  
  auto R = CGM.getPlotRuntime();
  
  const FrameDecl* FD = S.getFrameDecl();
  
  VarDecl* vd = E->getVar();
  if(vd){
    if(FD->hasVar(vd)){
      return ConstantInt::get(R.Int32Ty, FD->getVarId(vd));
    }
    
    uint32_t varId = S.getVarId(vd);
    if(varId != 0){
      return ConstantInt::get(R.Int32Ty, varId);
    }
    
    varId = S.getExtVarId(vd);
    if(varId != 0){
      return ConstantInt::get(R.Int32Ty, varId);
    }
  }
  
  if(CallExpr* c = dyn_cast_or_null<CallExpr>(E->toExpr())){
    uint32_t varId = S.getVarId(c);
    
    if(varId != 0){
      return ConstantInt::get(R.Int32Ty, varId);
    }
  }
  
  SpecArrayExpr* array = E->toArray();
  
  bool isConstant;
  if(array){
    isConstant = true;
    
    for(size_t i = 0; i < array->size(); ++i){
      PlotExprVisitor v(S);
      v.Visit(array->get(i)->toExpr());
    
      if(!v.isConstant()){
        isConstant = false;
        break;
      }
    }
  }
  else{
    PlotExprVisitor v(S);
    v.Visit(E->toExpr());
    isConstant = v.isConstant();
  }
  
  BasicBlock* prevBlock = Builder.GetInsertBlock();
  BasicBlock::iterator prevPoint = Builder.GetInsertPoint();
  
  llvm::Type* rt;
  
  if(array){
    llvm::Type* st = ConvertType(array->get(0)->toExpr()->getType());
    rt = llvm::VectorType::get(st, array->size());
  }
  else{
    rt = ConvertType(E->toExpr()->getType());
  }
  
  TypeVec params = {R.VoidPtrTy, R.Int64Ty};
  
  llvm::FunctionType* funcType;

  bool isVec = rt->isVectorTy();
  
  if(isVec){
    params.push_back(R.PointerTy(rt->getScalarType()));
    funcType = llvm::FunctionType::get(R.VoidTy, params, false);
  }
  else{
    funcType = llvm::FunctionType::get(rt, params, false);
  }
  
  llvm::Function* func =
  llvm::Function::Create(funcType,
                         llvm::Function::ExternalLinkage,
                         "plot.func",
                         &CGM.getModule());
  
  auto aitr = func->arg_begin();
  aitr->setName("plot.ptr");
  aitr++;
  aitr->setName("index");
  
  Value* vecPtr;
  
  if(isVec){
    aitr++;
    aitr->setName("vec.ptr");
    vecPtr = aitr;
  }
  
  BasicBlock* entry = BasicBlock::Create(CGM.getLLVMContext(), "entry", func);
  Builder.SetInsertPoint(entry);
  
  if(isVec){
    if(array){
      for(size_t i = 0; i < array->size(); ++i){
        Value* ai = EmitAnyExpr(array->get(i)->toExpr()).getScalarVal();
        Value* vi = Builder.CreateGEP(vecPtr, ConstantInt::get(R.Int32Ty, i));
        Builder.CreateStore(ai, vi);
      }
    }
    else{
      Value* val = EmitAnyExpr(E->toExpr()).getScalarVal();
      Builder.CreateStore(val, Builder.CreateBitCast(vecPtr, R.PointerTy(rt)));
    }
    
    Builder.CreateRetVoid();
  }
  else{
    Value* val = EmitAnyExpr(E->toExpr()).getScalarVal();
    Builder.CreateRet(val);
  }
    
  //func->dump();
  
  Builder.SetInsertPoint(prevBlock, prevPoint);
  
  Value* funcAlloc = Builder.CreateAlloca(func->getType());
  Builder.CreateStore(func, funcAlloc);
  Value* funcPtr = Builder.CreateBitCast(Builder.CreateLoad(funcAlloc), R.VoidPtrTy, "func.ptr");
  
  Value* vid =
  ConstantInt::get(R.Int32Ty, useVarId == 0 ? S.nextVarId() : useVarId);
  
  ValueVec args;

  uint32_t allFlags = flags;
  if(isConstant){
    allFlags |= FLAG_VAR_CONSTANT;
  }
  
  if(isVec){
    args =
    {PlotPtr, vid, funcPtr,
      ConstantInt::get(R.Int32Ty, rt->getVectorNumElements()),
      ConstantInt::get(R.Int32Ty, allFlags)};
  }
  else{
    args =
    {PlotPtr, vid, funcPtr, ConstantInt::get(R.Int32Ty, allFlags)};
  }
  
  if(rt->isIntegerTy(32)){
    Builder.CreateCall(R.PlotAddVarI32Func(), args);
  }
  else if(rt->isIntegerTy(64)){
    Builder.CreateCall(R.PlotAddVarI64Func(), args);
  }
  else if(rt->isFloatTy()){
    Builder.CreateCall(R.PlotAddVarFloatFunc(), args);
  }
  else if(rt->isDoubleTy()){
    Builder.CreateCall(R.PlotAddVarDoubleFunc(), args);
  }
  else if(isVec){
    llvm::Type* st = rt->getScalarType();
    
    if(st->isIntegerTy(32)){
      Builder.CreateCall(R.PlotAddVarI32VecFunc(), args);
    }
    else if(st->isIntegerTy(64)){
      Builder.CreateCall(R.PlotAddVarI64VecFunc(), args);
    }
    else if(st->isFloatTy()){
      Builder.CreateCall(R.PlotAddVarFloatVecFunc(), args);
    }
    else if(st->isDoubleTy()){
      Builder.CreateCall(R.PlotAddVarDoubleVecFunc(), args);
    }
    else{
      assert(false && "invalid plot var vector type");
    }
  }
  else{
    assert(false && "invalid plot var type");
  }
    
  return vid;
}

RValue CodeGenFunction::EmitPlotCall(const CallExpr* C){
  using namespace std;
  using namespace llvm;
  
  llvm::Function* func = Builder.GetInsertBlock()->getParent();
  
  auto aitr = func->arg_begin();
  
  Value* plotPtr = aitr++;
  Value* index = aitr++;
  
  typedef vector<Value*> ValueVec;
  
  auto R = CGM.getPlotRuntime();
  
  assert(CurrentPlotStmt);
  
  uint32_t varId = CurrentPlotStmt->getVarId(C);
  assert(varId != 0);
  
  const FunctionDecl* F = C->getDirectCallee();
  
  ValueVec args = {plotPtr, ConstantInt::get(R.Int32Ty, varId), index};
  
  llvm::Type* rt = ConvertType(F->getReturnType());
  
  Value* ret;
  
  if(rt->isIntegerTy(32)){
    ret = Builder.CreateCall(R.PlotGetI32Func(), args);
  }
  else if(rt->isIntegerTy(64)){
    ret = Builder.CreateCall(R.PlotGetI64Func(), args);
  }
  else if(rt->isFloatTy()){
    ret = Builder.CreateCall(R.PlotGetFloatFunc(), args);
  }
  else if(rt->isDoubleTy()){
    ret = Builder.CreateCall(R.PlotGetDoubleFunc(), args);
  }
  else{
    assert(false && "invalid frame var type");
  }
  
  return RValue::get(ret);
}

void CodeGenFunction::EmitPlotStmt(const PlotStmt &S) {
  using namespace std;
  using namespace llvm;
  
  typedef vector<Value*> ValueVec;
  //typedef vector<llvm::Type*> TypeVec;
  
  CurrentPlotStmt = &S;
  
  auto R = CGM.getPlotRuntime();

  ValueVec args = {ConstantInt::get(R.Int64Ty, uint64_t(&S))};
  
  Value* plotPtr = Builder.CreateCall(R.PlotGetFunc(), args, "plot.ptr");
  
  args = {plotPtr};
  Value* ready = Builder.CreateCall(R.PlotReadyFunc(), args, "ready");
  
  BasicBlock* mergeBlock = createBasicBlock("merge");
  BasicBlock* initBlock = createBasicBlock("init");
  
  Builder.CreateCondBr(ready, mergeBlock, initBlock);
  
  EmitBlock(initBlock);
  
  const VarDecl* frame = S.getFrameVar();
  
  const FrameDecl* fd = S.getFrameDecl();
    
  Value* framePtr;
  
  PlotVarsVisitor visitor(S);
  visitor.Visit(const_cast<SpecObjectExpr*>(S.getSpec()));
  
  if(frame){
    const FrameType* ft =
    dyn_cast<FrameType>(frame->getType().getNonReferenceType().getTypePtr());
    
    if(ft){
      GetFrameBaseAddr(frame, framePtr);
    }
    else{
      const MeshType* mt = dyn_cast<MeshType>(frame->getType().getTypePtr());
      assert(mt && "expected a frame or mesh");
      
      MeshDecl* MD = mt->getDecl();
      
      auto vs = visitor.getVarSet();
      
      map<string, VarDecl*> ns;
      for(VarDecl* vd : vs){
        ns.insert({vd->getName().str(), vd});
      }
      
      SmallVector<Value*, 3> Dimensions;
      GetMeshDimensions(mt, Dimensions);
      
      bool hasCells = false;
      bool hasVertices = false;
      bool hasEdges = false;
      bool hasFaces = false;
      
      for(MeshDecl::field_iterator fitr = MD->field_begin(),
          fitrEnd = MD->field_end(); fitr != fitrEnd; ++fitr){
        if(fitr->isCellLocated()){
          hasCells = true;
        }
        else if(fitr->isVertexLocated()){
          hasVertices = true;
        }
        else if(fitr->isEdgeLocated()){
          hasEdges = true;
        }
        else if(fitr->isFaceLocated()){
          hasFaces = true;
        }
        else{
          assert(false && "unrecognized element");
        }
      }
      
      llvm::Value* numCells = 0;
      llvm::Value* numVertices = 0;
      llvm::Value* numEdges = 0;
      llvm::Value* numFaces = 0;
      
      GetNumMeshItems(Dimensions,
                      hasCells ? &numCells : 0,
                      hasVertices ? &numVertices : 0,
                      hasEdges ? &numEdges : 0,
                      hasFaces ? &numFaces : 0);
      
      while(Dimensions.size() < 3){
        Dimensions.push_back(ConstantInt::get(R.Int64Ty, 0));
      }
      
      ValueVec args =
      {Builder.CreateTrunc(Dimensions[0], R.Int32Ty, "width"),
        Builder.CreateTrunc(Dimensions[1], R.Int32Ty, "height"),
        Builder.CreateTrunc(Dimensions[2], R.Int32Ty, "depth")};
      
      framePtr = Builder.CreateCall(R.CreateMeshFrameFunc(), args, "frame.ptr");
      
      for(MeshDecl::field_iterator fitr = MD->field_begin(),
          fitrEnd = MD->field_end(); fitr != fitrEnd; ++fitr){
        MeshFieldDecl* field = *fitr;
        
        auto itr = ns.find(field->getName().str());
        
        if(itr == ns.end()){
          continue;
        }
        
        llvm::Value* varId = ConstantInt::get(R.Int32Ty, fd->getVarId(itr->second));
        
        Value* meshPtr;
        
        GetMeshBaseAddr(frame, meshPtr);
        
        Value* fieldPtr = Builder.CreateStructGEP(0, meshPtr, field->getFieldIndex());
        fieldPtr = Builder.CreateLoad(fieldPtr);
        
        llvm::PointerType* pt = dyn_cast<llvm::PointerType>(fieldPtr->getType());
        
        fieldPtr = Builder.CreateBitCast(fieldPtr, R.VoidPtrTy, "field.ptr");
        
        llvm::Type* et = pt->getElementType();
        
        llvm::Value* fieldType;
        
        if(et->isIntegerTy(32)){
          fieldType = R.ElementInt32Val;
        }
        else if(et->isIntegerTy(64)){
          fieldType = R.ElementInt64Val;
        }
        else if(et->isFloatTy()){
          fieldType = R.ElementFloatVal;
        }
        else if(et->isDoubleTy()){
          fieldType = R.ElementDoubleVal;
        }
        else{
          assert(false && "unrecognized mesh field type");
        }
        
        llvm::Value* numElements = 0;
        
        if(field->isCellLocated()){
          numElements = numCells;
        }
        else if(field->isVertexLocated()){
          numElements = numVertices;
        }
        else if(field->isEdgeLocated()){
          numElements = numEdges;
        }
        else if(field->isFaceLocated()){
          numElements = numFaces;
        }
        
        ValueVec args = {framePtr, varId, fieldType, fieldPtr, numElements};
        
        Builder.CreateCall(R.FrameAddArrayVarFunc(), args);
      }
    }
  }
  else{
    framePtr = ConstantPointerNull::get(R.VoidPtrTy);
  }
  
  const VarDecl* target = S.getRenderTargetVar();
  
  Value* targetPtr;
  
  if((target->hasLinkage() || target->isStaticDataMember())
     && target->getTLSKind() != VarDecl::TLS_Dynamic){
    targetPtr = Builder.CreateLoad(CGM.GetAddrOfGlobalVar(target));
  }
  else{
    targetPtr = LocalDeclMap.lookup(target);
  }
  
  assert(targetPtr);
  
  llvm::Type* ptrType = R.PointerTy(VoidPtrTy);
  targetPtr = Builder.CreateBitCast(targetPtr, ptrType);
  
  targetPtr = Builder.CreateLoad(targetPtr);
  
  args = {plotPtr, framePtr, targetPtr};
  Builder.CreateCall(R.PlotInitFunc(), args);
  
  auto ev = visitor.getExtVarSet();
  
  for(VarDecl* vd : ev){
    llvm::Type* t = ConvertType(vd->getType());
    
    llvm::Value* type;
    if(t->isIntegerTy(32)){
      type = R.ElementInt32Val;
    }
    else if(t->isIntegerTy(64)){
      type = R.ElementInt64Val;
    }
    else if(t->isFloatTy()){
      type = R.ElementFloatVal;
    }
    else if(t->isDoubleTy()){
      type = R.ElementDoubleVal;
    }
    else{
      assert(false && "invalid external var type");
    }
    
    uint32_t vid = S.addExtVar(vd);
    
    args = {plotPtr, ConstantInt::get(R.Int32Ty, vid), type};
    Builder.CreateCall(R.PlotAddVarFunc(), args);
  }
  
  const SpecObjectExpr* spec = S.getSpec();
  
  auto cs = visitor.getCallSet();
  
  for(const CallExpr* c : cs){
    S.addCall(c);
  }
  
  auto m = spec->memberMap();
  
  auto vm = S.varMap();
  
  for(auto& itr : m){
    const string& k = itr.first;
    SpecExpr* v = itr.second.second;
   
    if(k == "var"){
      SpecObjectExpr* vo = v->toObject();
      
      auto& m = vo->memberMap();
      
      for(auto& itr : m){
        const string& k = itr.first;
        SpecExpr* e = itr.second.second;
        
        auto vitr = vm.find(k);
        assert(vitr != vm.end());
        EmitPlotExpr(S, plotPtr, e, 0, vitr->second.second);
      }
    }
  }
  
  auto cm = S.callMap();
  
  for(auto& itr : cm){
    const CallExpr* c = itr.first;
    uint32_t retVarId = itr.second;
    
    const FunctionDecl* func = c->getDirectCallee();
    DeclarationNameInfo nameInfo = func->getNameInfo();
    string name = nameInfo.getName().getAsString();

    //llvm::errs() << "hash for '" << name << "' = " <<
    //hash<string>()(name) << "\n";
    
    size_t aggId;
    if(name == "sum"){
      aggId = 0;
    }
    else if(name == "mean"){
      aggId = 1;
    }
    else if(name == "variance"){
      aggId = 2;
    }
    else{
      assert(false && "invalid aggregate function");
    }
    
    args = {plotPtr, ConstantInt::get(R.Int64Ty, aggId)};
    
    llvm::Type* rt = ConvertType(func->getReturnType());
    
    if(rt->isIntegerTy(32)){
      args.push_back(R.ElementInt32Val);
    }
    else if(rt->isIntegerTy(64)){
      args.push_back(R.ElementInt64Val);
    }
    else if(rt->isFloatTy()){
      args.push_back(R.ElementFloatVal);
    }
    else if(rt->isDoubleTy()){
      args.push_back(R.ElementDoubleVal);
    }
    else{
      assert(false && "invalid aggregate return type");
    }
    
    args.push_back(ConstantInt::get(R.Int32Ty, retVarId));
    
    Value* aggPtr = Builder.CreateCall(R.PlotAddAggregateFunc(), args, "agg.ptr");
    
    unsigned n = c->getNumArgs();
    for(unsigned i = 0; i < n; ++i){
      const Expr* ei = c->getArg(i);
      SpecValueExpr se(const_cast<Expr*>(ei));
      
      Value* argVarId = EmitPlotExpr(S, plotPtr, &se);
      args = {aggPtr, argVarId};
      Builder.CreateCall(R.AggregateAddVarFunc(), args);
    }
  }
  
  for(auto& itr : m){
    const string& k = itr.first;
    SpecExpr* v = itr.second.second;
    
    if(k == "lines" || k == "points" || k == "area" || k == "interval"){
      SpecObjectExpr* o = v->toObject();
      
      Value* cv = EmitPlotExpr(S, plotPtr, o->get("color"));
      Value* xy;
      
      if(k == "interval"){
        xy = ConstantInt::get(R.Int32Ty, S.nextVarId());
      }
      else{
        xy = EmitPlotExpr(S, plotPtr, o->get("position"), FLAG_VAR_POSITION);
      }
      
      if(k == "lines" || k == "points"){
        Value* sv = EmitPlotExpr(S, plotPtr, o->get("size"));
        
        Value* l;
        
        if(o->has("label")){
          l = EmitPlotExpr(S, plotPtr, o->get("label"));
        }
        else{
          l = ConstantInt::get(R.Int32Ty, nullVarId);
        }
        
        args = {plotPtr, xy, sv, cv, l};
        
        if(k == "lines"){
          Builder.CreateCall(R.PlotAddLinesFunc(), args);
        }
        else{
          Builder.CreateCall(R.PlotAddPointsFunc(), args);
        }
      }
      else if(k == "area"){
        args = {plotPtr, xy, cv};
        Builder.CreateCall(R.PlotAddAreaFunc(), args);
      }
      else{
        SpecObjectExpr* bo = o->get("position")->toObject();
        
        Value* vi = EmitPlotExpr(S, plotPtr, bo->get("bin"));
        Value* n = ConstantInt::get(R.Int32Ty, bo->get("n")->getInteger());
        
        args = {plotPtr, vi, xy, n};
        Builder.CreateCall(R.PlotAddBinsFunc(), args);
        
        args = {plotPtr, xy, cv};
        Builder.CreateCall(R.PlotAddIntervalFunc(), args);
      }
    }
    else if(k == "pie"){
      SpecObjectExpr* o = v->toObject();
      
      Value* xy = ConstantInt::get(R.Int32Ty, S.nextVarId());
      
      args = {plotPtr, xy};
      Value* propPtr = Builder.CreateCall(R.PlotAddProportionFunc(), args);
      
      SpecArrayExpr* pa = o->get("proportion")->toArray();
      
      for(size_t i = 0; i < pa->size(); ++i){
        Value* pv = EmitPlotExpr(S, plotPtr, pa->get(i));
        args = {propPtr, pv};
        Builder.CreateCall(R.PlotProportionAddVarFunc(), args);
      }
      
      Value* cv;
      SpecExpr* c = o->get("color");
      if(c){
        cv = EmitPlotExpr(S, plotPtr, c);
      }
      else{
        cv = ConstantInt::get(R.Int32Ty, 0);
      }
      
      args = {plotPtr, xy, cv};
      Builder.CreateCall(R.PlotAddPieFunc(), args);
    }
    else if(k == "line"){
      SpecObjectExpr* o = v->toObject();
      
      Value* xy1 = EmitPlotExpr(S, plotPtr, o->get("start"), FLAG_VAR_POSITION);
      Value* xy2 = EmitPlotExpr(S, plotPtr, o->get("end"), FLAG_VAR_POSITION);
      Value* sv = EmitPlotExpr(S, plotPtr, o->get("size"));
      Value* cv = EmitPlotExpr(S, plotPtr, o->get("color"));

      args = {plotPtr, xy1, xy2, sv, cv};
      Builder.CreateCall(R.PlotAddLineFunc(), args);
    }
    else if(k == "axis"){
      SpecObjectExpr* av = v->toObject();
      uint32_t dim = av->get("dim")->getInteger();
      string label = av->get("label")->getString();
      uint32_t major = av->get("major")->getInteger();
      uint32_t minor = av->get("minor")->getInteger();
      
      args =
      {plotPtr, ConstantInt::get(R.Int32Ty, dim),
        Builder.CreateGlobalStringPtr(label),
        ConstantInt::get(R.Int32Ty, major),
        ConstantInt::get(R.Int32Ty, minor)};
      Builder.CreateCall(R.PlotAddAxisFunc(), args);
    }
    else if(k == "antialiased"){
      args = {plotPtr, ConstantInt::get(R.Int1Ty, v->getBool())};
      Builder.CreateCall(R.PlotSetAntialiasedFunc(), args);
    }
    else if(k == "output"){
      args = {plotPtr, Builder.CreateGlobalStringPtr(v->getString())};
      Builder.CreateCall(R.PlotSetOutputFunc(), args);
    }
    else if(k == "range"){
      SpecObjectExpr* o = v->toObject();
      
      for(size_t i = 0; i < 2; ++i){
        bool x = i == 0;
        SpecExpr* d = x ? o->get("x") : o->get("y");
        SpecArrayExpr* a = d->toArray();
        Value* min = EmitAnyExpr(a->get(0)->toExpr()).getScalarVal();
        Value* max = EmitAnyExpr(a->get(1)->toExpr()).getScalarVal();
        args =
        {plotPtr, ConstantInt::get(R.Int1Ty, x), min, max};
        Builder.CreateCall(R.PlotSetRangeFunc(), args);
      }
    }
  }
  
  Builder.CreateBr(mergeBlock);
  EmitBlock(mergeBlock);
  
  auto em = S.extVarMap();
  for(auto& itr : em){
    const VarDecl* vd = itr.first;
    uint32_t vid = itr.second;
    
    Value* vp = LocalDeclMap[vd];
    assert(vp);
    
    Value* v = Builder.CreateLoad(vp);
    
    args = {plotPtr, ConstantInt::get(R.Int32Ty, vid), v};
    
    llvm::Type* t = ConvertType(vd->getType());
    
    if(t->isIntegerTy(32)){
      Builder.CreateCall(R.PlotCaptureI32Func(), args);
    }
    else if(t->isIntegerTy(64)){
      Builder.CreateCall(R.PlotCaptureI64Func(), args);
    }
    else if(t->isFloatTy()){
      Builder.CreateCall(R.PlotCaptureFloatFunc(), args);
    }
    else if(t->isDoubleTy()){
      Builder.CreateCall(R.PlotCaptureDoubleFunc(), args);
    }
    else{
      assert(false && "invalid external var type");
    }
  }
  
  args = {plotPtr};
  Builder.CreateCall(R.PlotRenderFunc(), args);
  
  if(inLLDB()){
    Builder.CreateCall(R.PlotRefreshFunc(), args);
  }
  
  CurrentPlotStmt = nullptr;
}
