

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

// scout - includes
#include <stdio.h>
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "clang/AST/Decl.h"
#include "CGBlocks.h"
#include "clang/Analysis/Analyses/Dominators.h"

using namespace clang;
using namespace CodeGen;


// scout - Scout Stmts
void CodeGenFunction::EmitForAllStmtWrapper(const ForAllStmt &S) {
  
  DEBUG_OUT("EmitForAllStmtWrapper");
  
  // Clear stale mesh elements.
  MeshMembers.clear();
  ScoutMeshSizes.clear();
  
  llvm::StringRef MeshName = S.getMesh()->getName();
  const MeshType *MT = S.getMeshType();
  MeshType::MeshDimensionVec dims = MT->dimensions();
  MeshDecl *MD = MT->getDecl();
    
  typedef std::map<std::string, bool> MeshFieldMap;
  MeshFieldMap meshFieldMap;

  MeshBaseAddr = GetMeshBaseAddr(S);

  // We use 'IRNameStr' to hold the names we emit for various items at
  // the IR level.  It is a bit tricky to always nail down the string
  // length for these -- should probably give this aspect some more
  // thought (it is easy to introduce leaks here given the code
  // structure).  The naming is key to making the IR code more
  // readable...
  char *IRNameStr = new char[MeshName.size() + 16];
  const char *DimNames[] = { "width", "height", "depth" };
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), DimNames[i]);
    llvm::Value* lval = Builder.CreateConstInBoundsGEP2_32(MeshBaseAddr, 0, i+1, IRNameStr);
    ScoutMeshSizes.push_back(lval);
  }
  delete []IRNameStr;
  
  typedef MeshDecl::mesh_field_iterator MeshFieldIterator;
  MeshFieldIterator it = MD->mesh_field_begin(), it_end = MD->mesh_field_end();
  
  for(unsigned i = 0; it != it_end; ++it, ++i) {
    MeshFieldDecl *MFD = dyn_cast<MeshFieldDecl>(*it);
    llvm::StringRef FieldName = MFD->getName();
    QualType Ty = MFD->getType();
    
    meshFieldMap[FieldName.str()] = true;

    if (! MFD->isImplicit()) {
      IRNameStr = new char[MeshName.size() + FieldName.size() + 16];
      sprintf(IRNameStr, "%s.%s.ptr", MeshName.str().c_str(), FieldName.str().c_str());
      llvm::Value *FieldPtr = Builder.CreateStructGEP(MeshBaseAddr, i+4, IRNameStr);
      FieldPtr = Builder.CreateLoad(FieldPtr);

      //insertMeshDump(addr);
      
      sprintf(IRNameStr, "%s.%s", MeshName.str().c_str(), FieldName.str().c_str());
      llvm::Value *FieldVar = Builder.CreateAlloca(FieldPtr->getType(), 0, IRNameStr);

      Builder.CreateStore(FieldPtr, FieldVar);
      MeshMembers[FieldName] = std::make_pair(Builder.CreateLoad(FieldVar) , Ty);
      MeshMembers[FieldName].first->setName(FieldVar->getName());

      delete []IRNameStr;
    }
  }

  // Acquire a local copy of colors buffer.
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
  EmitForAllStmt(S);

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
  if (isa<RenderAllStmt>(S))
    ForallFn->setName("uniRenderallCellsFn");
  else
    ForallFn->setName("uniForallCellsFn");

  if (isGPU()) {

    std::string name = ForallFn->getName().str();
    assert(name.find(".") == std::string::npos && "Illegal PTX identifier (function name).\n");

    // Add metadata for scout kernel function.
    llvm::NamedMDNode *ScoutMetadata;
    ScoutMetadata = CGM.getModule().getOrInsertNamedMetadata("scout.kernels");
    
    SmallVector<llvm::Value *, 4> KMD; // Kernel MetaData
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
        // even need it...  It should refect wether the current argument
        // is a signed value or not... 
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
	gs = llvm::ConstantDataArray::getString(getLLVMContext(),
						(*it).getName());
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
    for(unsigned i = 0, e = dims.size(); i < e; ++i) {
      args.push_back(TranslateExprToValue(S.getStart(i)));
      args.push_back(TranslateExprToValue(S.getEnd(i)));
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
    //insertMeshDump(MeshBaseAddr);
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

  CGBlockInfo blockInfo(S.getBlock()->getBlockDecl(), TheName.c_str());

  llvm::Value *BlockFn = EmitScoutBlockLiteral(S.getBlock(),
                                               blockInfo,
                                               ScoutMeshSizes,
                                               inputs);

  // Generate a function call to BlockFn.
  EmitScoutBlockFnCall(BlockFn, blockInfo,
                       ScoutMeshSizes, inputs);
  
  //insertMeshDump(MeshBaseAddr);
}

void CodeGenFunction::EmitForAllArrayStmt(const ForAllArrayStmt &S) {
  
  llvm::SmallVector< llvm::Value *, 3 > ranges;
  for(size_t i = 0; i < 3; ++i){
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

  /*
  typedef llvm::SetVector<llvm::Value *> ValueSet;
  ValueSet ce_inputs, ce_outputs;
  codeExtractor.findInputsOutputs(ce_inputs, ce_outputs);
  ValueSet::iterator vsit, vsend;
  
  llvm::errs() << "*** forall body inputs\n";  
  vsend = ce_inputs.end();
  for(vsit = ce_inputs.begin(); vsit != vsend; vsit++) {
    llvm::Value *v = *vsit;
    llvm::errs() << "\t" << v->getName().str() << "\n";
  }
  
  llvm::errs() << "*** forall body outputs\n";  
  vsend = ce_outputs.end();
  for(vsit = ce_outputs.begin(); vsit != vsend; vsit++) {
    llvm::Value *v = *vsit;
    llvm::errs() << "\t" << v->getName().str() << "\n";
  }  
  */

  llvm::Function *ForallArrayFn = codeExtractor.extractCodeRegion();
  
  ForallArrayFn->setName("forall_array");
    
  if(isSequential() || isGPU()){
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
    if(llvm::CallInst *call = dyn_cast< llvm::CallInst >(I)) {
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
  
  CGBlockInfo blockInfo(S.getBlock()->getBlockDecl(), TheName.c_str());
  
  CurrentForAllArrayStmt = &S;
  
  llvm::Value *BlockFn = EmitScoutBlockLiteral(S.getBlock(),
                                               blockInfo,
                                               ranges,
                                               inputs);
  
  CurrentForAllArrayStmt = 0;
  
  // Generate a function call to BlockFn.
  EmitScoutBlockFnCall(BlockFn, blockInfo, ranges, inputs);
}

bool CodeGenFunction::hasCalledFn(llvm::Function *Fn, llvm::StringRef name) {
  typedef llvm::Function::iterator BBIterator;
  BBIterator BB = Fn->begin(), BB_end = Fn->end();
  typedef llvm::BasicBlock::iterator InstIterator;
  for( ; BB != BB_end; ++BB) {
    InstIterator Inst = BB->begin(), Inst_end = BB->end();
    for( ; Inst != Inst_end; ++Inst) {
      if(isCalledFn(Inst, name)) return true;
    }
  }
  return false;
}

bool CodeGenFunction::isCalledFn(llvm::Instruction *Instn, llvm::StringRef name) {
  if(isa< llvm::CallInst >(Instn)) {
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
  (void)meshName; //supress warning

  // Get the number and size of the mesh's dimensions.
  const MeshType *MT = S.getMeshType();
  MeshType::MeshDimensionVec dims = MT->dimensions();

  typedef std::vector< unsigned > Vector;
  typedef Vector::iterator VecIterator;
  typedef Vector::reverse_iterator VecRevIterator;

  ForallTripCount = one;
  std::vector< llvm::Value * > start, end, diff;
  
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    //start.push_back(TranslateExprToValue(S.getStart(i)));
    //end.push_back(TranslateExprToValue(S.getEnd(i)));

    //diff.push_back(Builder.CreateSub(end.back(), start.back()));
    //ForallTripCount = Builder.CreateMul(ForallTripCount, diff.back());

    llvm::Value* msi;
    
    switch(i){
      case 0:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], "dim.x");
        break;
      case 1:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], "dim.y");
        break;
      case 2:
        msi = Builder.CreateLoad(ScoutMeshSizes[i], "dim.z");
        break;
      default:
        assert(false && "Dimension case not handled in EmitForAllStmt");
    }
    
    start.push_back(zero);
    end.push_back(msi);
    diff.push_back(msi);
    
    ForallTripCount = Builder.CreateMul(ForallTripCount, msi);
  }

  llvm::Value *indVar = Builder.CreateAlloca(Int32Ty, 0, name);
  if(isSequential() || isCPU())
    Builder.CreateStore(zero, indVar);

  ForallIndVar = indVar;

  // Clear the list of stale ScoutIdxVars.
  ScoutIdxVars.clear();

  // Initialize the index variables.
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    llvm::Value *lval;
    
    switch(i){
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

    Builder.CreateStore(start[i], lval);
    ScoutIdxVars.push_back(lval);
  }

  llvm::Value *lval;
  llvm::Value *cond;
  llvm::BasicBlock *CondBlock;
  if(isSequential() || isCPU()) {
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
  if(isSequential() || isCPU()) {
    ExitBlock = createBasicBlock("forall.end");
    Builder.SetInsertPoint(CondBlock);
    Builder.CreateCondBr(cond, ForallBody, ExitBlock);
  }

  // As long as the condition is true, iterate the loop.
  EmitBlock(ForallBody);
  Builder.SetInsertPoint(ForallBody);

  // Set each dimension's index variable from induction variable.
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    lval = getGlobalIdx();
    llvm::Value *val;
    if(i > 0) {
      if(i == 1)
        val = diff[i - 1];
      else
        val = Builder.CreateMul(diff[i-1], diff[i - 2]);
      lval = Builder.CreateUDiv(lval, val);
    }

    lval = Builder.CreateURem(lval, diff[i]);
    lval = Builder.CreateAdd(lval, start[i]);
    Builder.CreateStore(lval, ScoutIdxVars[i]);
  }

  // Generate the statements in the body of the forall.
  EmitStmt(S.getBody());

  if(isSequential() || isCPU()) {
    // Increment the induction variables.
    lval = getGlobalIdx();
    Builder.CreateStore(Builder.CreateAdd(lval, one), ForallIndVar);
    Builder.CreateBr(CondBlock);

    EmitBlock(ExitBlock);
  }
}

void CodeGenFunction::EmitRenderAllStmt(const RenderAllStmt &S) {
  /*
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
  */

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
  MeshType::MeshDimensionVec dims = MT->dimensions();
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
  typedef MeshDecl::mesh_field_iterator MeshFieldIterator;
  MeshFieldIterator it = MD->mesh_field_begin(), it_end = MD->mesh_field_end();
  
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
 /* 
  llvm::Function *delRendFunc = CGM.getModule().getFunction("__scrt_renderall_delete");
  
  if(!delRendFunc){
    
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
*/

  if (DI)
    DI->EmitLexicalBlockEnd(Builder, S.getRBracLoc());
  
}