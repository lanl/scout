/**
 * @file   PTXBackend.h
 * @date   08.08.2009
 * @author Helge Rhodin
 *
 *
 * Copyright (C) 2009, 2010 Saarland University
 *
 * This file is part of llvmptxbackend.
 *
 * llvmptxbackend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * llvmptxbackend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with llvmptxbackend.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef PTXBACKEND_H
#define PTXBACKEND_H

#include "PTXTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Config/config.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "PTXPasses.h"

using namespace llvm;


#define PTX_TEX "__ptx_tex"
#define PTX_CONST "__ptx_const"
#define PTX_LOCAL "__ptx_local"
#define PTX_SHARED "__ptx_shared"

extern cl::opt<unsigned> PTX_PTR_SIZE;

//namespace {
  class CBEMCAsmInfo : public MCAsmInfo {
  public:
    CBEMCAsmInfo() {
      GlobalPrefix = "";
      PrivateGlobalPrefix = "";
    }
  };

  /// PTXWriter - This class is the main chunk of code that converts an LLVM
  /// module to a C translation unit.
  class PTXWriter : public FunctionPass, public InstVisitor<PTXWriter>
  {
    formatted_raw_ostream &Out;
    IntrinsicLowering *IL;
    Mangler *Mang;
    DenseMap<const Value*, unsigned> AnonValueNumbers;
    LoopInfo *LI;
    const Module *TheModule;
    const MCAsmInfo* TAsm;
    const MCRegisterInfo *MRI;
    MCContext *TCtx;
    const TargetData* TD;
    std::map<const Value *, const Value *>& parentPointers;
    unsigned FPCounter;
    unsigned NextAnonValueNumber;

    llvm::StringSet<MallocAllocator> reservedKeywords;

  public:
    static char ID;

    explicit PTXWriter(formatted_raw_ostream &o, std::map<const Value *,
                       const Value *>& parentCompositePointer)
      : FunctionPass(ID), Out(o), IL(0), Mang(0), LI(0),
      TheModule(0), TAsm(0), MRI(0), TD(0), parentPointers(parentCompositePointer),
      NextAnonValueNumber(0)
    {
      FPCounter = 0;

      reservedKeywords.insert("abs");
      reservedKeywords.insert("add");
      reservedKeywords.insert("addc");
      reservedKeywords.insert("and");
      reservedKeywords.insert("atom");
      reservedKeywords.insert("bar");
      reservedKeywords.insert("bfe");
      reservedKeywords.insert("bfi");
      reservedKeywords.insert("bfind");
      reservedKeywords.insert("bra");
      reservedKeywords.insert("brev");
      reservedKeywords.insert("brkpt");
      reservedKeywords.insert("call");
      reservedKeywords.insert("clz");
      reservedKeywords.insert("cnot");
      reservedKeywords.insert("cos");
      reservedKeywords.insert("cvt");
      reservedKeywords.insert("cvta");
      reservedKeywords.insert("div");
      reservedKeywords.insert("ex2");
      reservedKeywords.insert("exit");
      reservedKeywords.insert("fma");
      reservedKeywords.insert("isspacep");
      reservedKeywords.insert("ld");
      reservedKeywords.insert("ldu");
      reservedKeywords.insert("lg2");
      reservedKeywords.insert("mad");
      reservedKeywords.insert("mad24");
      reservedKeywords.insert("max");
      reservedKeywords.insert("membar");
      reservedKeywords.insert("min");
      reservedKeywords.insert("mov");
      reservedKeywords.insert("mul");
      reservedKeywords.insert("mul24");
      reservedKeywords.insert("neg");
      reservedKeywords.insert("not");
      reservedKeywords.insert("or");
      reservedKeywords.insert("pmevent");
      reservedKeywords.insert("popc");
      reservedKeywords.insert("prefetch");
      reservedKeywords.insert("prefetchu");
      reservedKeywords.insert("prmt");
      reservedKeywords.insert("rcp");
      reservedKeywords.insert("red");
      reservedKeywords.insert("rem");
      reservedKeywords.insert("ret");
      reservedKeywords.insert("rsqrt");
      reservedKeywords.insert("sad");
      reservedKeywords.insert("selp");
      reservedKeywords.insert("set");
      reservedKeywords.insert("setp");
      reservedKeywords.insert("shl");
      reservedKeywords.insert("shr");
      reservedKeywords.insert("sin");
      reservedKeywords.insert("slct");
      reservedKeywords.insert("sqrt");
      reservedKeywords.insert("st");
      reservedKeywords.insert("sub");
      reservedKeywords.insert("subc");
      reservedKeywords.insert("suld");
      reservedKeywords.insert("sured");
      reservedKeywords.insert("sust");
      reservedKeywords.insert("suq");
      reservedKeywords.insert("tex");
      reservedKeywords.insert("txq");
      reservedKeywords.insert("trap");
      reservedKeywords.insert("vabsdiff");
      reservedKeywords.insert("vadd");
      reservedKeywords.insert("vmad");
      reservedKeywords.insert("vmax");
      reservedKeywords.insert("vmin");
      reservedKeywords.insert("vset");
      reservedKeywords.insert("vshl");
      reservedKeywords.insert("vshr");
      reservedKeywords.insert("vsub");
      reservedKeywords.insert("vote");
      reservedKeywords.insert("xor");
    }

    virtual const char *getPassName() const { return "PTX backend"; }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.setPreservesAll();
    }

    virtual bool doInitialization(Module &M);

    bool runOnFunction(Function &F) {
      LI = &getAnalysis<LoopInfo>();

      printFunction(F);
      return false;
    }

    virtual bool doFinalization(Module &M) {
      // Free memory...
      delete IL;
      delete TD;
      delete TAsm;
      delete MRI;
      delete TCtx;
      delete Mang;

      // Ugly hack to avoid leaking memory
      delete &parentPointers;
//       FPConstantMap.clear();
//       TypeNames.clear();
//       ByValParams.clear();
//      intrinsicPrototypesAlreadyGenerated.clear();
      return false;
    }

    std::string getTypeStr(Type *Ty,
                        bool isSigned = false,
                        bool IgnoreName = false,
                        const AttrListPtr &PAL = AttrListPtr());
    std::string getSimpleTypeStr(Type *Ty,
                              bool isSigned);

    void printStructReturnPointerFunctionType(formatted_raw_ostream &Out,
                                              const AttrListPtr &PAL,
                                              PointerType *Ty);

    std::string getOperandStr(const Value *Operand);

    static unsigned long getTypeBitSize(Type* Ty)
    { //TODO: use existing function (TD_>??)
      switch (Ty->getTypeID()) {
      case Type::VoidTyID:    assert(false && "void type?????");
      case Type::PointerTyID: return PTX_PTR_SIZE;
      case Type::IntegerTyID: return cast<IntegerType>(Ty)->getBitWidth();
      case Type::FloatTyID:   return 32UL;
      case Type::DoubleTyID:  return 64UL;
      case Type::VectorTyID:
      {
        const VectorType* VecTy = cast<VectorType>(Ty);
        return VecTy->getNumElements()
          * getTypeBitSize(VecTy->getElementType());
      }
//        case Type::ArrayTyID:
//       {
//         ArrayType* ArrTy = cast<ArrayType>(Ty);
//         unsigned int size = ArrTy->getNumElements()
//                        * getTypeBitSize(ArrTy->getElementType());
//       }
//       case Type::StructTyID:
//       {
//        StructType* StrTy = cast<StructType>(Ty);
//        unsigned int size = 0;
//        for(unsigned int subtype=0; subtype < StrTy->getNumElements();
//                  subtype++)
//        {
//          Type* elementType = StrTy->getElementType(subtype);
//          unsigned int align = getAlignmentByte(elementType);
//          size += 8 * getPadding(size*8, align);
//          size += getTypeBitSize(elementType);
//        }
//        return size;
//       }
      case Type::ArrayTyID:
      {
        ArrayType* ArrTy = cast<ArrayType>(Ty);
        Type* elementType = ArrTy->getElementType();
        unsigned long size_element = getTypeBitSize(elementType);
        unsigned long size = ArrTy->getNumElements() * size_element;
        unsigned long align = 8 * getAlignmentByte(elementType);
        /*
        if(size == 0)
          ArrTy->dump();
        assert(size!=0 && "no multiple of 8");
        */

        size += (ArrTy->getNumElements()-1) * getPadding(size_element, align);
        //-1 because the last element needs no "fillup"
        return size;
      }
      case Type::StructTyID:
      {
        StructType* StrTy = cast<StructType>(Ty);
        unsigned long size = 0;
        for(unsigned int subtype=0; subtype < StrTy->getNumElements();
            subtype++)
        {
          Type* elementType = StrTy->getElementType(subtype);
          unsigned long align = 8 * getAlignmentByte(elementType);
          size += getPadding(size, align);
          size += getTypeBitSize(elementType);
        }
        return size;
      }
      default:
        errs() << "Unknown type" <<  *Ty << "\n";
        abort();
      }
    }

    static unsigned long getTypeByteSize(Type* Ty)
    {
      unsigned long size_bit = getTypeBitSize(Ty);
      assert((size_bit%8==0) && "no multiple of 8");
      return size_bit/8;
    }

    static unsigned int getAlignmentByte(Type* Ty)
    {
      const unsigned int MAX_ALIGN = 8; //maximum size is 8 for doubles

      switch (Ty->getTypeID()) {
      case Type::VoidTyID:    assert(false && "void type?????");
      case Type::VectorTyID:
        return getAlignmentByte(cast<VectorType>(Ty)->getElementType());
      case Type::PointerTyID:
      case Type::IntegerTyID:
      case Type::FloatTyID:
      case Type::DoubleTyID:
        return getTypeBitSize(Ty)/8;
      case Type::ArrayTyID:
        return getAlignmentByte(cast<ArrayType>(Ty)->getElementType());
      case Type::StructTyID:
      {
        StructType* StrTy = cast<StructType>(Ty);
        unsigned int maxa = 0;
        for(unsigned int subtype=0; subtype < StrTy->getNumElements();
            subtype++)
        {
          maxa = std::max(getAlignmentByte(StrTy->getElementType(subtype)),
                          maxa);
          if(maxa==MAX_ALIGN)
            return maxa;
        }
        return maxa;
      }
      default:
        errs() << "Unknown type" <<  *Ty << "\n";
        abort();
      }
    }

    static unsigned int getPadding(unsigned int offset, unsigned int align)
    {
      //second % needed if offset == 0
      return (align - (offset % align)) % align;
    }

    std::string getSignedConstOperand(Instruction *Operand, unsigned int op);

  private :

    std::string getTmpValueName(Type *Ty, int index = 0)
    {
      std::stringstream name_tmp;
      name_tmp << "ptx_tmp"
               << getTypeStr(Ty)
               << '_' << index;
      std::string name = name_tmp.str();

      while(name.find(".") !=std::string::npos)
        name.replace(name.find("."),1,"_");
      return name;
    }

    bool hasSignedOperand(Instruction &I)
    {
        for(unsigned int op=0; op < I.getNumOperands(); op++)
        {
          if(isSignedOperand(I,op))
                return true;
        }
        return false;
    }

    //returns number of signed operands
    bool isSignedOperand(Instruction &I, unsigned int op_number)
    {
      switch (I.getOpcode())
      {
        default:
          return false;

        case Instruction::SDiv:
          //        case Instruction::LShr: logical shift => zero extension
        case Instruction::AShr: //arithmetic shift => sign extension
        case Instruction::SRem:
          return op_number<2;
        case Instruction::SExt:
        case Instruction::SIToFP:
          return op_number<1;
      }
    }

    bool isSignedDestinationInstruction(Instruction &I)
    {
      switch (I.getOpcode())
      {
        default:
          return false;

        case Instruction::SDiv:
        case Instruction::LShr:
        case Instruction::SRem:
          return true;
        case Instruction::SExt:
        case Instruction::FPToSI:
          return true;
      }
    }

    std::string InterpretASMConstraint(InlineAsm::ConstraintInfo& c);

    void lowerIntrinsics(Function &F);

    void printModule(Module *M);
    void printContainedStructs(Type *Ty, std::set<Type *> &);
    void printFunctionSignature(const Function *F, bool Prototype);

    void printFunction(Function &);
    void printEntryFunctionSignature(const Function *F, bool Prototype);
    void printFunctionArguments(const Function *F, bool Prototype,
                                bool entryFunction);
    void loadEntryFunctionParams(const Function *F);
    void printBasicBlock(BasicBlock *BB);

    void printCast(unsigned opcode, Type *SrcTy, Type *DstTy);
    std::string getConstant(const Constant *CPV, int dept);
    std::string getConstantCompositeAsArray(const Constant *C, int dept);
    void printConstantWithCast(const Constant *CPV, unsigned Opcode);
    bool printConstExprCast(const ConstantExpr *CE, bool Static);
    std::string getConstantArray(const ConstantArray *CPA, bool Static);
    std::string getConstantVector(const ConstantVector *CV, bool Static);

    /// isAddressExposed - Return true if the specified value's name needs to
    /// have its address taken in order to get a C value of the correct type.
    /// This happens for global variables, byval parameters, and direct allocas.
    bool isAddressExposed(const Value *V) const {
      assert(0 && "not implemented!");

    }

    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const Instruction &I) {
      assert(0 && "not implemented!");
    }

    // isInlineAsm - Check if the instruction is a call to an inline asm chunk
    static bool isInlineAsm(const Instruction& I) {
      const CallInst *call = dyn_cast<CallInst>(&I);
      if (call && isa<InlineAsm>(call->getCalledFunction()))
        return true;
      return false;
    }

    // Instruction visitation functions
    friend class InstVisitor<PTXWriter>;

    void visitReturnInst(ReturnInst &I);
    void visitBranchInst(BranchInst &I);
    void visitSwitchInst(SwitchInst &I);
    void visitInvokeInst(InvokeInst &I) {
      assert(0 && "Lowerinvoke pass didn't work!");
    }

    void visitUnwindInst(UnwindInst &I) {
      assert(0 && "Lowerinvoke pass didn't work!");
    }
    void visitUnreachableInst(UnreachableInst &I);

    void visitPHINode(PHINode &I);
    void visitBinaryOperator(Instruction &I);
    void visitICmpInst(ICmpInst &I);
    void visitFCmpInst(FCmpInst &I);

    void visitCastInst (CastInst &I);
    void visitCmpInst(CmpInst &I);
    void visitSelectInst(SelectInst &I);
    void visitCallInst (CallInst &I);
    void visitInlineAsm(CallInst &I);
    bool visitBuiltinCall(CallInst &I, bool &WroteCallee);

    void visitAllocaInst(AllocaInst &I);
    void visitLoadInst  (LoadInst   &I);
    void visitStoreInst (StoreInst  &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitVAArgInst (VAArgInst &I);

    void visitInsertElementInst(InsertElementInst &I);
    void visitExtractElementInst(ExtractElementInst &I);
    void visitShuffleVectorInst(ShuffleVectorInst &SVI);

    void visitInsertValueInst(InsertValueInst &I);
    void visitExtractValueInst(ExtractValueInst &I);

    void visitInstruction(Instruction &I) {
      errs() << "C Writer does not know about " << I;
      abort();
    }

    void defineRegister(std::string name, Type *Ty, Value *v,
                        bool isSigned = false)
    {
      Out << "  .reg ";
      if(isa<CmpInst>(v) && Ty->getPrimitiveSizeInBits()==1) //predicate?!
        Out << ".pred";
      else
        Out << getTypeStr(Ty,isSigned);
      Out << ' ' << name << ";\n";
    }

    bool isEntryFunction(const Function *F)
    {
      // no users and no return type => entry func
      return (F->use_begin() == F->use_end()
         && F->getReturnType()->getTypeID() == Type::VoidTyID);

    }

    const Value* getParentPointer(const Value* ptr)
    {
      if(parentPointers.find(ptr)!=parentPointers.end())
        return parentPointers.find(ptr)->second;
      else // no parent pointer => this is a parent
        return ptr;
    }

    //determines adress space from underlying object name
    std::string getAddressSpace(const Value *v)
    {
      // GetUnderlyingObject resolves normal GEP instructions and casts,
      // getParentPointer resolves "replaced"(by add/mul inst.) GEP instr.
      //TODO: do iteratively till result is stable??
      const Value* parent =
        getParentPointer(GetUnderlyingObject(getParentPointer(v)));

      //allocated data is stored in local memory
      if(isa<AllocaInst>(parent))
        return ".local";
      else if(getValueName(parent).find(PTX_CONST)!=std::string::npos)
        return ".const";
      else if(getValueName(parent).find(PTX_SHARED)!=std::string::npos)
        return ".shared";
      else if(getValueName(parent).find(PTX_TEX)!=std::string::npos)
        return ".tex";

      return ".global";
    }

    bool isSpecialRegister(const Value *v)
    {
      std::string sourceName = getValueName(v);
      return sourceName.find("__ptx_sreg_") == 0;// =std::string::npos;
    }

    std::string getSpecialRegisterName(const Value *v)
    {
      std::string sourceName = getValueName(v);
      if(!sourceName.find("__ptx_sreg_") == 0)// != std::string::npos)
        assert(false && "this is not an special register!");

      std::string sregName;
      //determine type
      if(sourceName.find("_tid_") != std::string::npos)
        sregName = "%tid.";
      else if(sourceName.find("_ntid_") != std::string::npos)
        sregName = "%ntid.";
      else if(sourceName.find("_ctaid_") != std::string::npos)
        sregName = "%ctaid.";
      else if(sourceName.find("_nctaid_") != std::string::npos)
        sregName = "%nctaid.";
      else if(sourceName.find("_gridid") != std::string::npos)
        return "%gridid";
      else if(sourceName.find("_clock") != std::string::npos)
        return "%clock";
      else
        assert(false && "not implemented");

      //get x,y,z
      if(sourceName.find("id_x") != std::string::npos)
        sregName += 'x';
      else if(sourceName.find("id_y") != std::string::npos)
        sregName += 'y';
      else if(sourceName.find("id_z") != std::string::npos)
        sregName += 'z';

      return sregName;
    }

    std::string getPredicate(Value* predicate, bool negated)
    {
      std::string pred = "";
      if(predicate!=0)
      {
        if(negated)
          pred = "@!";
        else
          pred = " @";
        pred.append(getOperandStr(predicate));
      }
      return pred;
    }

    bool isGotoCodeNecessary(BasicBlock *From, BasicBlock *To);
    void printPHICopiesForSuccessor(BasicBlock *CurBlock,
                                    BasicBlock *Successor,
                                    Value* predicate = 0, bool = false);
    void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                            std::string predicate = "");
    void printGEPExpressionStep(GetElementPtrInst &Ptr,
                                CompositeType *CompTy,
                                unsigned int operand, bool usedGepReg);
    std::string getConstantGEPExpression(const User *Ptr);

    std::string getValueName(const Value *Operand);
  };

#endif
