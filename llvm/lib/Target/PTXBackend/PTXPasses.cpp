/**
 * @file   PTXPasses.cpp
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
#include <PTXPasses.h>

//using namespace llvm;

#define LOG2_E 1.442695041
#define LOG2_E_REC 0.6931471806

bool PTXBackendInsertSpecialInstructions::replaceSpecialFunctionsWithPTXInstr(
							      CallInst* callI)
{
  Function* F = callI->getCalledFunction();

  //no called function? Happens, iff called function is a pointer?! TODO
  if(F==0)
    return false;
  //F->dump();
  unsigned int intrinsicID = F->getIntrinsicID();

  //TODO: test for more input values, i think this versions(exp,pow)
  //are inaccurate or broken

  //exp(x) = ex2(x * lg2(e))
  if(intrinsicID == Intrinsic::exp || F->getName() == "expf")
    {
      Constant* lg2EConst =
	ConstantFP::get(Type::getPrimitiveType(callI->getContext(),
					       Type::FloatTyID),LOG2_E);

      BinaryOperator* MulInst =
	BinaryOperator::Create(Instruction::FMul, callI->getArgOperand(0),
			       lg2EConst, "", callI);
      callI->setCalledFunction(ex2fFun);  // set function call do exp2
      callI->setArgOperand(0,MulInst); // set source to our calculated tmp value
      return true;
    }
  //log(x) = lg2(x) * (1/lg2(e))
  else if(intrinsicID == Intrinsic::log || F->getName() == "logf")
    {
      callI->setCalledFunction(lg2fFun); //convert log to lg2 function call
      Constant* lg2EConst =
	ConstantFP::get(Type::getPrimitiveType(callI->getContext(),
					       Type::FloatTyID),LOG2_E_REC);

      BinaryOperator* MulInst =
	BinaryOperator::Create(Instruction::FMul, callI, lg2EConst);
      MulInst->insertAfter(callI);
      // replace uses of CallI with our new result fron FDiv
      callI->replaceAllUsesWith(MulInst);
      //reset operand 0 to ex2 call (it got replaced by "replaceAllUsesWith()"
      MulInst->setOperand(0,callI);
      return true;
    }

  // pow (a,x) = ex2(x * lg2(a)) = a^x
  //TODO: not working???? why??? check phong shader calculation: specular
  else if(intrinsicID ==  Intrinsic::pow || F->getName() == "powf")
    {

      Function *ex2SizeFun;
      Function *lg2SizeFun;

      if(callI->getType()->isFloatTy()) {
        ex2SizeFun = ex2fFun;
        lg2SizeFun = lg2fFun;

      } else if(callI->getType()->isDoubleTy()) {
        ex2SizeFun = ex2Fun;
        lg2SizeFun = lg2Fun;

      } else {
        assert(false);
      }

      // create and insert instructions
      CallInst* callLg2 =
        CallInst::Create(lg2SizeFun, callI->getArgOperand(0)); //log2(a)
      callLg2->insertBefore(callI);
      BinaryOperator* MulInst =
        BinaryOperator::Create(Instruction::FMul,
                               callI->getArgOperand(1), callLg2); //x * log2(a)

      MulInst->insertBefore(callI);
      CallInst* callEx2 = CallInst::Create(ex2SizeFun, MulInst); // ex2(x * lg2(a))
      callEx2->insertBefore(callI);

      //remove old pow call
      callI->replaceAllUsesWith(callEx2);
      callI->eraseFromParent();
      return true;
    }


  //sinf = llvm.sin
  if(F->getName() == "sinf")
    {
      callI->setCalledFunction(sinfFun);  // set function call do llvm.sin
      return true;
    }
  //cosf = llvm.cos
  else if(F->getName() == "cosf")
    {
      callI->setCalledFunction(cosfFun);  // set function call do llvm.sin
      return true;
    }
  // tan(x) = sin(x) / cos(x)
  else if(F->getName() == "tanf")
    {
      CallInst* callSin = callI;
      callSin->setCalledFunction(sinfFun);  // set function call do sin
      CallInst* callCos = CallInst::Create(cosfFun,callI->getArgOperand(0));
      callCos->insertAfter(callSin);
      BinaryOperator* DivInst =
	BinaryOperator::Create(Instruction::FDiv, callSin, callCos);

      DivInst->insertAfter(callCos);
      callI->replaceAllUsesWith(DivInst);
      //reset operand 0 to sin call (it got replaced by "replaceAllUsesWith()"
      DivInst->setOperand(0,callSin);
      return true;
    }
  // atan(x) = cos(x) / sin(x)
  else if(F->getName() == "atanf")
    {
      CallInst* callSin = callI;
      callSin->setCalledFunction(sinfFun);  // set function call do sin
      CallInst* callCos = CallInst::Create(cosfFun,callI->getArgOperand(0));
      callCos->insertAfter(callSin);
      BinaryOperator* DivInst =
	BinaryOperator::Create(Instruction::FDiv, callCos, callSin);
      DivInst->insertAfter(callCos);
      callI->replaceAllUsesWith(DivInst);
      //reset operand 0 to sin call (it got replaced by "replaceAllUsesWith()"
      DivInst->setOperand(1,callSin);
      return true;
    }
  else if(F->getName().startswith("llvm.dbg"))
    {
      callI->eraseFromParent();
    }

  return false;
}


bool PTXBackendInsertSpecialInstructions::simplifyGEPInstructions(
				       GetElementPtrInst* GEPInst)
{
  Value* parentPointer = GEPInst->getOperand(0);
  const Value* topParent = parentPointer;
  CompositeType* CompTy = cast<CompositeType>(parentPointer->getType());


  if(isa<GlobalVariable>(parentPointer)) //HACK: !!!!
  {
    Function *constWrapper =
      Function::Create(FunctionType::get(parentPointer->getType(),true),
		       GlobalValue::ExternalLinkage,
		       Twine(CONSTWRAPPERNAME));

    std::vector<Value*> params;
    params.push_back(parentPointer);

    //create and insert wrapper call
    CallInst * wrapperCall =
      CallInst::Create(constWrapper, params, "", GEPInst);
    parentPointer = wrapperCall;
  }

  Value* currentAddrInst =
    new PtrToIntInst(parentPointer,
		     IntegerType::get(GEPInst->getContext(),
				      PTX_PTR_SIZE),
		     "", GEPInst);

  unsigned int constantOffset = 0;

  for(unsigned int op=1; op<GEPInst->getNumOperands(); ++op)
  {
    unsigned int TypeIndex;
    //we have a constant struct/array acces
    if(ConstantInt* ConstOP = dyn_cast<ConstantInt>(GEPInst->getOperand(op)))
    {
      unsigned int offset = 0;
      TypeIndex = ConstOP->getZExtValue();
      for(unsigned int ty_i=0; ty_i<TypeIndex; ty_i++)
      {
	Type* elementType = CompTy->getTypeAtIndex(ty_i);
	unsigned int align = PTXWriter::getAlignmentByte(elementType);
	offset += PTXWriter::getPadding(offset, align);
	offset += PTXWriter::getTypeByteSize(elementType);
      }

      //add padding for accessed type
      unsigned int align =
	PTXWriter::getAlignmentByte(CompTy->getTypeAtIndex(TypeIndex));
      offset += PTXWriter::getPadding(offset, align);

      constantOffset += offset;

      /*
      //insert addition of new offset before GEPInst
      Constant* newConstOffset = ConstantInt::get(IntegerType::get(GEPInst->getContext(), PTX_PTR_SIZE),offset);
      currentAddrInst = BinaryOperator::Create(Instruction::Add, currentAddrInst, newConstOffset,"", GEPInst);
      */
    }
    // none constant index (=> only array/verctor allowed)
    else
    {
      // we only have array/vectors here,
      // therefore all elements have the same size
      TypeIndex = 0;

      Type* elementType = CompTy->getTypeAtIndex(TypeIndex);
      unsigned int size = PTXWriter::getTypeByteSize(elementType);

      //add padding
      unsigned int align = PTXWriter::getAlignmentByte(elementType);
      size += PTXWriter::getPadding(size, align);

      Constant* newConstSize =
	ConstantInt::get(IntegerType::get(GEPInst->getContext(),
					  PTX_PTR_SIZE),
			 size);

      Value *operand = GEPInst->getOperand(op);

      //HACK TODO: Inserted by type replacement.. this code could break something????
      if(PTXWriter::getTypeByteSize(operand->getType())>4 && PTX_PTR_SIZE != 64)
      {
	//trunctate
	operand =
	  new TruncInst(operand,
			IntegerType::get(GEPInst->getContext(),
					 PTX_PTR_SIZE),
			"", GEPInst);
      }

      Type *curTy = newConstSize->getType();
      Type *oldTy = operand->getType();
      if(oldTy != curTy) {
        operand = CastInst::Create(Instruction::ZExt, operand, curTy, "", GEPInst);
      }
      BinaryOperator* tmpMul =
	BinaryOperator::Create(Instruction::Mul, newConstSize, operand,
			       "", GEPInst);
      currentAddrInst =
	BinaryOperator::Create(Instruction::Add, currentAddrInst, tmpMul,
			       "", GEPInst);
	}

      //step down in type hirachy
      CompTy = dyn_cast<CompositeType>(CompTy->getTypeAtIndex(TypeIndex));
    }

  //insert addition of new offset before GEPInst
  Constant* newConstOffset =
    ConstantInt::get(IntegerType::get(GEPInst->getContext(),
				      PTX_PTR_SIZE),
		     constantOffset);
  currentAddrInst =
    BinaryOperator::Create(Instruction::Add, currentAddrInst,
			   newConstOffset, "", GEPInst);

  //convert offset to ptr type (nop)
  IntToPtrInst* intToPtrInst =
    new IntToPtrInst(currentAddrInst,GEPInst->getType(),"", GEPInst);

  //replace uses of the GEP instruction with the newly calculated pointer
  GEPInst->replaceAllUsesWith(intToPtrInst);
  GEPInst->eraseFromParent();

  //insert new pointer into parent list
  while(parentPointers.find(topParent)!=parentPointers.end())
    topParent = parentPointers.find(topParent)->second;
  parentPointers[intToPtrInst] = topParent;

  return true;
}

char PTXBackendInsertSpecialInstructions::ID = 0;
char PTXPolishBeforeCodegenPass::ID = 0;
