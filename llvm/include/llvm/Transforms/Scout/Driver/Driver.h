/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 */

#ifndef _SC_DRIVER_H_
#define _SC_DRIVER_H_

#include <llvm/IRBuilder.h>
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

static unsigned getSizeInBytes(llvm::Type *type) {
  if(type->isSingleValueType() && !type->isPointerTy()) {
    return type->getPrimitiveSizeInBits() / 8;
  } else if(type->isArrayTy()) {
    int numElements = llvm::cast< llvm::ArrayType >(type)->getNumElements();
    return numElements * getSizeInBytes(type->getContainedType(0));
  } else {
    unsigned size = 0;

    typedef llvm::Type::subtype_iterator SubTypeIterator;
    SubTypeIterator subtype = type->subtype_begin();
    for( ; subtype != type->subtype_end(); ++subtype) {
      size += getSizeInBytes(*subtype);
    }
    return size;
  }
}

static llvm::Type *getOrInsertType(llvm::Module &module, std::string name, llvm::Type *type = NULL) {
  if(llvm::Type *ty = module.getTypeByName(name))
    return ty;

  if(type == NULL)
    type = llvm::StructType::create(llvm::getGlobalContext(), name);
  else
    type = llvm::StructType::create(llvm::ArrayRef< llvm::Type * >(type), name);

  return type;
}

class Driver {
public:
  Driver(llvm::Module &module, llvm::IRBuilder<> &builder, bool debug = false);

  typedef llvm::BasicBlock::iterator InstIterator;
  void setInsertPoint(llvm::BasicBlock *BB);
  void setInsertPoint(llvm::BasicBlock *BB, InstIterator I);

  llvm::Function *declareFunction(llvm::Type *result,
                                  std::string name,
                                  llvm::Type *a = NULL,
                                  llvm::Type *b = NULL,
                                  llvm::Type *c = NULL,
                                  llvm::Type *d = NULL,
                                  llvm::Type *e = NULL,
                                  llvm::Type *f = NULL,
                                  llvm::Type *g = NULL,
                                  llvm::Type *h = NULL,
                                  llvm::Type *i = NULL);

  llvm::Value *insertCall(llvm::StringRef name);
  llvm::Value *insertCall(llvm::StringRef name,
                    llvm::Value **begin,
                    llvm::Value **end);
  llvm::Value *insertCall(llvm::StringRef name,
                          llvm::ArrayRef< llvm::Value * > args);

  llvm::Value *insertGet(llvm::StringRef name);

  bool getDebug();
  llvm::IRBuilder<> getBuilder();

  llvm::PointerType *getPtrTy(llvm::Type *type);

  llvm::Module& getModule(){
    return _module;
  }

 protected:
  llvm::Module &_module;
  llvm::IRBuilder<> &_builder;
  bool _debug;
  llvm::Type *i8Ty;
  llvm::Type *i8PtrTy;
  llvm::Type *i8DblPtrTy;
  llvm::Type *i16Ty;
  llvm::Type *i32Ty;
  llvm::Type *i32PtrTy;
  llvm::Type *i64Ty;
  llvm::Type *fltTy;
  llvm::Type *voidTy;
};

#endif
