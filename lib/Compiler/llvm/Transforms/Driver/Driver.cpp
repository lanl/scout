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

#include "llvm/Transforms/Scout/Driver/Driver.h"

using namespace llvm;

llvm::Type *getOrInsertType(llvm::Module &module, std::string name, llvm::Type *type) {
  if(llvm::Type *ty = module.getTypeByName(name))
    return ty;

  if(type == NULL)
    type = llvm::StructType::create(llvm::getGlobalContext(), name);
  else
    type = llvm::StructType::create(llvm::ArrayRef< llvm::Type * >(type), name);

  return type;
}

unsigned getSizeInBytes(llvm::Type *type) {
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

Driver::Driver(Module &module, IRBuilder<> &builder, bool debug)
  : _module(module), _builder(builder), _debug(debug),
    i8Ty((Type::getInt8Ty(_builder.getContext()))),
    i8PtrTy(Type::getInt8PtrTy(_builder.getContext())),
    i8DblPtrTy(PointerType::getUnqual(i8PtrTy)),
    i16Ty(Type::getInt16Ty(_builder.getContext())),
    i32Ty(Type::getInt32Ty(_builder.getContext())),
    i32PtrTy(Type::getInt32PtrTy(_builder.getContext())),
    i64Ty(Type::getInt64Ty(_builder.getContext())),
    fltTy(Type::getFloatTy(_builder.getContext())),
    voidTy(Type::getVoidTy(_builder.getContext()))
{
}

void Driver::setInsertPoint(BasicBlock *BB) {
  _builder.SetInsertPoint(BB);
}

void Driver::setInsertPoint(BasicBlock *BB, InstIterator I) {
  _builder.SetInsertPoint(BB, I);
}

Function *Driver::declareFunction(Type *result,
                                  std::string name,
                                  Type *a,
                                  Type *b,
                                  Type *c,
                                  Type *d,
                                  Type *e,
                                  Type *f,
                                  Type *g,
                                  Type *h,
                                  Type *i) {
  if(Function *func = _module.getFunction(name)) {
    return func;
  }

  std::vector< Type * > types;
  if(a) types.push_back(a);
  if(b) types.push_back(b);
  if(c) types.push_back(c);
  if(d) types.push_back(d);
  if(e) types.push_back(e);
  if(f) types.push_back(f);
  if(g) types.push_back(g);
  if(h) types.push_back(h);
  if(i) types.push_back(i);
  return Function::Create(FunctionType::get(result, types, false),
                          GlobalValue::ExternalLinkage,
                          name, &_module);
}

Value *Driver::insertCall(StringRef name) {
  return _builder.CreateCall(_module.getFunction(name));
}

Value *Driver::insertCall(StringRef name,
                          ArrayRef< Value * > args) {
  return _builder.CreateCall(_module.getFunction(name), args);
}

Value *Driver::insertCall(StringRef name,
                          Value **begin,
                          Value **end) {
  ArrayRef< Value* > args(begin, end);
  return _builder.CreateCall(_module.getFunction(name), args);
}

Value *Driver::insertGet(StringRef name) {
  return _builder.CreateLoad(_module.getNamedGlobal(name));
}

bool Driver::getDebug() {
  return _debug;
}

PointerType *Driver::getPtrTy(Type *type) {
  return PointerType::get(type, 0);
}

IRBuilder<> Driver::getBuilder() {
  return _builder;
}
