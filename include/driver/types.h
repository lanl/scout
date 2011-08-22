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

#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"

#ifndef _SC_TYPES_H_
#define _SC_TYPES_H_

llvm::Type *i8Ty       = llvm::Type::getInt8Ty(llvm::getGlobalContext());
llvm::Type *i8PtrTy    = llvm::PointerType::get(i8Ty, 0);
llvm::Type *i8DblPtrTy = llvm::PointerType::get(i8PtrTy, 0);

llvm::Type *i16Ty      = llvm::Type::getInt16Ty(llvm::getGlobalContext());
llvm::Type *i32Ty      = llvm::Type::getInt32Ty(llvm::getGlobalContext());
llvm::Type *i32PtrTy   = llvm::PointerType::get(i32Ty, 0);

llvm::Type *i64Ty      = llvm::Type::getInt64Ty(llvm::getGlobalContext());

llvm::Type *fltTy      = llvm::Type::getFloatTy(llvm::getGlobalContext());

llvm::Type *voidTy     = llvm::Type::getVoidTy(llvm::getGlobalContext());

#endif
