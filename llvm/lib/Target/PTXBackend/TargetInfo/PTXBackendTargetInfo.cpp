/**
 * @file   PTXBackendTargetInfo.cpp
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
#include <PTXTargetMachine.h>
#include <llvm/Module.h>
#include <llvm/Support/TargetRegistry.h>
using namespace llvm;

Target llvm::ThePTXBackendTarget;

extern "C" void LLVMInitializePTXBackendTargetInfo() {
  RegisterTarget<> X(ThePTXBackendTarget, "simple-ptx", "Simple PTX backend");
}

extern "C" void LLVMInitializePTXBackendTargetMC() {}
