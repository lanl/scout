/**
 * @file   PTXTargetMachine.h
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

#ifndef CTARGETMACHINE_H
#define CTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

struct SimplePTXTargetMachine : public TargetMachine {
  //  const TargetData DataLayout;       // Calculates type size & alignment

 SimplePTXTargetMachine(const Target &T, StringRef TT,
                        StringRef CPU, StringRef FS,
                        const TargetOptions &Options,
			Reloc::Model& RM, CodeModel::Model& CM,
			CodeGenOpt::Level OL)
   // ndm - MERGE
   // : TargetMachine(T, TT, CPU, FS) {}
   : TargetMachine(T, TT, CPU, FS, Options) {}

  //const Module &M, const std::string &FS)
  //  : DataLayout(&M) {}

  virtual bool WantsWholeFile() const { return true; }
  virtual bool addPassesToEmitFile(PassManagerBase &PM,
				   formatted_raw_ostream &Out,
				   CodeGenFileType FileType,
				   CodeGenOpt::Level OptLevel,
				   bool DisableVerify);

  // This class always works, but shouldn't be the default in most cases.
  //static unsigned getModuleMatchQuality(const Module &M) { return 1; }

  virtual const TargetData *getTargetData() const { return 0; }
};

extern Target ThePTXBackendTarget;

} // End llvm namespace


#endif
