##===- lib/Target/CBackend/Makefile ------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = PTXBackend
BUILD_RELINKED=1
SHARED_LIBRARY=1

LLVMLIBS=LLVMMC.a LLVMTarget.a LLVMCodeGen.a
LINK_LIBS_IN_SHARED=1

CFlags += -fPIC

include $(LEVEL)/Makefile.common
