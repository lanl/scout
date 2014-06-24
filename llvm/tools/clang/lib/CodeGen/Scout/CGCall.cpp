/*
 * ###########################################################################
 * Copyright (c) 2010, Los Alamos National Security, LLC.
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
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 

#include "CGCall.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"

using namespace clang;
using namespace CodeGen;

const CGFunctionInfo &
CodeGenTypes::arrangeStencilFunctionDeclaration(const FunctionDecl *FD) {

  CanQualType FTy = FD->getType()->getCanonicalTypeUnqualified();
  CanQual<FunctionProtoType> FTP = FTy.getAs<FunctionProtoType>();

  SmallVector<CanQualType, 16> argTypes;

  //stencil args
  for (unsigned i = 0; i <=6; i++) {
    argTypes.push_back(Context.getPointerType(
        CanQualType::CreateUnsafe(Context.getIntPtrType())));
  }

  // Add the formal parameters.
  for (unsigned i = 0, e = FTP->getNumParams(); i != e; ++i)
    argTypes.push_back(FTP->getParamType(i));


  return arrangeLLVMFunctionInfo(FTP->getReturnType(), false, argTypes,
                                     FTP->getExtInfo(), RequiredArgs::All);

}


// based on  &arrangeLLVMFunctionInfo(CodeGenTypes &CGT, bool IsInstanceMethod,
// SmallVectorImpl<CanQualType> &prefix,CanQual<FunctionProtoType> FTP,
// FunctionType::ExtInfo extInfo)
static const CGFunctionInfo &arrangeLLVMStencilFunctionInfo(CodeGenTypes &CGT,
                                                     bool IsInstanceMethod,
                                       SmallVectorImpl<CanQualType> &prefix,
                                             CanQual<FunctionProtoType> FTP,
                                              FunctionType::ExtInfo extInfo) {
  RequiredArgs required = RequiredArgs::forPrototypePlus(FTP, prefix.size());

  //stencil args
  for (unsigned i=0; i<=6; i++) {
    prefix.push_back(CGT.getContext().getPointerType(
          CanQualType::CreateUnsafe(CGT.getContext().getIntPtrType())));
  }

  for (unsigned i = 0, e = FTP->getNumParams(); i != e; ++i)
    prefix.push_back(FTP->getParamType(i));
  CanQualType resultType = FTP->getReturnType().getUnqualifiedType();
  return CGT.arrangeLLVMFunctionInfo(resultType, IsInstanceMethod, prefix,
                                     extInfo, required);
}

// based on &arrangeFreeFunctionType(CodeGenTypes &CGT,
//SmallVectorImpl<CanQualType> &prefix,  CanQual<FunctionProtoType> FTP)
static const CGFunctionInfo &arrangeStencilFunctionType(CodeGenTypes &CGT,
                                      SmallVectorImpl<CanQualType> &prefix,
                                            CanQual<FunctionProtoType> FTP) {
  return arrangeLLVMStencilFunctionInfo(CGT, false, prefix, FTP, FTP->getExtInfo());
}

// based on CodeGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> FTP)
const CGFunctionInfo &
CodeGenTypes::arrangeStencilFunctionType(CanQual<FunctionProtoType> FTP) {
  SmallVector<CanQualType, 16> argTypes;
  return ::arrangeStencilFunctionType(*this, argTypes, FTP);
}

// currently unused
// based on CodeGenTypes::arrangeCXXConstructorCall
const CGFunctionInfo &
CodeGenTypes::arrangeStencilFunctionCall(const CallArgList &args,
                                        const FunctionDecl *D) {

  CanQualType FTy = D->getType()->getCanonicalTypeUnqualified();
  CanQual<FunctionProtoType> FTP = FTy.getAs<FunctionProtoType>();

  SmallVector<CanQualType, 16> ArgTypes;
  //stencil args
  for (unsigned i = 0; i <=6; i++) {
    ArgTypes.push_back(Context.getPointerType(
        CanQualType::CreateUnsafe(Context.getIntPtrType())));
  }
  for (CallArgList::const_iterator i = args.begin(), e = args.end(); i != e;
      ++i)
    ArgTypes.push_back(Context.getCanonicalParamType(i->Ty));

  return arrangeLLVMFunctionInfo(FTP->getReturnType(), false, ArgTypes,
                                       FTP->getExtInfo(), RequiredArgs::All);
}
