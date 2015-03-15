/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
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
 * ##########################################################################
 */

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

void ScoutExpr::printPretty() const{
  Stmt::printPretty(llvm::errs(), 0, PrintingPolicy(LangOptions()));
}

bool SpecExpr::isSymbol(const char* s){
  bool match = false;
      
  size_t i = 0;
  char c;

  for(;;){
    c = s[i];
    
    if(c == '\0'){
      return match;
    }

    if(isalpha(c)){
      match = true;
    }    
    else if(isdigit(c)){
      if(!match){
        return false;
      }
    }
    else if(c != '_'){
      return false;
    }

    ++i;
  }
      
  return match;
}

std::string SpecExpr::toUpper(const std::string& str){
  std::string ret = str;
  transform(ret.begin(), ret.begin() + 1, ret.begin(), ::toupper);
  return ret;
}

SpecValueExpr* SpecExpr::toValue(){
  if(kind() != SpecValue){
    return 0;
  }
  
  return static_cast<SpecValueExpr*>(this);
}

bool SpecValueExpr::isFrameVar(){
  DeclRefExpr* dr = dyn_cast<DeclRefExpr>(Exp);
  
  if(!dr){
    return false;
  }
  
  return isa<FrameVarType>(dr->getDecl()->getType().getTypePtr());
}

VarDecl* SpecValueExpr::getFrameVar(){
  DeclRefExpr* dr = dyn_cast<DeclRefExpr>(Exp);
  assert(dr);
  
  assert(isa<FrameVarType>(dr->getDecl()->getType().getTypePtr()));
  
  return cast<VarDecl>(dr->getDecl());
}

bool SpecValueExpr::isInteger(){
  return isa<IntegerLiteral>(Exp);
}

int64_t SpecValueExpr::getInteger(){
  IntegerLiteral* i = dyn_cast<IntegerLiteral>(Exp);
  assert(i);
  
  return i->getValue().getSExtValue();
}

bool SpecValueExpr::isString(){
  return isa<StringLiteral>(Exp);
}

std::string SpecValueExpr::getString(){
  StringLiteral* s = dyn_cast<StringLiteral>(Exp);
  assert(s);
  
  return s->getString().str();
}

Expr* SpecExpr::toExpr(){
  SpecValueExpr* v = toValue();
  if(v){
    return v->getExpression();
  }
  
  return 0;
}

SpecObjectExpr* SpecExpr::toObject(){
  if(kind() != SpecObject){
    return 0;
  }
  
  return static_cast<SpecObjectExpr*>(this);
}

SpecArrayExpr* SpecExpr::toArray(){
  if(kind() != SpecArray){
    return 0;
  }
  
  return static_cast<SpecArrayExpr*>(this);
}

bool SpecExpr::isFrameVar(){
  SpecValueExpr* v = toValue();
  if(v){
    return v->isFrameVar();
  }
  
  return false;
}

VarDecl* SpecExpr::getFrameVar(){
  SpecValueExpr* v = toValue();
  assert(v);
  return v->getFrameVar();
}

bool SpecExpr::isInteger(){
  SpecValueExpr* v = toValue();
  if(v){
    return v->isInteger();
  }
  
  return false;
}

int64_t SpecExpr::getInteger(){
  SpecValueExpr* v = toValue();
  assert(v);
  
  return v->getInteger();
}

bool SpecExpr::isString(){
  SpecValueExpr* v = toValue();
  if(v){
    return v->isString();
  }
  
  return false;
}

std::string SpecExpr::getString(){
  SpecValueExpr* v = toValue();
  assert(v);
  
  return v->getString();
}
