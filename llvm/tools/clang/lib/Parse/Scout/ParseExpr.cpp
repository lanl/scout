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

#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

// scout - Parse the right hand side of a vector expression, e.g:
// 1.0, or float3(1.0, 1.0, 1.0)
ExprResult Parser::ParseScoutVectorRHS(BuiltinType::Kind kind, ScoutVectorType vectorType){
  size_t length;
  tok::TokenKind expectKind;
  QualType type;

  // set the dimensions and types based on the vector keyword

  bool isFloatVec = false;
  bool isDoubleVec = false;

  CanQualType elementType;
  switch(kind){
    case BuiltinType::Bool2:
    {
      expectKind = tok::kw_bool2;
      length = 2;
      type = Actions.Context.Bool2Ty;
      elementType = Actions.Context.BoolTy;
      break;
    }
    case BuiltinType::Char2:
    {
      expectKind = tok::kw_char2;
      length = 2;
      type = Actions.Context.Char2Ty;
      elementType = Actions.Context.CharTy;
      break;
    }
    case BuiltinType::Short2:
    {
      expectKind = tok::kw_short2;
      length = 2;
      type = Actions.Context.Short2Ty;
      elementType = Actions.Context.ShortTy;
      break;
    }
    case BuiltinType::Int2:
    {
      expectKind = tok::kw_int2;
      length = 2;
      type = Actions.Context.Int2Ty;
      elementType = Actions.Context.IntTy;
      break;
    }
    case BuiltinType::Long2:
    {
      expectKind = tok::kw_long2;
      length = 2;
      type = Actions.Context.Long2Ty;
      elementType = Actions.Context.LongTy;
      break;
    }
    case BuiltinType::Float2:
    {
      expectKind = tok::kw_float2;
      length = 2;
      type = Actions.Context.Float2Ty;
      elementType = Actions.Context.FloatTy;
      isFloatVec = true;
      break;
    }
    case BuiltinType::Double2:
    {
      expectKind = tok::kw_double2;
      length = 2;
      type = Actions.Context.Double2Ty;
      elementType = Actions.Context.DoubleTy;
      isDoubleVec = true;
      break;
    }
    case BuiltinType::Bool3:
    {
      expectKind = tok::kw_bool3;
      length = 3;
      type = Actions.Context.Bool3Ty;
      elementType = Actions.Context.BoolTy;
      break;
    }
    case BuiltinType::Char3:
    {
      expectKind = tok::kw_char3;
      length = 3;
      type = Actions.Context.Char3Ty;
      elementType = Actions.Context.CharTy;
      break;
    }
    case BuiltinType::Short3:
    {
      expectKind = tok::kw_short3;
      length = 3;
      type = Actions.Context.Short3Ty;
      elementType = Actions.Context.ShortTy;
      break;
    }
    case BuiltinType::Int3:
    {
      expectKind = tok::kw_int3;
      length = 3;
      type = Actions.Context.Int3Ty;
      elementType = Actions.Context.IntTy;
      break;
    }
    case BuiltinType::Long3:
    {
      expectKind = tok::kw_long3;
      length = 3;
      type = Actions.Context.Long3Ty;
      elementType = Actions.Context.LongTy;
      break;
    }
    case BuiltinType::Float3:
    {
      expectKind = tok::kw_float3;
      length = 3;
      type = Actions.Context.Float3Ty;
      elementType = Actions.Context.FloatTy;
      isFloatVec = true;
      break;
    }
    case BuiltinType::Double3:
    {
      expectKind = tok::kw_double3;
      length = 3;
      type = Actions.Context.Double3Ty;
      elementType = Actions.Context.DoubleTy;
      isDoubleVec = true;
      break;
    }
    case BuiltinType::Bool4:
    {
      expectKind = tok::kw_bool4;
      length = 4;
      type = Actions.Context.Bool4Ty;
      elementType = Actions.Context.BoolTy;
      break;
    }
    case BuiltinType::Char4:
    {
      expectKind = tok::kw_char4;
      length = 4;
      type = Actions.Context.Char4Ty;
      elementType = Actions.Context.CharTy;
      break;
    }
    case BuiltinType::Short4:
    {
      expectKind = tok::kw_short4;
      length = 4;
      type = Actions.Context.Short4Ty;
      elementType = Actions.Context.ShortTy;
      break;
    }
    case BuiltinType::Int4:
    {
      expectKind = tok::kw_int4;
      length = 4;
      type = Actions.Context.Int4Ty;
      elementType = Actions.Context.IntTy;
      break;
    }
    case BuiltinType::Long4:
    {
      expectKind = tok::kw_long4;
      length = 4;
      type = Actions.Context.Long4Ty;
      elementType = Actions.Context.LongTy;
      break;
    }
    case BuiltinType::Float4:
    {
      expectKind = tok::kw_float4;
      length = 4;
      type = Actions.Context.Float4Ty;
      elementType = Actions.Context.FloatTy;
      isFloatVec = true;
      break;
    }
    case BuiltinType::Double4:
    {
      expectKind = tok::kw_double4;
      length = 4;
      type = Actions.Context.Double4Ty;
      elementType = Actions.Context.DoubleTy;
      isDoubleVec = true;
      break;
    }
    default:
      assert(false && "expected a scout vector kind");
  }

  if(Tok.is(expectKind)){
    ConsumeToken();

    if(Tok.isNot(tok::l_paren)){
      Diag(Tok, diag::err_expected_lparen);
      return ExprError();
    }

    SourceLocation LParenLoc = ConsumeParen();

    ExprVector Exprs;
    CommaLocsTy CommaLocs;

    if(ParseExpressionList(Exprs, CommaLocs)){
      return ExprError();
    }

    for(size_t i = 0; i < Exprs.size(); ++i){
      if(FloatingLiteral* fl = dyn_cast<FloatingLiteral>(Exprs[i])){
        if(isFloatVec){
          float floatValue = fl->getValueAsApproximateDouble();
          fl->setValue(Actions.Context, llvm::APFloat(fl->getSemantics(), floatValue));
        }
        else if(isDoubleVec){
          double doubleValue = fl->getValueAsApproximateDouble();
          fl->setValue(Actions.Context, llvm::APFloat(fl->getSemantics(), doubleValue));
        }

        if(vectorType == ScoutVectorColor){
          double v = fl->getValue().convertToDouble();
          if(v < 0 || v > 1){
            Diag(CommaLocs[i], diag::warn_vector_color_clamp);
          }
        }
      }
      else if(IntegerLiteral* il = dyn_cast<IntegerLiteral>(Exprs[i])){
        if(vectorType == ScoutVectorColor){
          double v = il->getValue().roundToDouble(true);
          if(v < 0 || v > 1){
            Diag(CommaLocs[i], diag::warn_vector_color_clamp);
          }
        }
      }
    }

    if(Exprs.size() != length){
      Diag(Tok, diag::err_invalid_scout_vector_init);
      return ExprError();
    }

    if(Tok.isNot(tok::r_paren)){
      Diag(Tok, diag::err_expected_rparen);
      return ExprError();
    }

    SourceLocation RParenLoc = ConsumeParen();

    InitListExpr* le =
    new (Actions.Context)
    InitListExpr(Actions.Context, LParenLoc, Exprs, RParenLoc);

    le->setType(type);

    return le;
  }
  else if(Tok.is(tok::numeric_constant)){
    SourceLocation Loc = Tok.getLocation();

    ExprResult er = ParseExpression();

    if(vectorType == ScoutVectorColor){
      if(FloatingLiteral* fl = dyn_cast<FloatingLiteral>(er.get())){
        double v = fl->getValue().convertToDouble();
        if(v < 0 || v > 1){
          Diag(Loc, diag::warn_vector_color_clamp);
        }
      }
      else if(IntegerLiteral* il = dyn_cast<IntegerLiteral>(er.get())){
        double v = il->getValue().roundToDouble(true);
        if(v < 0 || v > 1){
          Diag(Loc, diag::warn_vector_color_clamp);
        }
      }
    }

    if(er.isInvalid()){
      return ExprError();
    }

    SmallVector<Expr *, 4> initExprs;

    for(size_t i = 0; i < length; ++i){
      initExprs.push_back(er.get());
    }

    InitListExpr* le =
    new (Actions.Context)
    InitListExpr(Actions.Context, Loc, initExprs, Loc);

    le->setType(elementType);

    return le;
  }

  return ParseExpression();
}


