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
#include "ParsePragma.h"
#include "RAIIObjectsForParser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

// scout - parser utility method
// insert CPP code into the lexer stream for parsing.
// Inserts a stream of tokens before or after the current token Tok.
// This is a good method for handling cases such as inserting the call
// to initScout(argc, argv) at the beginning of main(), for other cases
// it may be necessary to construct the AST manually.
void Parser::InsertCPPCode(const std::string& code,
                           SourceLocation location,
                           bool beforeLookAhead){

  typedef std::vector<Token> TokenStream;
  Lexer lexer(code, PP);

  TokenStream tokenStream;

  for(;;){
    Token tok;
    lexer.Lex(tok);
    tok.setLocation(location);
    if(tok.is(tok::eof)){
      break;
    }
    tokenStream.push_back(tok);
  }

  if(beforeLookAhead){
    tokenStream.push_back(Tok);
  }

  for(TokenStream::reverse_iterator itr = tokenStream.rbegin(),
      itrEnd = tokenStream.rend(); itr != itrEnd; ++itr){
    PP.EnterToken(*itr);
  }

  if(beforeLookAhead){
    ConsumeAnyToken();
  }
}

// scout - debugging method for displaying the next N lookahead tokens
void Parser::DumpLookAheads(unsigned N){
  for(unsigned i = 0; i < N; ++i){
    const Token& t = GetLookAheadToken(i);

    llvm::outs() << "lookahead[" << i << "]: " << t.getName();

    if(t.is(tok::eof) || t.is(tok::semi)){
      llvm::outs() << "\n";
      break;
    }

    if(t.isAnnotation()){
      llvm::outs() << "\n";
      continue;
    }

    size_t length = t.getLength();

    if(t.isLiteral()){
      std::string str = t.getLiteralData();
      str = str.substr(0, length);
      llvm::outs() << " = " << str;
    }
    else if(t.is(tok::identifier)){
      llvm::outs() << " = " << t.getIdentifierInfo()->getName().str();
    }
    else if(t.is(tok::raw_identifier)){
      std::string str = t.getRawIdentifierData();
      str = str.substr(0, length);
      llvm::outs() << " = " << str;
    }
    llvm::outs() << "\n";
  }
  llvm::outs() << "\n";
}

std::string Parser::TokToStr(const Token& tok){
  if(tok.isLiteral()){
    size_t length = tok.getLength();

    std::string ret = tok.getLiteralData();
    ret = ret.substr(0, length);
    return ret;
  }
  else if(tok.is(tok::identifier)){
    return tok.getIdentifierInfo()->getName().str();
  }

  switch(tok.getKind()){
    case tok::l_square:{
      return "[";
    }
    case tok::r_square:{
      return "]";
    }
    case tok::l_paren:{
      return "(";
    }
    case tok::r_paren:{
      return ")";
    }
    case tok::l_brace:{
      return "{";
    }
    case tok::r_brace:{
      return "}";
    }
    case tok::period:{
      return ".";
    }
    case tok::ellipsis:{
      return "...";
    }
    case tok::amp:{
      return "&";
    }
    case tok::ampamp:{
      return "&&";
    }
    case tok::ampequal:{
      return "&=";
    }
    case tok::star:{
      return "*";
    }
    case tok::starequal:{
      return "*=";
    }
    case tok::plus:{
      return "+";
    }
    case tok::plusplus:{
      return "++";
    }
    case tok::plusequal:{
      return "+=";
    }
    case tok::minus:{
      return "-";
    }
    case tok::minusequal:{
      return "-=";
    }
    case tok::tilde:{
      return "~";
    }
    case tok::exclaim:{
      return "!";
    }
    case tok::exclaimequal:{
      return "!=";
    }
    case tok::slash:{
      return "/";
    }
    case tok::slashequal:{
      return "/=";
    }
    case tok::percent:{
      return "%";
    }
    case tok::percentequal:{
      return "%=";
    }
    case tok::less:{
      return "<";
    }
    case tok::lessless:{
      return "<<";
    }
    case tok::lessequal:{
      return "<=";
    }
    case tok::lesslessequal:{
      return "<<=";
    }
    case tok::greater:{
      return ">";
    }
    case tok::greatergreater:{
      return ">>";
    }
    case tok::greaterequal:{
      return ">=";
    }
    case tok::greatergreaterequal:{
      return ">>=";
    }
    case tok::caret:{
      return "^";
    }
    case tok::caretequal:{
      return "^=";
    }
    case tok::pipe:{
      return "|";
    }
    case tok::pipepipe:{
      return "||";
    }
    case tok::pipeequal:{
      return "|=";
    }
    case tok::question:{
      return "?";
    }
    case tok::colon:{
      return ":";
    }
    case tok::semi:{
      return ";";
    }
    case tok::equal:{
      return "=";
    }
    case tok::equalequal:{
      return "==";
    }
    case tok::comma:{
      return ",";
    }
    case tok::hash:{
      return "#";
    }
    case tok::hashhash:{
      return "##";
    }
    case tok::hashat:{
      return "#@";
    }
    case tok::periodstar:{
      return ".*";
    }
    case tok::arrowstar:{
      return "->*";
    }
    case tok::coloncolon:{
      return "::";
    }
    case tok::at:{
      return "@";
    }
    case tok::lesslessless:{
      return "<<<";
    }
    case tok::greatergreatergreater:{
      return ">>>";
    }
    default:
    {
      return tok.getName();
    }
  }
}

bool Parser::isScoutVectorValueDecl(Decl* decl,
                                    BuiltinType::Kind &kind) const{
  if(ValueDecl* vd = dyn_cast<ValueDecl>(decl)){
    if(const BuiltinType* bt =
       dyn_cast<BuiltinType>(vd->getType().getTypePtr())){

      kind = bt->getKind();

      switch(kind){
        case BuiltinType::Bool2:
        case BuiltinType::Bool3:
        case BuiltinType::Bool4:
        case BuiltinType::Char2:
        case BuiltinType::Char3:
        case BuiltinType::Char4:
        case BuiltinType::Short2:
        case BuiltinType::Short3:
        case BuiltinType::Short4:
        case BuiltinType::Int2:
        case BuiltinType::Int3:
        case BuiltinType::Int4:
        case BuiltinType::Long2:
        case BuiltinType::Long3:
        case BuiltinType::Long4:
        case BuiltinType::Float2:
        case BuiltinType::Float3:
        case BuiltinType::Float4:
        case BuiltinType::Double2:
        case BuiltinType::Double3:
        case BuiltinType::Double4:
        {
          return true;
        }
        default:
          break;
      }
    }
  }
  return false;
}
