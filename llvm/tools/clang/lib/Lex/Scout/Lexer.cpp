/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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

#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

// Enable our keywords only when the file being lexed from is a ".sc", or
// ".sch" file -- this extra check is necessary because we might be including
// a C++ header from a .sc file which would otherwise pick up the Scout
// keyword extensions, potentially causing conflicts...
void Lexer::ScoutEnable(Preprocessor &PP) {
  std::string bufferName = PP.getSourceManager().getBufferName(FileLoc);
  std::string ext;

  //bool valid = false;
  for(int i = bufferName.length() - 1; i >= 0; --i) {
    if (bufferName[i] == '.') {
      //valid = true;
      break;
    }
    ext.insert(0, 1, bufferName[i]);
  }
  // SC_TODO : Why do we need this but Clang doesn't?
  // and why the special case for buffername Parse???
  if (bufferName != "Parse" && ext != "sc" && ext != "sch" &&
      ext != "scpp" && "schpp") {
    LangOpts.ScoutC = false;
    LangOpts.ScoutCPlusPlus = false;
  } else {
    LangOpts.ScoutC         = true;
    LangOpts.ScoutCPlusPlus = true;
  }
}

// If we are lexing from a non-Scout file, then we need to treat Scout
// keywords as ordinary identifiers…
void Lexer::ScoutKeywordsAsIdentifiers(Token &Result) {
  IdentifierInfo* NII = 0;
  switch(Result.getKind()) {
  #define SCOUT_KEYWORD(X) case tok::kw_##X: \
    NII = PP->getScoutIdentifier(#X);        \
    break;
  #include "clang/Basic/TokenKinds.def"
  default:
    break;
  }
  if (NII) {
    Result.setIdentifierInfo(NII);
    Result.setKind(tok::identifier);
  }
}

