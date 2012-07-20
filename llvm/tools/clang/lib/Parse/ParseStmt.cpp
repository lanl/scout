//===--- ParseStmt.cpp - Statement and Block Parser -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Statement and Block portions of the Parser
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/SmallString.h"

// scout
#include "clang/AST/ASTContext.h"
#include <iostream>

using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

/// ParseStatementOrDeclaration - Read 'statement' or 'declaration'.
///       StatementOrDeclaration:
///         statement
///         declaration
///
///       statement:
///         labeled-statement
///         compound-statement
///         expression-statement
///         selection-statement
///         iteration-statement
///         jump-statement
/// [C++]   declaration-statement
/// [C++]   try-block
/// [MS]    seh-try-block
/// [OBC]   objc-throw-statement
/// [OBC]   objc-try-catch-statement
/// [OBC]   objc-synchronized-statement
/// [GNU]   asm-statement
/// [OMP]   openmp-construct             [TODO]
///
///       labeled-statement:
///         identifier ':' statement
///         'case' constant-expression ':' statement
///         'default' ':' statement
///
///       selection-statement:
///         if-statement
///         switch-statement
///
///       iteration-statement:
///         while-statement
///         do-statement
///         for-statement
///
///       expression-statement:
///         expression[opt] ';'
///
///       jump-statement:
///         'goto' identifier ';'
///         'continue' ';'
///         'break' ';'
///         'return' expression[opt] ';'
/// [GNU]   'goto' '*' expression ';'
///
/// [OBC] objc-throw-statement:
/// [OBC]   '@' 'throw' expression ';'
/// [OBC]   '@' 'throw' ';'
///
StmtResult
Parser::ParseStatementOrDeclaration(StmtVector &Stmts, bool OnlyStatement,
                                    SourceLocation *TrailingElseLoc) {
  // scout - test hook into parsing stmts from the main file
  if(Actions.SourceMgr.isFromMainFile(Tok.getLocation())){
    //DumpLookAheads(20);
  }

  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  ParsedAttributesWithRange Attrs(AttrFactory);
  MaybeParseCXX0XAttributes(Attrs, 0, /*MightBeObjCMessageSend*/ true);

  StmtResult Res = ParseStatementOrDeclarationAfterAttributes(Stmts,
                                 OnlyStatement, TrailingElseLoc, Attrs);

  assert((Attrs.empty() || Res.isInvalid() || Res.isUsable()) &&
         "attributes on empty statement");

  if (Attrs.empty() || Res.isInvalid())
    return Res;

  return Actions.ProcessStmtAttributes(Res.get(), Attrs.getList(), Attrs.Range);
}

StmtResult
Parser::ParseStatementOrDeclarationAfterAttributes(StmtVector &Stmts,
          bool OnlyStatement, SourceLocation *TrailingElseLoc,
          ParsedAttributesWithRange &Attrs) {
  const char *SemiError = 0;
  StmtResult Res;

  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
Retry:
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::at: // May be a @try or @throw statement
    {
      ProhibitAttributes(Attrs); // TODO: is it correct?
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::code_completion:
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Statement);
    cutOffParsing();
    return StmtError();

  case tok::identifier: {
    Token Next = NextToken();
    if (Next.is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement(Attrs);
    }

    if (Next.isNot(tok::coloncolon)) {
      CXXScopeSpec SS;
      IdentifierInfo *Name = Tok.getIdentifierInfo();
      SourceLocation NameLoc = Tok.getLocation();

      if (getLangOpts().CPlusPlus)
        CheckForTemplateAndDigraph(Next, ParsedType(),
                                   /*EnteringContext=*/false, *Name, SS);

      Sema::NameClassification Classification
        = Actions.ClassifyName(getCurScope(), SS, Name, NameLoc, Next);
      switch (Classification.getKind()) {
      case Sema::NC_Keyword:
        // The identifier was corrected to a keyword. Update the token
        // to this keyword, and try again.
        if (Name->getTokenID() != tok::identifier) {
          Tok.setIdentifierInfo(Name);
          Tok.setKind(Name->getTokenID());
          goto Retry;
        }

        // Fall through via the normal error path.
        // FIXME: This seems like it could only happen for context-sensitive
        // keywords.

      case Sema::NC_Error:
        // Handle errors here by skipping up to the next semicolon or '}', and
        // eat the semicolon if that's what stopped us.
        SkipUntil(tok::r_brace, /*StopAtSemi=*/true, /*DontConsume=*/true);
        if (Tok.is(tok::semi))
          ConsumeToken();
        return StmtError();

      case Sema::NC_Unknown:
        // Either we don't know anything about this identifier, or we know that
        // we're in a syntactic context we haven't handled yet.
        break;

      case Sema::NC_Type:
      {
        // scout - parse a mesh declaration -this is handled as a special
        // case because the square brackets look like an array specification
        // when Clang normally parses a declaration
        
        if(isScoutLang() && Actions.isScoutSource(Tok.getLocation())){
          QualType qt = Sema::GetTypeFromParser(Classification.getType());
          
          if(qt->getAs<MeshType>() &&
             GetLookAheadToken(1).is(tok::identifier)){
            
            if(GetLookAheadToken(2).is(tok::l_square)){
              
              std::string meshType = TokToStr(Tok);
              ConsumeToken();
              std::string meshName = TokToStr(Tok);
              ConsumeToken();
              
              // parse mesh dimensions, e.g: [512,512]
              
              MeshType::MeshDimensionVec dims;
              
              ConsumeBracket();
              
              for(;;){
                
                if(Tok.isNot(tok::numeric_constant)){
                  Diag(Tok, diag::err_expected_numeric_constant_in_mesh_def);
                  SkipUntil(tok::r_square);
                  SkipUntil(tok::semi);
                  return StmtError();
                }
                
                dims.push_back(Actions.ActOnNumericConstant(Tok).get());
                
                ConsumeToken();
                
                if(Tok.is(tok::r_square)){
                  break;
                }
                
                if(Tok.is(tok::eof)){
                  Diag(Tok, diag::err_expected_lsquare);
                  return StmtError();
                }
                
                if(Tok.isNot(tok::comma)){
                  Diag(Tok, diag::err_expected_comma);
                  SkipUntil(tok::r_square);
                  SkipUntil(tok::semi);
                  return StmtError();
                }
                
                ConsumeToken();
              }
              
              ConsumeBracket();
              
              InsertCPPCode(meshType + " " + meshName, NameLoc);

              DeclaringMesh = true;
              StmtResult result = ParseStatementOrDeclaration(Stmts, OnlyStatement);
              
              DeclaringMesh = false;
              
              DeclStmt* ds = dyn_cast<DeclStmt>(result.get());
              assert(ds->isSingleDecl());
              
              VarDecl* vd = dyn_cast<VarDecl>(ds->getSingleDecl());
              assert(vd);
              
              const MeshType* mt = 
              dyn_cast<MeshType>(vd->getType().getCanonicalType().getTypePtr());
              assert(mt);
              
              MeshType* mdt = new MeshType(mt->getDecl());
              mdt->setDimensions(dims);
              vd->setType(QualType(mdt, 0));
              
              return result;
            }
            else if(!DeclaringMesh){
              Diag(Tok, diag::err_expected_lsquare);
              
              SkipUntil(tok::r_square);
              SkipUntil(tok::semi);
              return StmtError();
            }
          }
        }

        Tok.setKind(tok::annot_typename);
        setTypeAnnotation(Tok, Classification.getType());
        Tok.setAnnotationEndLoc(NameLoc);
        PP.AnnotateCachedTokens(Tok);
        
        break;
      }

      case Sema::NC_Expression:
        Tok.setKind(tok::annot_primary_expr);
        setExprAnnotation(Tok, Classification.getExpression());
        Tok.setAnnotationEndLoc(NameLoc);
        PP.AnnotateCachedTokens(Tok);
        break;

      case Sema::NC_TypeTemplate:
      case Sema::NC_FunctionTemplate: {
        ConsumeToken(); // the identifier
        UnqualifiedId Id;
        Id.setIdentifier(Name, NameLoc);
        if (AnnotateTemplateIdToken(
                            TemplateTy::make(Classification.getTemplateName()),
                                    Classification.getTemplateNameKind(),
                                    SS, SourceLocation(), Id,
                                    /*AllowTypeAnnotation=*/false)) {
          // Handle errors here by skipping up to the next semicolon or '}', and
          // eat the semicolon if that's what stopped us.
          SkipUntil(tok::r_brace, /*StopAtSemi=*/true, /*DontConsume=*/true);
          if (Tok.is(tok::semi))
            ConsumeToken();
          return StmtError();
        }

        // If the next token is '::', jump right into parsing a
        // nested-name-specifier. We don't want to leave the template-id
        // hanging.
        if (NextToken().is(tok::coloncolon) && TryAnnotateCXXScopeToken(false)){
          // Handle errors here by skipping up to the next semicolon or '}', and
          // eat the semicolon if that's what stopped us.
          SkipUntil(tok::r_brace, /*StopAtSemi=*/true, /*DontConsume=*/true);
          if (Tok.is(tok::semi))
            ConsumeToken();
          return StmtError();
        }

        // We've annotated a template-id, so try again now.
        goto Retry;
      }

      case Sema::NC_NestedNameSpecifier:
        // FIXME: Implement this!
        break;
      }

      // scout - detect the forall shorthand, e.g:
      // m.a[1..width-2][1..height-2] = MAX_TEMP;
      if(isScoutLang() && Actions.isScoutSource(NameLoc)){
        if(GetLookAheadToken(1).is(tok::period) &&
           GetLookAheadToken(2).is(tok::identifier) &&
           GetLookAheadToken(3).is(tok::l_square)){

          LookupResult
          Result(Actions, Name, NameLoc, Sema::LookupOrdinaryName);

          Actions.LookupName(Result, getCurScope());

          if(Result.getResultKind() == LookupResult::Found){
            if(VarDecl* vd = dyn_cast<VarDecl>(Result.getFoundDecl())){
              if(isa<MeshType>(vd->getType().getCanonicalType().getTypePtr())){
                return ParseForAllShortStatement(Name, NameLoc, vd);
              }
            }
          }
        }
      }
    }

    // Fall through
  }

  default: {
    if ((getLangOpts().CPlusPlus || !OnlyStatement) && isDeclarationStatement()) {
      SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
      DeclGroupPtrTy Decl = ParseDeclaration(Stmts, Declarator::BlockContext,
                                             DeclEnd, Attrs);

      // scout - test
      //StmtResult r = Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
      //r.get()->dump();
      //return r;
      return Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
    }

    if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return StmtError();
    }

    return ParseExprStatement();
  }

  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement();
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement();

  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement();
  case tok::semi: {                 // C99 6.8.3p3: expression[opt] ';'
    bool HasLeadingEmptyMacro = Tok.hasLeadingEmptyMacro();
    return Actions.ActOnNullStmt(ConsumeToken(), HasLeadingEmptyMacro);
  }

  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement(TrailingElseLoc);
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement(TrailingElseLoc);

  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement(TrailingElseLoc);
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement();
    SemiError = "do/while";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement(TrailingElseLoc);

  // scout - Stmts
  case tok::kw_forall: {
    const Token& t = GetLookAheadToken(1);
    switch(t.getKind()){
      case tok::kw_cells:
      case tok::kw_vertices:
      case tok::kw_faces:
      case tok::kw_edges:
        return ParseForAllStatement(Attrs);
      default:
        return ParseForAllArrayStatement(Attrs);
    }
  }

  case tok::kw_renderall:
    return ParseForAllStatement(Attrs, false);
  case tok::kw_window:
      return ParseWindowOrImageDeclaration(true, Stmts, OnlyStatement);
  case tok::kw_image:
    return ParseWindowOrImageDeclaration(false, Stmts, OnlyStatement);
  case tok::kw_camera:
    return ParseCameraDeclaration(Stmts, OnlyStatement);
    
  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement();
    SemiError = "goto";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement();
    SemiError = "continue";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement();
    SemiError = "break";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement();
    SemiError = "return";
    break;

  case tok::kw_asm: {
    ProhibitAttributes(Attrs);
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    Res = Actions.ActOnFinishFullStmt(Res.get());
    if (msAsm) return move(Res);
    SemiError = "asm";
    break;
  }

  case tok::kw_try:                 // C++ 15: try-block
    return ParseCXXTryBlock();

  case tok::kw___try:
    ProhibitAttributes(Attrs); // TODO: is it correct?
    return ParseSEHTryBlock();

  case tok::annot_pragma_vis:
    ProhibitAttributes(Attrs);
    HandlePragmaVisibility();
    return StmtEmpty();

  case tok::annot_pragma_pack:
    ProhibitAttributes(Attrs);
    HandlePragmaPack();
    return StmtEmpty();
  }

  // If we reached this code, the statement must end in a semicolon.
  if (Tok.is(tok::semi)) {
    ConsumeToken();
  } else if (!Res.isInvalid()) {
    // If the result was valid, then we do want to diagnose this.  Use
    // ExpectAndConsume to emit the diagnostic, even though we know it won't
    // succeed.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_after_stmt, SemiError);
    // Skip until we see a } or ;, but don't eat it.
    SkipUntil(tok::r_brace, true, true);
  }

  return move(Res);
}

/// \brief Parse an expression statement.
StmtResult Parser::ParseExprStatement() {
  // If a case keyword is missing, this is where it should be inserted.
  Token OldToken = Tok;

  // expression[opt] ';'
  ExprResult Expr(ParseExpression());
  if (Expr.isInvalid()) {
    // If the expression is invalid, skip ahead to the next semicolon or '}'.
    // Not doing this opens us up to the possibility of infinite loops if
    // ParseExpression does not consume any tokens.
    SkipUntil(tok::r_brace, /*StopAtSemi=*/true, /*DontConsume=*/true);
    if (Tok.is(tok::semi))
      ConsumeToken();
    return StmtError();
  }

  if (Tok.is(tok::colon) && getCurScope()->isSwitchScope() &&
      Actions.CheckCaseExpression(Expr.get())) {
    // If a constant expression is followed by a colon inside a switch block,
    // suggest a missing case keyword.
    Diag(OldToken, diag::err_expected_case_before_expression)
      << FixItHint::CreateInsertion(OldToken.getLocation(), "case ");

    // Recover parsing as a case statement.
    return ParseCaseStatement(/*MissingCase=*/true, Expr);
  }

  // Otherwise, eat the semicolon.
  ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
  return Actions.ActOnExprStmt(Actions.MakeFullExpr(Expr.get()));
}

StmtResult Parser::ParseSEHTryBlock() {
  assert(Tok.is(tok::kw___try) && "Expected '__try'");
  SourceLocation Loc = ConsumeToken();
  return ParseSEHTryBlockCommon(Loc);
}

/// ParseSEHTryBlockCommon
///
/// seh-try-block:
///   '__try' compound-statement seh-handler
///
/// seh-handler:
///   seh-except-block
///   seh-finally-block
///
StmtResult Parser::ParseSEHTryBlockCommon(SourceLocation TryLoc) {
  if(Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok,diag::err_expected_lbrace));

  StmtResult TryBlock(ParseCompoundStatement());
  if(TryBlock.isInvalid())
    return move(TryBlock);

  StmtResult Handler;
  if (Tok.is(tok::identifier) &&
      Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHExceptBlock(Loc);
  } else if (Tok.is(tok::kw___finally)) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHFinallyBlock(Loc);
  } else {
    return StmtError(Diag(Tok,diag::err_seh_expected_handler));
  }

  if(Handler.isInvalid())
    return move(Handler);

  return Actions.ActOnSEHTryBlock(false /* IsCXXTry */,
                                  TryLoc,
                                  TryBlock.take(),
                                  Handler.take());
}

/// ParseSEHExceptBlock - Handle __except
///
/// seh-except-block:
///   '__except' '(' seh-filter-expression ')' compound-statement
///
StmtResult Parser::ParseSEHExceptBlock(SourceLocation ExceptLoc) {
  PoisonIdentifierRAIIObject raii(Ident__exception_code, false),
    raii2(Ident___exception_code, false),
    raii3(Ident_GetExceptionCode, false);

  if(ExpectAndConsume(tok::l_paren,diag::err_expected_lparen))
    return StmtError();

  ParseScope ExpectScope(this, Scope::DeclScope | Scope::ControlScope);

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(false);
    Ident___exception_info->setIsPoisoned(false);
    Ident_GetExceptionInfo->setIsPoisoned(false);
  }
  ExprResult FilterExpr(ParseExpression());

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(true);
    Ident___exception_info->setIsPoisoned(true);
    Ident_GetExceptionInfo->setIsPoisoned(true);
  }

  if(FilterExpr.isInvalid())
    return StmtError();

  if(ExpectAndConsume(tok::r_paren,diag::err_expected_rparen))
    return StmtError();

  StmtResult Block(ParseCompoundStatement());

  if(Block.isInvalid())
    return move(Block);

  return Actions.ActOnSEHExceptBlock(ExceptLoc, FilterExpr.take(), Block.take());
}

/// ParseSEHFinallyBlock - Handle __finally
///
/// seh-finally-block:
///   '__finally' compound-statement
///
StmtResult Parser::ParseSEHFinallyBlock(SourceLocation FinallyBlock) {
  PoisonIdentifierRAIIObject raii(Ident__abnormal_termination, false),
    raii2(Ident___abnormal_termination, false),
    raii3(Ident_AbnormalTermination, false);

  StmtResult Block(ParseCompoundStatement());
  if(Block.isInvalid())
    return move(Block);

  return Actions.ActOnSEHFinallyBlock(FinallyBlock,Block.take());
}

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
StmtResult Parser::ParseLabeledStatement(ParsedAttributesWithRange &attrs) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");

  // identifier ':' statement
  SourceLocation ColonLoc = ConsumeToken();

  // Read label attributes, if present. attrs will contain both C++11 and GNU
  // attributes (if present) after this point.
  MaybeParseGNUAttributes(attrs);

  StmtResult SubStmt(ParseStatement());

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  LabelDecl *LD = Actions.LookupOrCreateLabel(IdentTok.getIdentifierInfo(),
                                              IdentTok.getLocation());
  if (AttributeList *Attrs = attrs.getList()) {
    Actions.ProcessDeclAttributeList(Actions.CurScope, LD, Attrs);
    attrs.clear();
  }

  return Actions.ActOnLabelStmt(IdentTok.getLocation(), LD, ColonLoc,
                                SubStmt.get());
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
StmtResult Parser::ParseCaseStatement(bool MissingCase, ExprResult Expr) {
  assert((MissingCase || Tok.is(tok::kw_case)) && "Not a case stmt!");

  // It is very very common for code to contain many case statements recursively
  // nested, as in (but usually without indentation):
  //  case 1:
  //    case 2:
  //      case 3:
  //         case 4:
  //           case 5: etc.
  //
  // Parsing this naively works, but is both inefficient and can cause us to run
  // out of stack space in our recursive descent parser.  As a special case,
  // flatten this recursion into an iterative loop.  This is complex and gross,
  // but all the grossness is constrained to ParseCaseStatement (and some
  // wierdness in the actions), so this is just local grossness :).

  // TopLevelCase - This is the highest level we have parsed.  'case 1' in the
  // example above.
  StmtResult TopLevelCase(true);

  // DeepestParsedCaseStmt - This is the deepest statement we have parsed, which
  // gets updated each time a new case is parsed, and whose body is unset so
  // far.  When parsing 'case 4', this is the 'case 3' node.
  Stmt *DeepestParsedCaseStmt = 0;

  // While we have case statements, eat and stack them.
  SourceLocation ColonLoc;
  do {
    SourceLocation CaseLoc = MissingCase ? Expr.get()->getExprLoc() :
                                           ConsumeToken();  // eat the 'case'.

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteCase(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    /// We don't want to treat 'case x : y' as a potential typo for 'case x::y'.
    /// Disable this form of error recovery while we're parsing the case
    /// expression.
    ColonProtectionRAIIObject ColonProtection(*this);

    ExprResult LHS(MissingCase ? Expr : ParseConstantExpression());
    MissingCase = false;
    if (LHS.isInvalid()) {
      SkipUntil(tok::colon);
      return StmtError();
    }

    // GNU case range extension.
    SourceLocation DotDotDotLoc;
    ExprResult RHS;
    if (Tok.is(tok::ellipsis)) {
      Diag(Tok, diag::ext_gnu_case_range);
      DotDotDotLoc = ConsumeToken();

      RHS = ParseConstantExpression();
      if (RHS.isInvalid()) {
        SkipUntil(tok::colon);
        return StmtError();
      }
    }

    ColonProtection.restore();

    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();

    // Treat "case blah;" as a typo for "case blah:".
    } else if (Tok.is(tok::semi)) {
      ColonLoc = ConsumeToken();
      Diag(ColonLoc, diag::err_expected_colon_after) << "'case'"
        << FixItHint::CreateReplacement(ColonLoc, ":");
    } else {
      SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
      Diag(ExpectedLoc, diag::err_expected_colon_after) << "'case'"
        << FixItHint::CreateInsertion(ExpectedLoc, ":");
      ColonLoc = ExpectedLoc;
    }

    StmtResult Case =
      Actions.ActOnCaseStmt(CaseLoc, LHS.get(), DotDotDotLoc,
                            RHS.get(), ColonLoc);

    // If we had a sema error parsing this case, then just ignore it and
    // continue parsing the sub-stmt.
    if (Case.isInvalid()) {
      if (TopLevelCase.isInvalid())  // No parsed case stmts.
        return ParseStatement();
      // Otherwise, just don't add it as a nested case.
    } else {
      // If this is the first case statement we parsed, it becomes TopLevelCase.
      // Otherwise we link it into the current chain.
      Stmt *NextDeepest = Case.get();
      if (TopLevelCase.isInvalid())
        TopLevelCase = move(Case);
      else
        Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, Case.get());
      DeepestParsedCaseStmt = NextDeepest;
    }

    // Handle all case statements.
  } while (Tok.is(tok::kw_case));

  assert(!TopLevelCase.isInvalid() && "Should have parsed at least one case!");

  // If we found a non-case statement, start by parsing it.
  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Nicely diagnose the common error "switch (X) { case 4: }", which is
    // not valid.
    SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
    Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
      << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(SourceLocation());

  // Install the body into the most deeply-nested case.
  Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, SubStmt.get());

  // Return the top level parsed statement tree.
  return move(TopLevelCase);
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
StmtResult Parser::ParseDefaultStatement() {
  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  SourceLocation ColonLoc;
  if (Tok.is(tok::colon)) {
    ColonLoc = ConsumeToken();

  // Treat "default;" as a typo for "default:".
  } else if (Tok.is(tok::semi)) {
    ColonLoc = ConsumeToken();
    Diag(ColonLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateReplacement(ColonLoc, ":");
  } else {
    SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
    Diag(ExpectedLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateInsertion(ExpectedLoc, ":");
    ColonLoc = ExpectedLoc;
  }

  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Diagnose the common error "switch (X) {... default: }", which is
    // not valid.
    SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
    Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
      << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc,
                                  SubStmt.get(), getCurScope());
}

StmtResult Parser::ParseCompoundStatement(bool isStmtExpr) {
  return ParseCompoundStatement(isStmtExpr, Scope::DeclScope);
}

/// ParseCompoundStatement - Parse a "{}" block.
///
///       compound-statement: [C99 6.8.2]
///         { block-item-list[opt] }
/// [GNU]   { label-declarations block-item-list } [TODO]
///
///       block-item-list:
///         block-item
///         block-item-list block-item
///
///       block-item:
///         declaration
/// [GNU]   '__extension__' declaration
///         statement
/// [OMP]   openmp-directive            [TODO]
///
/// [GNU] label-declarations:
/// [GNU]   label-declaration
/// [GNU]   label-declarations label-declaration
///
/// [GNU] label-declaration:
/// [GNU]   '__label__' identifier-list ';'
///
/// [OMP] openmp-directive:             [TODO]
/// [OMP]   barrier-directive
/// [OMP]   flush-directive
///
StmtResult Parser::ParseCompoundStatement(bool isStmtExpr,
                                          unsigned ScopeFlags) {
  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");

  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope CompoundScope(this, ScopeFlags);

  // Parse the statements in the body.
  return ParseCompoundStatementBody(isStmtExpr);
}

/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
StmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
                                Tok.getLocation(),
                                "in compound statement ('{}')");
  InMessageExpressionRAIIObject InMessage(*this, false);
  BalancedDelimiterTracker T(*this, tok::l_brace);
  if (T.consumeOpen())
    return StmtError();

  Sema::CompoundScopeRAII CompoundScope(Actions);

  StmtVector Stmts(Actions);

  // "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.
  while (Tok.is(tok::kw___label__)) {
    SourceLocation LabelLoc = ConsumeToken();
    Diag(LabelLoc, diag::ext_gnu_local_label);

    SmallVector<Decl *, 8> DeclsInGroup;
    while (1) {
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      SourceLocation IdLoc = ConsumeToken();
      DeclsInGroup.push_back(Actions.LookupOrCreateLabel(II, IdLoc, LabelLoc));

      if (!Tok.is(tok::comma))
        break;
      ConsumeToken();
    }

    DeclSpec DS(AttrFactory);
    DeclGroupPtrTy Res = Actions.FinalizeDeclaratorGroup(getCurScope(), DS,
                                      DeclsInGroup.data(), DeclsInGroup.size());
    StmtResult R = Actions.ActOnDeclStmt(Res, LabelLoc, Tok.getLocation());

    ExpectAndConsumeSemi(diag::err_expected_semi_declaration);
    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    if (Tok.is(tok::annot_pragma_unused)) {
      HandlePragmaUnused();
      continue;
    }

    if (getLangOpts().MicrosoftExt && (Tok.is(tok::kw___if_exists) ||
        Tok.is(tok::kw___if_not_exists))) {
      ParseMicrosoftIfExistsStatement(Stmts);
      continue;
    }

    StmtResult R;
    if (Tok.isNot(tok::kw___extension__)) {
      // scout - store the stmt vec so we can insert statements into it
      // when Stmts is not available as a parameter
      StmtsStack.push_back(&Stmts);

      R = ParseStatementOrDeclaration(Stmts, false);
      // scout
      StmtsStack.pop_back();
    } else {
      // __extension__ can start declarations and it can also be a unary
      // operator for expressions.  Consume multiple __extension__ markers here
      // until we can determine which is which.
      // FIXME: This loses extension expressions in the AST!
      SourceLocation ExtLoc = ConsumeToken();
      while (Tok.is(tok::kw___extension__))
        ConsumeToken();

      ParsedAttributesWithRange attrs(AttrFactory);
      MaybeParseCXX0XAttributes(attrs, 0, /*MightBeObjCMessageSend*/ true);

      // If this is the start of a declaration, parse it as such.
      if (isDeclarationStatement()) {
        // __extension__ silences extension warnings in the subdeclaration.
        // FIXME: Save the __extension__ on the decl as a node somehow?
        ExtensionRAIIObject O(Diags);

        SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
        DeclGroupPtrTy Res = ParseDeclaration(Stmts,
                                              Declarator::BlockContext, DeclEnd,
                                              attrs);
        R = Actions.ActOnDeclStmt(Res, DeclStart, DeclEnd);
      } else {
        // Otherwise this was a unary __extension__ marker.
        ExprResult Res(ParseExpressionWithLeadingExtension(ExtLoc));

        if (Res.isInvalid()) {
          SkipUntil(tok::semi);
          continue;
        }

        // FIXME: Use attributes?
        // Eat the semicolon at the end of stmt and convert the expr into a
        // statement.
        ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
        R = Actions.ActOnExprStmt(Actions.MakeFullExpr(Res.get()));
      }
    }

    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  SourceLocation CloseLoc = Tok.getLocation();

  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.isNot(tok::r_brace)) {
    Diag(Tok, diag::err_expected_rbrace);
    Diag(T.getOpenLocation(), diag::note_matching) << "{";
    // Recover by creating a compound statement with what we parsed so far,
    // instead of dropping everything and returning StmtError();
  } else {
    if (!T.consumeClose())
      CloseLoc = T.getCloseLocation();
  }

  return Actions.ActOnCompoundStmt(T.getOpenLocation(), CloseLoc,
                                   move_arg(Stmts), isStmtExpr);
}

/// ParseParenExprOrCondition:
/// [C  ]     '(' expression ')'
/// [C++]     '(' condition ')'       [not allowed if OnlyAllowCondition=true]
///
/// This function parses and performs error recovery on the specified condition
/// or expression (depending on whether we're in C++ or C mode).  This function
/// goes out of its way to recover well.  It returns true if there was a parser
/// error (the right paren couldn't be found), which indicates that the caller
/// should try to recover harder.  It returns false if the condition is
/// successfully parsed.  Note that a successful parse can still have semantic
/// errors in the condition.
bool Parser::ParseParenExprOrCondition(ExprResult &ExprResult,
                                       Decl *&DeclResult,
                                       SourceLocation Loc,
                                       bool ConvertToBoolean) {
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  if (getLangOpts().CPlusPlus)
    ParseCXXCondition(ExprResult, DeclResult, Loc, ConvertToBoolean);
  else {
    ExprResult = ParseExpression();
    DeclResult = 0;

    // If required, convert to a boolean value.
    if (!ExprResult.isInvalid() && ConvertToBoolean)
      ExprResult
        = Actions.ActOnBooleanCondition(getCurScope(), Loc, ExprResult.get());
  }

  // If the parser was confused by the condition and we don't have a ')', try to
  // recover by skipping ahead to a semi and bailing out.  If condexp is
  // semantically invalid but we have well formed code, keep going.
  if (ExprResult.isInvalid() && !DeclResult && Tok.isNot(tok::r_paren)) {
    SkipUntil(tok::semi);
    // Skipping may have stopped if it found the containing ')'.  If so, we can
    // continue parsing the if statement.
    if (Tok.isNot(tok::r_paren))
      return true;
  }

  // Otherwise the condition is valid or the rparen is present.
  T.consumeClose();

  // Check for extraneous ')'s to catch things like "if (foo())) {".  We know
  // that all callers are looking for a statement after the condition, so ")"
  // isn't valid.
  while (Tok.is(tok::r_paren)) {
    Diag(Tok, diag::err_extraneous_rparen_in_condition)
      << FixItHint::CreateRemoval(Tok.getLocation());
    ConsumeParen();
  }

  return false;
}


/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
StmtResult Parser::ParseIfStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "if";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the if statement is a block.  This is not
  // the case for C90.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  ParseScope IfScope(this, Scope::DeclScope | Scope::ControlScope, C99orCXX);

  // Parse the condition.
  ExprResult CondExp;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(CondExp, CondVar, IfLoc, true))
    return StmtError();

  FullExprArg FullCondExp(Actions.MakeFullExpr(CondExp.get(), IfLoc));

  // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // For C++ we create a scope for the condition and a new scope for
  // substatements because:
  // -When the 'then' scope exits, we want the condition declaration to still be
  //    active for the 'else' scope too.
  // -Sema will detect name clashes by considering declarations of a
  //    'ControlScope' as part of its direct subscope.
  // -If we wanted the condition and substatement to be in the same scope, we
  //    would have to notify ParseStatement not to create a new scope. It's
  //    simpler to let it create a new scope.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the 'then' stmt.
  SourceLocation ThenStmtLoc = Tok.getLocation();

  SourceLocation InnerStatementTrailingElseLoc;
  StmtResult ThenStmt(ParseStatement(&InnerStatementTrailingElseLoc));

  // Pop the 'if' scope if needed.
  InnerScope.Exit();

  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  StmtResult ElseStmt;

  if (Tok.is(tok::kw_else)) {
    if (TrailingElseLoc)
      *TrailingElseLoc = Tok.getLocation();

    ElseLoc = ConsumeToken();
    ElseStmtLoc = Tok.getLocation();

    // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
    // there is no compound stmt.  C90 does not have this clause.  We only do
    // this if the body isn't a compound statement to avoid push/pop in common
    // cases.
    //
    // C++ 6.4p1:
    // The substatement in a selection-statement (each substatement, in the else
    // form of the if statement) implicitly defines a local scope.
    //
    ParseScope InnerScope(this, Scope::DeclScope,
                          C99orCXX && Tok.isNot(tok::l_brace));

    ElseStmt = ParseStatement();

    // Pop the 'else' scope if needed.
    InnerScope.Exit();
  } else if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteAfterIf(getCurScope());
    cutOffParsing();
    return StmtError();
  } else if (InnerStatementTrailingElseLoc.isValid()) {
    Diag(InnerStatementTrailingElseLoc, diag::warn_dangling_else);
  }

  IfScope.Exit();

  // If the condition was invalid, discard the if statement.  We could recover
  // better by replacing it with a valid expr, but don't do that yet.
  if (CondExp.isInvalid() && !CondVar)
    return StmtError();

  // If the then or else stmt is invalid and the other is valid (and present),
  // make turn the invalid one into a null stmt to avoid dropping the other
  // part.  If both are invalid, return error.
  if ((ThenStmt.isInvalid() && ElseStmt.isInvalid()) ||
      (ThenStmt.isInvalid() && ElseStmt.get() == 0) ||
      (ThenStmt.get() == 0  && ElseStmt.isInvalid())) {
    // Both invalid, or one is invalid and other is non-present: return error.
    return StmtError();
  }

  // Now if either are invalid, replace with a ';'.
  if (ThenStmt.isInvalid())
    ThenStmt = Actions.ActOnNullStmt(ThenStmtLoc);
  if (ElseStmt.isInvalid())
    ElseStmt = Actions.ActOnNullStmt(ElseStmtLoc);

  return Actions.ActOnIfStmt(IfLoc, FullCondExp, CondVar, ThenStmt.get(),
                             ElseLoc, ElseStmt.get());
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
/// [C++]   'switch' '(' condition ')' statement
StmtResult Parser::ParseSwitchStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "switch";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the switch statement is a block.  This is
  // not the case for C90.  Start the switch scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags = Scope::BreakScope | Scope::SwitchScope;
  if (C99orCXX)
    ScopeFlags |= Scope::DeclScope | Scope::ControlScope;
  ParseScope SwitchScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(Cond, CondVar, SwitchLoc, false))
    return StmtError();

  StmtResult Switch
    = Actions.ActOnStartOfSwitchStmt(SwitchLoc, Cond.get(), CondVar);

  if (Switch.isInvalid()) {
    // Skip the switch body.
    // FIXME: This is not optimal recovery, but parsing the body is more
    // dangerous due to the presence of case and default statements, which
    // will have no place to connect back with the switch.
    if (Tok.is(tok::l_brace)) {
      ConsumeBrace();
      SkipUntil(tok::r_brace, false, false);
    } else
      SkipUntil(tok::semi);
    return move(Switch);
  }

  // C99 6.8.4p3 - In C99, the body of the switch statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the scopes.
  InnerScope.Exit();
  SwitchScope.Exit();

  if (Body.isInvalid()) {
    // FIXME: Remove the case statement list from the Switch statement.

    // Put the synthesized null statement on the same line as the end of switch
    // condition.
    SourceLocation SynthesizedNullStmtLocation = Cond.get()->getLocEnd();
    Body = Actions.ActOnNullStmt(SynthesizedNullStmtLocation);
  }

  return Actions.ActOnFinishSwitchStmt(SwitchLoc, Switch.get(), Body.get());
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
/// [C++]   'while' '(' condition ')' statement
StmtResult Parser::ParseWhileStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "while";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.5p5 - In C99, the while statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags;
  if (C99orCXX)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;
  ParseScope WhileScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(Cond, CondVar, WhileLoc, true))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond.get(), WhileLoc));

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();
  WhileScope.Exit();

  if ((Cond.isInvalid() && !CondVar) || Body.isInvalid())
    return StmtError();

  return Actions.ActOnWhileStmt(WhileLoc, FullCond, CondVar, Body.get());
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
StmtResult Parser::ParseDoStatement() {
  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.

  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  unsigned ScopeFlags;
  if (getLangOpts().C99)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope DoScope(this, ScopeFlags);

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause. We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        (getLangOpts().C99 || getLangOpts().CPlusPlus) &&
                        Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Tok.isNot(tok::kw_while)) {
    if (!Body.isInvalid()) {
      Diag(Tok, diag::err_expected_while);
      Diag(DoLoc, diag::note_matching) << "do";
      SkipUntil(tok::semi, false, true);
    }
    return StmtError();
  }
  SourceLocation WhileLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "do/while";
    SkipUntil(tok::semi, false, true);
    return StmtError();
  }

  // Parse the parenthesized condition.
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  // FIXME: Do not just parse the attribute contents and throw them away
  ParsedAttributesWithRange attrs(AttrFactory);
  MaybeParseCXX0XAttributes(attrs);
  ProhibitAttributes(attrs);

  ExprResult Cond = ParseExpression();
  T.consumeClose();
  DoScope.Exit();

  if (Cond.isInvalid() || Body.isInvalid())
    return StmtError();

  return Actions.ActOnDoStmt(DoLoc, Body.get(), WhileLoc, T.getOpenLocation(),
                             Cond.get(), T.getCloseLocation());
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [C++0x] 'for' '(' for-range-declaration : for-range-initializer ) statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
/// [C++0x] for-range-declaration:
/// [C++0x]   attribute-specifier-seq[opt] type-specifier-seq declarator
/// [C++0x] for-range-initializer:
/// [C++0x]   expression
/// [C++0x]   braced-init-list            [TODO]
StmtResult Parser::ParseForStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLangOpts().C99 || getLangOpts().CPlusPlus ||
    getLangOpts().ObjC1;

  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  // C++ 6.5.3p1:
  // Names declared in the for-init-statement are in the same declarative-region
  // as those declared in the condition.
  //
  unsigned ScopeFlags;
  if (C99orCXXorObjC)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope ForScope(this, ScopeFlags);

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult Value;

  bool ForEach = false, ForRange = false;
  StmtResult FirstPart;
  bool SecondPartIsInvalid = false;
  FullExprArg SecondPart(Actions);
  ExprResult Collection;
  ForRangeInit ForRangeInit;
  FullExprArg ThirdPart(Actions);
  Decl *SecondVar = 0;

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(),
                                     C99orCXXorObjC? Sema::PCC_ForInit
                                                   : Sema::PCC_Expression);
    cutOffParsing();
    return StmtError();
  }

  ParsedAttributesWithRange attrs(AttrFactory);
  MaybeParseCXX0XAttributes(attrs);

  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    ProhibitAttributes(attrs);
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isForInitDeclaration()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);

    ParsedAttributesWithRange attrs(AttrFactory);
    MaybeParseCXX0XAttributes(attrs);

    // In C++0x, "for (T NS:a" might not be a typo for ::
    bool MightBeForRangeStmt = getLangOpts().CPlusPlus;
    ColonProtectionRAIIObject ColonProtection(*this, MightBeForRangeStmt);

    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    StmtVector Stmts(Actions);
    DeclGroupPtrTy DG = ParseSimpleDeclaration(Stmts, Declarator::ForContext,
                                               DeclEnd, attrs, false,
                                               MightBeForRangeStmt ?
                                                 &ForRangeInit : 0);
    FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());

    if (ForRangeInit.ParsedForRangeDecl()) {
      Diag(ForRangeInit.ColonLoc, getLangOpts().CPlusPlus0x ?
           diag::warn_cxx98_compat_for_range : diag::ext_for_range);

      ForRange = true;
    } else if (Tok.is(tok::semi)) {  // for (int x = 4;
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      Actions.ActOnForEachDeclStmt(DG);
      // ObjC: for (id x in expr)
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DG);
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
    }
  } else {
    ProhibitAttributes(attrs);
    Value = ParseExpression();

    ForEach = isTokIdentifier_in();

    // Turn the expression into a stmt.
    if (!Value.isInvalid()) {
      if (ForEach)
        FirstPart = Actions.ActOnForEachLValueExpr(Value.get());
      else
        FirstPart = Actions.ActOnExprStmt(Actions.MakeFullExpr(Value.get()));
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (ForEach) {
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DeclGroupPtrTy());
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else if (getLangOpts().CPlusPlus0x && Tok.is(tok::colon) && FirstPart.get()) {
      // User tried to write the reasonable, but ill-formed, for-range-statement
      //   for (expr : expr) { ... }
      Diag(Tok, diag::err_for_range_expected_decl)
        << FirstPart.get()->getSourceRange();
      SkipUntil(tok::r_paren, false, true);
      SecondPartIsInvalid = true;
    } else {
      if (!Value.isInvalid()) {
        Diag(Tok, diag::err_expected_semi_for);
      } else {
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, true, true);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
    }
  }
  if (!ForEach && !ForRange) {
    assert(!SecondPart.get() && "Shouldn't have a second expression yet.");
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
    } else if (Tok.is(tok::r_paren)) {
      // missing both semicolons.
    } else {
      ExprResult Second;
      if (getLangOpts().CPlusPlus)
        ParseCXXCondition(Second, SecondVar, ForLoc, true);
      else {
        Second = ParseExpression();
        if (!Second.isInvalid())
          Second = Actions.ActOnBooleanCondition(getCurScope(), ForLoc,
                                                 Second.get());
      }
      SecondPartIsInvalid = Second.isInvalid();
      SecondPart = Actions.MakeFullExpr(Second.get(), ForLoc);
    }

    if (Tok.isNot(tok::semi)) {
      if (!SecondPartIsInvalid || SecondVar)
        Diag(Tok, diag::err_expected_semi_for);
      else
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, true, true);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    }

    // Parse the third part of the for specifier.
    if (Tok.isNot(tok::r_paren)) {   // for (...;...;)
      ExprResult Third = ParseExpression();
      ThirdPart = Actions.MakeFullExpr(Third.take());
    }
  }
  // Match the ')'.
  T.consumeClose();

  // We need to perform most of the semantic analysis for a C++0x for-range
  // statememt before parsing the body, in order to be able to deduce the type
  // of an auto-typed loop variable.
  StmtResult ForRangeStmt;
  StmtResult ForEachStmt;

  if (ForRange) {
    ForRangeStmt = Actions.ActOnCXXForRangeStmt(ForLoc, T.getOpenLocation(),
                                                FirstPart.take(),
                                                ForRangeInit.ColonLoc,
                                                ForRangeInit.RangeExpr.get(),
                                                T.getCloseLocation());


  // Similarly, we need to do the semantic analysis for a for-range
  // statement immediately in order to close over temporaries correctly.
  } else if (ForEach) {
    ForEachStmt = Actions.ActOnObjCForCollectionStmt(ForLoc, T.getOpenLocation(),
                                                     FirstPart.take(),
                                                     Collection.take(),
                                                     T.getCloseLocation());
  }

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for
  // for-init-statement/condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXXorObjC && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (ForEach)
   return Actions.FinishObjCForCollectionStmt(ForEachStmt.take(),
                                              Body.take());

  if (ForRange)
    return Actions.FinishCXXForRangeStmt(ForRangeStmt.take(), Body.take());

  return Actions.ActOnForStmt(ForLoc, T.getOpenLocation(), FirstPart.take(),
                              SecondPart, SecondVar, ThirdPart,
                              T.getCloseLocation(), Body.take());
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseGotoStatement() {
  assert(Tok.is(tok::kw_goto) && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.

  StmtResult Res;
  if (Tok.is(tok::identifier)) {
    LabelDecl *LD = Actions.LookupOrCreateLabel(Tok.getIdentifierInfo(),
                                                Tok.getLocation());
    Res = Actions.ActOnGotoStmt(GotoLoc, Tok.getLocation(), LD);
    ConsumeToken();
  } else if (Tok.is(tok::star)) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    ExprResult R(ParseExpression());
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, R.take());
  } else {
    Diag(Tok, diag::err_expected_ident);
    return StmtError();
  }

  return move(Res);
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseContinueStatement() {
  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, getCurScope());
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseBreakStatement() {
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, getCurScope());
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
StmtResult Parser::ParseReturnStatement() {
  assert(Tok.is(tok::kw_return) && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.

  ExprResult R;
  if (Tok.isNot(tok::semi)) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteReturn(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    if (Tok.is(tok::l_brace) && getLangOpts().CPlusPlus) {
      R = ParseInitializer();
      if (R.isUsable())
        Diag(R.get()->getLocStart(), getLangOpts().CPlusPlus0x ?
             diag::warn_cxx98_compat_generalized_initializer_lists :
             diag::ext_generalized_initializer_lists)
          << R.get()->getSourceRange();
    } else
        R = ParseExpression();
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
  }
  return Actions.ActOnReturnStmt(ReturnLoc, R.take());
}

// needSpaceAsmToken - This function handles whitespace around asm punctuation.
// Returns true if a space should be emitted.
static inline bool needSpaceAsmToken(Token currTok) {
  static Token prevTok;

  // No need for space after prevToken.
  switch(prevTok.getKind()) {
  default:
    break;
  case tok::l_square:
  case tok::r_square:
  case tok::l_brace:
  case tok::r_brace:
  case tok::colon:
    prevTok = currTok;
    return false;
  }

  // No need for a space before currToken.
  switch(currTok.getKind()) {
  default:
    break;
  case tok::l_square:
  case tok::r_square:
  case tok::l_brace:
  case tok::r_brace:
  case tok::comma:
  case tok::colon:
    prevTok = currTok;
    return false;
  }
  prevTok = currTok;
  return true;
}

/// ParseMicrosoftAsmStatement. When -fms-extensions/-fasm-blocks is enabled,
/// this routine is called to collect the tokens for an MS asm statement.
///
/// [MS]  ms-asm-statement:
///         ms-asm-block
///         ms-asm-block ms-asm-statement
///
/// [MS]  ms-asm-block:
///         '__asm' ms-asm-line '\n'
///         '__asm' '{' ms-asm-instruction-block[opt] '}' ';'[opt]
///
/// [MS]  ms-asm-instruction-block
///         ms-asm-line
///         ms-asm-line '\n' ms-asm-instruction-block
///
StmtResult Parser::ParseMicrosoftAsmStatement(SourceLocation AsmLoc) {
  SourceManager &SrcMgr = PP.getSourceManager();
  SourceLocation EndLoc = AsmLoc;
  SmallVector<Token, 4> AsmToks;
  SmallVector<unsigned, 4> LineEnds;
  do {
    bool InBraces = false;
    unsigned short savedBraceCount = 0;
    bool InAsmComment = false;
    FileID FID;
    unsigned LineNo = 0;
    unsigned NumTokensRead = 0;
    SourceLocation LBraceLoc;

    if (Tok.is(tok::l_brace)) {
      // Braced inline asm: consume the opening brace.
      InBraces = true;
      savedBraceCount = BraceCount;
      EndLoc = LBraceLoc = ConsumeBrace();
      ++NumTokensRead;
    } else {
      // Single-line inline asm; compute which line it is on.
      std::pair<FileID, unsigned> ExpAsmLoc =
          SrcMgr.getDecomposedExpansionLoc(EndLoc);
      FID = ExpAsmLoc.first;
      LineNo = SrcMgr.getLineNumber(FID, ExpAsmLoc.second);
    }

    SourceLocation TokLoc = Tok.getLocation();
    do {
      // If we hit EOF, we're done, period.
      if (Tok.is(tok::eof))
        break;

      // The asm keyword is a statement separator, so multiple asm statements
      // are allowed.
      if (!InAsmComment && Tok.is(tok::kw_asm))
        break;

      if (!InAsmComment && Tok.is(tok::semi)) {
        // A semicolon in an asm is the start of a comment.
        InAsmComment = true;
        if (InBraces) {
          // Compute which line the comment is on.
          std::pair<FileID, unsigned> ExpSemiLoc =
              SrcMgr.getDecomposedExpansionLoc(TokLoc);
          FID = ExpSemiLoc.first;
          LineNo = SrcMgr.getLineNumber(FID, ExpSemiLoc.second);
        }
      } else if (!InBraces || InAsmComment) {
        // If end-of-line is significant, check whether this token is on a
        // new line.
        std::pair<FileID, unsigned> ExpLoc =
            SrcMgr.getDecomposedExpansionLoc(TokLoc);
        if (ExpLoc.first != FID ||
            SrcMgr.getLineNumber(ExpLoc.first, ExpLoc.second) != LineNo) {
          // If this is a single-line __asm, we're done.
          if (!InBraces)
            break;
          // We're no longer in a comment.
          InAsmComment = false;
        } else if (!InAsmComment && Tok.is(tok::r_brace)) {
          // Single-line asm always ends when a closing brace is seen.
          // FIXME: This is compatible with Apple gcc's -fasm-blocks; what
          // does MSVC do here?
          break;
        }
      }
      if (!InAsmComment && InBraces && Tok.is(tok::r_brace) &&
          BraceCount == (savedBraceCount + 1)) {
        // Consume the closing brace, and finish
        EndLoc = ConsumeBrace();
        break;
      }

      // Consume the next token; make sure we don't modify the brace count etc.
      // if we are in a comment.
      EndLoc = TokLoc;
      if (InAsmComment)
        PP.Lex(Tok);
      else {
        AsmToks.push_back(Tok);
        ConsumeAnyToken();
      }
      TokLoc = Tok.getLocation();
      ++NumTokensRead;
    } while (1);

    LineEnds.push_back(AsmToks.size());

    if (InBraces && BraceCount != savedBraceCount) {
      // __asm without closing brace (this can happen at EOF).
      Diag(Tok, diag::err_expected_rbrace);
      Diag(LBraceLoc, diag::note_matching) << "{";
      return StmtError();
    } else if (NumTokensRead == 0) {
      // Empty __asm.
      Diag(Tok, diag::err_expected_lbrace);
      return StmtError();
    }
    // Multiple adjacent asm's form together into a single asm statement
    // in the AST.
    if (!Tok.is(tok::kw_asm))
      break;
    EndLoc = ConsumeToken();
  } while (1);

  // Collect the tokens into a string
  SmallString<512> Asm;
  SmallString<512> TokenBuf;
  TokenBuf.resize(512);
  unsigned AsmLineNum = 0;
  for (unsigned i = 0, e = AsmToks.size(); i < e; ++i) {
    const char *ThisTokBuf = &TokenBuf[0];
    bool StringInvalid = false;
    unsigned ThisTokLen =
      Lexer::getSpelling(AsmToks[i], ThisTokBuf, PP.getSourceManager(),
                         PP.getLangOpts(), &StringInvalid);
    if (i && (!AsmLineNum || i != LineEnds[AsmLineNum-1]) &&
        needSpaceAsmToken(AsmToks[i]))
      Asm += ' ';
    Asm += StringRef(ThisTokBuf, ThisTokLen);
    if (i + 1 == LineEnds[AsmLineNum] && i + 1 != AsmToks.size()) {
      Asm += '\n';
      ++AsmLineNum;
    }
  }

  // FIXME: We should be passing the tokens and source locations, rather than
  // (or possibly in addition to the) AsmString.  Sema is going to interact with
  // MC to determine Constraints, Clobbers, etc., which would be simplest to
  // do with the tokens.
  std::string AsmString = Asm.c_str();
  return Actions.ActOnMSAsmStmt(AsmLoc, AsmString, EndLoc);
}

/// ParseAsmStatement - Parse a GNU extended asm statement.
///       asm-statement:
///         gnu-asm-statement
///         ms-asm-statement
///
/// [GNU] gnu-asm-statement:
///         'asm' type-qualifier[opt] '(' asm-argument ')' ';'
///
/// [GNU] asm-argument:
///         asm-string-literal
///         asm-string-literal ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///                 ':' asm-clobbers
///
/// [GNU] asm-clobbers:
///         asm-string-literal
///         asm-clobbers ',' asm-string-literal
///
StmtResult Parser::ParseAsmStatement(bool &msAsm) {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();

  if (getLangOpts().MicrosoftExt && Tok.isNot(tok::l_paren) &&
      !isTypeQualifier()) {
    msAsm = true;
    return ParseMicrosoftAsmStatement(AsmLoc);
  }
  DeclSpec DS(AttrFactory);
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS, true, false);

  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "const";
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "restrict";

  // Remember if this was a volatile asm.
  bool isVolatile = DS.getTypeQualifiers() & DeclSpec::TQ_volatile;
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "asm";
    SkipUntil(tok::r_paren);
    return StmtError();
  }
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult AsmString(ParseAsmStringLiteral());
  if (AsmString.isInvalid()) {
    // Consume up to and including the closing paren.
    T.skipToEnd();
    return StmtError();
  }

  SmallVector<IdentifierInfo *, 4> Names;
  ExprVector Constraints(Actions);
  ExprVector Exprs(Actions);
  ExprVector Clobbers(Actions);

  if (Tok.is(tok::r_paren)) {
    // We have a simple asm expression like 'asm("foo")'.
    T.consumeClose();
    return Actions.ActOnAsmStmt(AsmLoc, /*isSimple*/ true, isVolatile,
                                /*NumOutputs*/ 0, /*NumInputs*/ 0, 0,
                                move_arg(Constraints), move_arg(Exprs),
                                AsmString.take(), move_arg(Clobbers),
                                T.getCloseLocation());
  }

  // Parse Outputs, if present.
  bool AteExtraColon = false;
  if (Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    AteExtraColon = Tok.is(tok::coloncolon);
    ConsumeToken();

    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }

  unsigned NumOutputs = Names.size();

  // Parse Inputs, if present.
  if (AteExtraColon ||
      Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    if (AteExtraColon)
      AteExtraColon = false;
    else {
      AteExtraColon = Tok.is(tok::coloncolon);
      ConsumeToken();
    }

    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }

  assert(Names.size() == Constraints.size() &&
         Constraints.size() == Exprs.size() &&
         "Input operand size mismatch!");

  unsigned NumInputs = Names.size() - NumOutputs;

  // Parse the clobbers, if present.
  if (AteExtraColon || Tok.is(tok::colon)) {
    if (!AteExtraColon)
      ConsumeToken();

    // Parse the asm-string list for clobbers if present.
    if (Tok.isNot(tok::r_paren)) {
      while (1) {
        ExprResult Clobber(ParseAsmStringLiteral());

        if (Clobber.isInvalid())
          break;

        Clobbers.push_back(Clobber.release());

        if (Tok.isNot(tok::comma)) break;
        ConsumeToken();
      }
    }
  }

  T.consumeClose();
  return Actions.ActOnAsmStmt(AsmLoc, false, isVolatile,
                              NumOutputs, NumInputs, Names.data(),
                              move_arg(Constraints), move_arg(Exprs),
                              AsmString.take(), move_arg(Clobbers),
                              T.getCloseLocation());
}

/// ParseAsmOperands - Parse the asm-operands production as used by
/// asm-statement, assuming the leading ':' token was eaten.
///
/// [GNU] asm-operands:
///         asm-operand
///         asm-operands ',' asm-operand
///
/// [GNU] asm-operand:
///         asm-string-literal '(' expression ')'
///         '[' identifier ']' asm-string-literal '(' expression ')'
///
//
// FIXME: Avoid unnecessary std::string trashing.
bool Parser::ParseAsmOperandsOpt(SmallVectorImpl<IdentifierInfo *> &Names,
                                 SmallVectorImpl<Expr *> &Constraints,
                                 SmallVectorImpl<Expr *> &Exprs) {
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return false;

  while (1) {
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      BalancedDelimiterTracker T(*this, tok::l_square);
      T.consumeOpen();

      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return true;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      ConsumeToken();

      Names.push_back(II);
      T.consumeClose();
    } else
      Names.push_back(0);

    ExprResult Constraint(ParseAsmStringLiteral());
    if (Constraint.isInvalid()) {
        SkipUntil(tok::r_paren);
        return true;
    }
    Constraints.push_back(Constraint.release());

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after) << "asm operand";
      SkipUntil(tok::r_paren);
      return true;
    }

    // Read the parenthesized expression.
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    ExprResult Res(ParseExpression());
    T.consumeClose();
    if (Res.isInvalid()) {
      SkipUntil(tok::r_paren);
      return true;
    }
    Exprs.push_back(Res.release());
    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return false;
    ConsumeToken();
  }
}

Decl *Parser::ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();

  if (SkipFunctionBodies && trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnFinishFunctionBody(Decl, 0);
  }

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, LBraceLoc,
                                      "parsing function body");

  // scout - insert call to __sc_init(argc, argv, gpu) at the top of main
  if(getLangOpts().Scout){
    FunctionDecl* fd = Actions.getCurFunctionDecl();
    
    std::string args;
    
    if(getLangOpts().ScoutNvidiaGPU){
      args = "true";
    }
    else{
      args = "false";
    }
    
    if(fd->isMain()){
      assert(Tok.is(tok::l_brace) &&
             "expected lbrace when inserting __sc_init()");

      if(fd->param_size() == 0){
        InsertCPPCode("__sc_init(" + args + ");", LBraceLoc, false);
      }
      else{
        assert(fd->param_size() == 2 && "expected main with two params");
        FunctionDecl::param_iterator itr = fd->param_begin();

        ParmVarDecl* paramArgc = *itr;
        ++itr;
        ParmVarDecl* paramArgv = *itr;

        std::string code = "__sc_init(" + paramArgc->getName().str() +
        ", " + paramArgv->getName().str() + ", " + args + ");";

        InsertCPPCode(code, LBraceLoc, false);
      }
    }
  }

  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  StmtResult FnBody(ParseCompoundStatementBody());

  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
}

/// ParseFunctionTryBlock - Parse a C++ function-try-block.
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
Decl *Parser::ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");
  SourceLocation TryLoc = ConsumeToken();

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, TryLoc,
                                      "parsing function try block");

  // Constructor initializer list?
  if (Tok.is(tok::colon))
    ParseConstructorInitializer(Decl);
  else
    Actions.ActOnDefaultCtorInitializers(Decl);

  if (SkipFunctionBodies && trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnFinishFunctionBody(Decl, 0);
  }

  SourceLocation LBraceLoc = Tok.getLocation();
  StmtResult FnBody(ParseCXXTryBlockCommon(TryLoc));
  // If we failed to parse the try-catch, we just give the function an empty
  // compound statement as the body.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
}

bool Parser::trySkippingFunctionBody() {
  assert(Tok.is(tok::l_brace));
  assert(SkipFunctionBodies &&
         "Should only be called when SkipFunctionBodies is enabled");

  // We're in code-completion mode. Skip parsing for all function bodies unless
  // the body contains the code-completion point.
  TentativeParsingAction PA(*this);
  ConsumeBrace();
  if (SkipUntil(tok::r_brace, /*StopAtSemi=*/false, /*DontConsume=*/false,
                /*StopAtCodeCompletion=*/PP.isCodeCompletionEnabled())) {
    PA.Commit();
    return true;
  }

  PA.Revert();
  return false;
}

/// ParseCXXTryBlock - Parse a C++ try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
StmtResult Parser::ParseCXXTryBlock() {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");

  SourceLocation TryLoc = ConsumeToken();
  return ParseCXXTryBlockCommon(TryLoc);
}

/// ParseCXXTryBlockCommon - Parse the common part of try-block and
/// function-try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
///       handler-seq:
///         handler handler-seq[opt]
///
///       [Borland] try-block:
///         'try' compound-statement seh-except-block
///         'try' compound-statment  seh-finally-block
///
StmtResult Parser::ParseCXXTryBlockCommon(SourceLocation TryLoc) {
  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));
  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?

  StmtResult TryBlock(ParseCompoundStatement(/*isStmtExpr=*/false,
                                             Scope::DeclScope|Scope::TryScope));
  if (TryBlock.isInvalid())
    return move(TryBlock);

  // Borland allows SEH-handlers with 'try'

  if ((Tok.is(tok::identifier) &&
       Tok.getIdentifierInfo() == getSEHExceptKeyword()) ||
      Tok.is(tok::kw___finally)) {
    // TODO: Factor into common return ParseSEHHandlerCommon(...)
    StmtResult Handler;
    if(Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHExceptBlock(Loc);
    }
    else {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHFinallyBlock(Loc);
    }
    if(Handler.isInvalid())
      return move(Handler);

    return Actions.ActOnSEHTryBlock(true /* IsCXXTry */,
                                    TryLoc,
                                    TryBlock.take(),
                                    Handler.take());
  }
  else {
    StmtVector Handlers(Actions);
    ParsedAttributesWithRange attrs(AttrFactory);
    MaybeParseCXX0XAttributes(attrs);
    ProhibitAttributes(attrs);

    if (Tok.isNot(tok::kw_catch))
      return StmtError(Diag(Tok, diag::err_expected_catch));
    while (Tok.is(tok::kw_catch)) {
      StmtResult Handler(ParseCXXCatchBlock());
      if (!Handler.isInvalid())
        Handlers.push_back(Handler.release());
    }
    // Don't bother creating the full statement if we don't have any usable
    // handlers.
    if (Handlers.empty())
      return StmtError();

    return Actions.ActOnCXXTryBlock(TryLoc, TryBlock.take(),move_arg(Handlers));
  }
}

/// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the standard
///
///       handler:
///         'catch' '(' exception-declaration ')' compound-statement
///
///       exception-declaration:
///         type-specifier-seq declarator
///         type-specifier-seq abstract-declarator
///         type-specifier-seq
///         '...'
///
StmtResult Parser::ParseCXXCatchBlock() {
  assert(Tok.is(tok::kw_catch) && "Expected 'catch'");

  SourceLocation CatchLoc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return StmtError();

  // C++ 3.3.2p3:
  // The name in a catch exception-declaration is local to the handler and
  // shall not be redeclared in the outermost block of the handler.
  ParseScope CatchScope(this, Scope::DeclScope | Scope::ControlScope);

  // exception-declaration is equivalent to '...' or a parameter-declaration
  // without default arguments.
  Decl *ExceptionDecl = 0;
  if (Tok.isNot(tok::ellipsis)) {
    DeclSpec DS(AttrFactory);
    if (ParseCXXTypeSpecifierSeq(DS))
      return StmtError();
    Declarator ExDecl(DS, Declarator::CXXCatchContext);
    ParseDeclarator(ExDecl);
    ExceptionDecl = Actions.ActOnExceptionDeclarator(getCurScope(), ExDecl);
  } else
    ConsumeToken();

  T.consumeClose();
  if (T.getCloseLocation().isInvalid())
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));

  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  StmtResult Block(ParseCompoundStatement());
  if (Block.isInvalid())
    return move(Block);

  return Actions.ActOnCXXCatchBlock(CatchLoc, ExceptionDecl, Block.take());
}

void Parser::ParseMicrosoftIfExistsStatement(StmtVector &Stmts) {
  IfExistsCondition Result;
  if (ParseMicrosoftIfExistsCondition(Result))
    return;

  // Handle dependent statements by parsing the braces as a compound statement.
  // This is not the same behavior as Visual C++, which don't treat this as a
  // compound statement, but for Clang's type checking we can't have anything
  // inside these braces escaping to the surrounding code.
  if (Result.Behavior == IEB_Dependent) {
    if (!Tok.is(tok::l_brace)) {
      Diag(Tok, diag::err_expected_lbrace);
      return;
    }

    StmtResult Compound = ParseCompoundStatement();
    if (Compound.isInvalid())
      return;

    StmtResult DepResult = Actions.ActOnMSDependentExistsStmt(Result.KeywordLoc,
                                                              Result.IsIfExists,
                                                              Result.SS,
                                                              Result.Name,
                                                              Compound.get());
    if (DepResult.isUsable())
      Stmts.push_back(DepResult.get());
    return;
  }

  BalancedDelimiterTracker Braces(*this, tok::l_brace);
  if (Braces.consumeOpen()) {
    Diag(Tok, diag::err_expected_lbrace);
    return;
  }

  switch (Result.Behavior) {
  case IEB_Parse:
    // Parse the statements below.
    break;

  case IEB_Dependent:
    llvm_unreachable("Dependent case handled above");

  case IEB_Skip:
    Braces.skipToEnd();
    return;
  }

  // Condition is true, parse the statements.
  while (Tok.isNot(tok::r_brace)) {
    // scout - store the stmt vec so we can insert statements into it
    // when Stmts is not available as a parameter
    StmtsStack.push_back(&Stmts);
    StmtResult R = ParseStatementOrDeclaration(Stmts, false);
    StmtsStack.pop_back();
    if (R.isUsable())
      Stmts.push_back(R.release());
  }
  Braces.consumeClose();
}

// scout - Stmts
StmtResult Parser::ParseForAllStatement(ParsedAttributes &attrs, bool ForAll) {
  if(ForAll)
    assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");
  else
    assert(Tok.is(tok::kw_renderall) && "Not a renderall stmt!");

  SourceLocation ForAllLoc = ConsumeToken();  // eat the 'forall' / 'renderall'

  tok::TokenKind VariableType = Tok.getKind();

  bool elements = false;
  
  ForAllStmt::ForAllType FT;
  switch(VariableType) {
    case tok::kw_cells:
      FT = ForAllStmt::Cells;
      break;
    case tok::kw_edges:
      FT = ForAllStmt::Edges;
      break;
    case tok::kw_vertices:
      FT = ForAllStmt::Vertices;
      break;
    case tok::kw_elements:
      if(!ForAll){
        elements = true;
        break;
      }
      // fall through if this is a forall
    default: {
      Diag(Tok, diag::err_expected_vertices_cells);
      SkipUntil(tok::semi);
      return StmtError();
    }
  }

  ConsumeToken();

  unsigned ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
  Scope::DeclScope | Scope::ControlScope;

  ParseScope ForAllScope(this, ScopeFlags);

  if(Tok.isNot(tok::identifier)){
    Diag(Tok, diag::err_expected_ident);
    SkipUntil(tok::semi);
    return StmtError();
  }

  IdentifierInfo* LoopVariableII = Tok.getIdentifierInfo();
  SourceLocation LoopVariableLoc = Tok.getLocation();

  ConsumeToken();

  if(elements){
    if(Tok.isNot(tok::kw_in)){
      Diag(Tok, diag::err_expected_in_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }  
  }
  else{
    if(Tok.isNot(tok::kw_of)){
      Diag(Tok, diag::err_expected_of_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }    
  }
  
  ConsumeToken();

  IdentifierInfo* MeshII = 0;
  SourceLocation MeshLoc;
  VarDecl* MVD = 0;
  
  MemberExpr* ElementMember = 0;
  Expr* ElementColor = 0;
  Expr* ElementRadius = 0;
  
  const MeshType *MT;
  
  IdentifierInfo* CameraII = 0;
  SourceLocation CameraLoc;

  if(elements){
    if(MemberExpr* me = dyn_cast<MemberExpr>(ParseExpression().get())){
      if(FieldDecl* fd = dyn_cast<FieldDecl>(me->getMemberDecl())){
        
        if(const ArrayType* at = 
           dyn_cast<ArrayType>(fd->getType().getTypePtr())){
        }
        else{
          Diag(Tok, diag::err_not_array_renderall_elements);
          SkipUntil(tok::semi);
          return StmtError();
        }

        if(fd->meshFieldType() == FieldDecl::FieldCells){
          ElementMember = me;
        }
      }
    }

    if(!ElementMember){
      Diag(Tok, diag::err_expected_mesh_field);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    if(Tok.isNot(tok::kw_as)){
      Diag(Tok, diag::err_expected_as_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }

    ConsumeToken();
    
    if(Tok.isNot(tok::kw_spheres)){
      Diag(Tok, diag::err_expected_spheres_kw);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    ConsumeToken();
    
    FT = ForAllStmt::ElementSpheres;
    
    if(Tok.isNot(tok::l_paren)){
      Diag(Tok, diag::err_expected_lparen);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    ConsumeParen();
    
    for(unsigned i = 0; i < 2; ++i){
      if(Tok.is(tok::r_paren)){
        break;
      }
      
      if(Tok.is(tok::identifier)){
        IdentifierInfo* II = Tok.getIdentifierInfo();
        SourceLocation IILoc = Tok.getLocation();
        
        ConsumeToken();
        
        if(Tok.isNot(tok::equal)){
          Diag(Tok, diag::err_expected_equal_after_element_default);
          SkipUntil(tok::semi);
          return StmtError();
        }
        
        ConsumeToken();
        
        ExprResult result = ParseAssignmentExpression();
        
        if(result.isInvalid()){
          return StmtError();
        }
        
        if(II->getName() == "radius"){
          if(ElementRadius){
            Diag(IILoc, diag::err_duplicate_radius_default);
            SkipUntil(tok::semi);
            return StmtError();
          }
          
          ElementRadius = result.get();
        }
        else if(II->getName() == "color"){
          if(ElementColor){
            Diag(Tok, diag::err_duplicate_color_default);
            SkipUntil(tok::semi);
            return StmtError();
          }
          
          ElementColor = result.get();
        }
        else{
          Diag(IILoc, diag::err_invalid_default_element);
          SkipUntil(tok::semi);
          return StmtError();
        }
      }
    }
    
    if(Tok.isNot(tok::r_paren)){
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    ConsumeParen();
  }
  else{
    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return StmtError();
    }
    
    MeshII = Tok.getIdentifierInfo();
    MeshLoc = Tok.getLocation();
    
    ConsumeToken();    
  }

  bool success = false;
  if(ForAll){
    success = Actions.ActOnForAllLoopVariable(getCurScope(),
                                              VariableType,
                                              LoopVariableII,
                                              LoopVariableLoc,
                                              MeshII,
                                              MeshLoc);
  }
  else{
    if(elements){
      
      MT = Actions.ActOnRenderAllElementsVariable(getCurScope(),
                                                  ElementMember,
                                                  VariableType,
                                                  LoopVariableII,
                                                  LoopVariableLoc);
      
      if(MT){
        success = true;
      }
      
    }
    else{
    
      success = Actions.ActOnRenderAllLoopVariable(getCurScope(),
                                                   VariableType,
                                                   LoopVariableII,
                                                   LoopVariableLoc,
                                                   MeshII,
                                                   MeshLoc);
    }
  }
  
  if(!success) {
    return StmtError();
  }

  
  Expr* Op;
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;
  
  if(!elements){
    Op = 0;
    
    // Lookup the meshtype and store it for the ForAllStmt Constructor.
    LookupResult LR(Actions, MeshII, MeshLoc, Sema::LookupOrdinaryName);
    Actions.LookupName(LR, getCurScope());
    MVD = cast<VarDecl>(LR.getFoundDecl());
    MT = 
    cast<MeshType>(MVD->getType().getCanonicalType().getNonReferenceType().getTypePtr());
    size_t FieldCount = 0;
    const MeshDecl* MD = MT->getDecl();
    
    for(MeshDecl::field_iterator FI = MD->field_begin(),
        FE = MD->field_end(); FI != FE; ++FI){
      FieldDecl* FD = *FI;
      switch(FT){
        case ForAllStmt::Cells:
          if(!FD->isMeshImplicit() &&
             FD->meshFieldType() == FieldDecl::FieldCells){
            ++FieldCount;
          }
          break;
        case ForAllStmt::Edges:
          if(!FD->isMeshImplicit() &&
             FD->meshFieldType() == FieldDecl::FieldEdges){
            ++FieldCount;
          }
          break;
        case ForAllStmt::Vertices:
          if(!FD->isMeshImplicit() &&
             FD->meshFieldType() == FieldDecl::FieldVertices){
            ++FieldCount;
          }
          break;
      }
    }
    
    if(FieldCount == 0){
      switch(FT){
        case ForAllStmt::Cells:
          Diag(ForAllLoc, diag::warn_no_cells_fields_forall);
          break;
        case ForAllStmt::Edges:
          Diag(ForAllLoc, diag::warn_no_edges_fields_forall);
          break;
        case ForAllStmt::Vertices:
          Diag(ForAllLoc, diag::warn_no_vertices_fields_forall);
          break;
      }
    }
   
    // If 3D and forall type is cell(volume renderall), it can accept a camera and window
    // ala "with camera onto win", where camera and window were
    // defined previously.  If none are given it can do a default, but that is
    // probably not going to be a good thing.  You may not see the volume if the 
    // camera is not pointing at it.


    if (!ForAll && (MT->dimensions().size() == 3) && (FT == ForAllStmt::Cells)) {
      if (Tok.is(tok::kw_with)) {
        ConsumeToken();
        if(Tok.isNot(tok::identifier)){
          Diag(Tok, diag::err_expected_ident);
          SkipUntil(tok::semi);
          return StmtError();
        }
    
        CameraII = Tok.getIdentifierInfo();
        CameraLoc = Tok.getLocation();
        ConsumeToken();
      }
    }
        
    if(Tok.is(tok::kw_where)){
      ConsumeToken();
      if(Tok.isNot(tok::l_paren)){
        Diag(Tok, diag::err_invalid_forall_op);
        SkipUntil(tok::l_brace);
        ConsumeToken();
        return StmtError();
      }
      
      LParenLoc = ConsumeParen();
      
      ExprResult OpResult = ParseExpression();
      if(OpResult.isInvalid()){
        Diag(Tok, diag::err_invalid_forall_op);
        SkipUntil(tok::l_brace);
        ConsumeToken();
        return StmtError();
      }
      
      Op = OpResult.get();
      
      if(Tok.isNot(tok::r_paren)){
        Diag(Tok, diag::err_expected_rparen);
        SkipUntil(tok::l_brace);
        ConsumeToken();
        return StmtError();
      }
      RParenLoc = ConsumeParen();
    }
    else{
      Op = 0;
    }

  }

  // Check if this is a volume renderall.  If the mesh is
  // three-dimensional and has cells as the ForAllType,
  // then we branch off here into other code to handle it.

  if (!ForAll && (MT->dimensions().size() == 3) && (FT == ForAllStmt::Cells)) {
    return(ParseVolumeRenderAll(getCurScope(), ForAllLoc, attrs, MeshII, MVD, 
          CameraII, CameraLoc, Op, LParenLoc, RParenLoc));
  }

  SourceLocation BodyLoc = Tok.getLocation();

  StmtResult BodyResult(ParseStatement());
  if(BodyResult.isInvalid()){
    if(ForAll)
      Diag(Tok, diag::err_invalid_forall_body);
    else
      Diag(Tok, diag::err_invalid_renderall_body);
    SkipUntil(tok::semi);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();

  StmtResult ForAllResult;
  if(ForAll){
    InsertCPPCode("^(void* m, int* i, int* j, int* k){}", BodyLoc);

    BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
    assert(Block && "expected a block expression");
    Block->getBlockDecl()->setBody(cast<class CompoundStmt>(Body));

    ForAllResult = Actions.ActOnForAllStmt(ForAllLoc, FT, MT, MVD,
                                           LoopVariableII, MeshII, LParenLoc,
                                           Op, RParenLoc, Body, Block);
    if(!ForAllResult.isUsable())
      return StmtError();
  }
  else {
    InsertCPPCode("^(void* m, int* i, int* j, int* k){}", BodyLoc);

    BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
    assert(Block && "expected a block expression");
    Block->getBlockDecl()->setBody(cast<class CompoundStmt>(Body));

    assert(!StmtsStack.empty());
    
    MeshType::MeshDimensionVec dims = MT->dimensions();

    assert(dims.size() >= 1);

    std::string bc;
    
    if(elements){
      bc = "scoutBeginRenderAllElements(";
    }
    else{
      bc = "__sc_begin_uniform_renderall(";
    }
    
    for(size_t i = 0; i < 3; ++i){
      if(i > 0){
        bc += ", ";
      }

      if(i >= dims.size()){
        bc += "0";
      }
      else{
        bc += ToCPPCode(dims[i]);
      }
    }

    bc += ");";

    InsertCPPCode(bc, Tok.getLocation());

    StmtResult BR = ParseStatementOrDeclaration(*StmtsStack.back(), true).get();

    StmtsStack.back()->push_back(BR.get());

    if(elements){
      InsertCPPCode("__sc_end_renderall();", BodyLoc);
    }
    else{
      InsertCPPCode("__sc_end_renderall();", BodyLoc);      
    }

    ForAllResult = Actions.ActOnRenderAllStmt(ForAllLoc, FT, MT, MVD,
                                              LoopVariableII, MeshII, LParenLoc,
                                              Op, RParenLoc, Body, Block);
  }

  ForAllStmt *FAS;
  if(ForAllResult.get()->getStmtClass() == Stmt::RenderAllStmtClass) {
    RenderAllStmt* RAS = cast<RenderAllStmt>(ForAllResult.get());
    
    if(elements){
      RAS->setElementMember(ElementMember);
      RAS->setElementColor(ElementColor);
      RAS->setElementRadius(ElementRadius);
    }
    
    FAS = cast<ForAllStmt>(RAS);
  } else {
    FAS = cast<ForAllStmt>(ForAllResult.get());
  }

  MeshType::MeshDimensionVec dims = MT->dimensions();
  ASTContext &C = Actions.getASTContext();
  Expr *zero = IntegerLiteral::Create(C, llvm::APInt(32, 0),
                                      C.IntTy, ForAllLoc);
  Expr *one = IntegerLiteral::Create(C, llvm::APInt(32, 1),
                                      C.IntTy, ForAllLoc);
  for(unsigned i = 0, e = dims.size(); i < e; ++i) {
    FAS->setStart(i, zero);
    FAS->setEnd(i, dims[i]);
    FAS->setStride(i, one);
  }
  return ForAllResult;
}

StmtResult
Parser::ParseForAllShortStatement(IdentifierInfo* Name,
                                  SourceLocation NameLoc,
                                  VarDecl* VD){
  ConsumeToken();

  assert(Tok.is(tok::period) && "expected period");
  ConsumeToken();

  assert(Tok.is(tok::identifier) && "expected identifier");

  IdentifierInfo* FieldName = Tok.getIdentifierInfo();
  SourceLocation FieldLoc = ConsumeToken();

  Expr* XStart = 0;
  Expr* XEnd = 0;
  Expr* YStart = 0;
  Expr* YEnd = 0;
  Expr* ZStart = 0;
  Expr* ZEnd = 0;

  Actions.SCLStack.push_back(VD);

  for(size_t i = 0; i < 3; ++i){

    assert(Tok.is(tok::l_square) && "expected l_square");
    ConsumeBracket();

    ExprResult Start = ParseExpression();
    if(Start.isInvalid()){
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    if(Tok.isNot(tok::periodperiod)){
      Diag(Tok, diag::err_expected_periodperiod);
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    ConsumeToken();

    ExprResult End = ParseExpression();
    if(End.isInvalid()){
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    switch(i){
      case 0:
      {
        XStart = Start.get();
        XEnd = End.get();
        break;
      }
      case 1:
      {
        YStart = Start.get();
        YEnd = End.get();
        break;
      }
      case 2:
      {
        ZStart = Start.get();
        ZEnd = End.get();
        break;
      }
    }

    if(Tok.isNot(tok::r_square)){
      Diag(Tok, diag::err_expected_rsquare);
      SkipUntil(tok::semi);
      Actions.SCLStack.pop_back();
      return StmtError();
    }

    ConsumeBracket();

    if(Tok.isNot(tok::l_square)){
      break;
    }
  }

  if(Tok.isNot(tok::equal) &&
     Tok.isNot(tok::plusequal) &&
     Tok.isNot(tok::minusequal) &&
     Tok.isNot(tok::starequal) &&
     Tok.isNot(tok::slashequal)){
    Diag(Tok, diag::err_expected_forall_binary_op);
    SkipUntil(tok::semi);
    Actions.SCLStack.pop_back();
    return StmtError();
  }

  std::string code = FieldName->getName().str() + " " + TokToStr(Tok) + " ";

  SourceLocation CodeLoc = ConsumeToken();

  ExprResult rhs = ParseExpression();

  if(rhs.isInvalid()){
    SkipUntil(tok::semi);
    Actions.SCLStack.pop_back();
    return StmtError();
  }

  code += ToCPPCode(rhs.get());

  InsertCPPCode(code, CodeLoc);

  Stmt* Body = ParseStatement().get();

  InsertCPPCode("^(void* m, int* i, int* j, int* k){}", CodeLoc);
  
  BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
  assert(Block && "expected a block expression");
  class CompoundStmt* CB =
  new (Actions.Context) class CompoundStmt(Actions.Context,
                                           &Body, 1, CodeLoc, CodeLoc);


  Block->getBlockDecl()->setBody(CB);

  // Lookup the meshtype and store it for the ForAllStmt Constructor.
  LookupResult LR(Actions, Name, NameLoc, Sema::LookupOrdinaryName);
  Actions.LookupName(LR, getCurScope());
  VarDecl* MVD = cast<VarDecl>(LR.getFoundDecl());
  const MeshType *MT = cast<MeshType>(MVD->getType().getCanonicalType());

  StmtResult ForAllResult =
  Actions.ActOnForAllStmt(NameLoc,
                          ForAllStmt::Cells,
                          MT,
                          MVD,
                          &Actions.Context.Idents.get("c"),
                          Name,
                          NameLoc,
                          0, NameLoc, Body, Block);

  ForAllStmt* FAS = cast<ForAllStmt>(ForAllResult.get());

  FAS->setXStart(XStart);
  FAS->setXEnd(XEnd);
  FAS->setYStart(YStart);
  FAS->setYEnd(YEnd);
  FAS->setZStart(ZStart);
  FAS->setZEnd(ZEnd);

  return ForAllResult;
}

StmtResult Parser::ParseForAllArrayStatement(ParsedAttributes &attrs){
  assert(Tok.is(tok::kw_forall) && "Not a forall stmt!");

  SourceLocation ForAllLoc = ConsumeToken();

  IdentifierInfo* IVII[3] = {0,0,0};
  SourceLocation IVSL[3];

  size_t count;
  for(size_t i = 0; i < 3; ++i){
    if(Tok.isNot(tok::identifier)){
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_brace);
      ConsumeBrace();
      return StmtError();
    }

    IVII[i] = Tok.getIdentifierInfo();
    IVSL[i] = ConsumeToken();

    count = i + 1;

    if(Tok.is(tok::kw_in)){
      break;
    }
    else if(Tok.isNot(tok::comma)){
      Diag(Tok, diag::err_expected_comma);
      SkipUntil(tok::r_brace);
      ConsumeBrace();
      return StmtError();
    }
    ConsumeToken();
  }

  if(Tok.isNot(tok::kw_in)){
    Diag(Tok, diag::err_expected_in_kw);
    SkipUntil(tok::r_brace);
    ConsumeBrace();
    return StmtError();
  }

  ConsumeToken();

  if(Tok.isNot(tok::l_square)){
    Diag(Tok, diag::err_expected_lsquare);
    SkipUntil(tok::r_brace);
    ConsumeBrace();
    return StmtError();
  }

  ConsumeBracket();

  Expr* Start[3] = {0,0,0};
  Expr* End[3] = {0,0,0};
  Expr* Stride[3] = {0,0,0};

  for(size_t i = 0; i < 3; ++i){
    if(Tok.is(tok::coloncolon)){
      ConsumeToken();
    }
    else{
      if(Tok.isNot(tok::colon)){
        ExprResult StartResult = ParseAssignmentExpression();
        if(StartResult.isInvalid() ||
           (Tok.isNot(tok::colon) && Tok.isNot(tok::coloncolon))){
          Diag(Tok, diag::err_invalid_start_forall_array);
          SkipUntil(tok::r_brace);
          ConsumeBrace();
          return StmtError();
        }
        Start[i] = StartResult.get();
      }

      if(Tok.is(tok::coloncolon)){
        ConsumeToken();
      }
      else{
        ConsumeToken();

        if(Tok.isNot(tok::colon)){
          ExprResult EndResult = ParseAssignmentExpression();
          if(EndResult.isInvalid() || Tok.isNot(tok::colon)){
            Diag(Tok, diag::err_invalid_end_forall_array);
            SkipUntil(tok::r_brace);
            ConsumeBrace();
            return StmtError();
          }
          End[i] = EndResult.get();
        }
        ConsumeToken();
      }
    }

    if(Tok.is(tok::comma) || Tok.is(tok::r_square)){
      Stride[i] =
      IntegerLiteral::Create(Actions.Context, llvm::APInt(32, 1),
                             Actions.Context.IntTy, ForAllLoc);
    }
    else{
      ExprResult StrideResult = ParseAssignmentExpression();
      if(StrideResult.isInvalid()){
        Diag(Tok, diag::err_invalid_stride_forall_array);
        SkipUntil(tok::r_brace);
        ConsumeBrace();
        return StmtError();
      }
      Stride[i] = StrideResult.get();
    }

    if(Tok.isNot(tok::comma)){
      if(i != count - 1){
        Diag(Tok, diag::err_mismatch_forall_array);
        SkipUntil(tok::r_brace);
        ConsumeBrace();
        return StmtError();
      }
      break;
    }

    ConsumeToken();
  }

  if(Tok.isNot(tok::r_square)){
    Diag(Tok, diag::err_expected_rsquare);
    SkipUntil(tok::r_brace);
    ConsumeBrace();
    return StmtError();
  }

  ConsumeBracket();

  unsigned ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
  Scope::DeclScope | Scope::ControlScope;

  ParseScope ForAllScope(this, ScopeFlags);

  for(size_t i = 0; i < count; ++i){
    if(!IVII[i]){
      break;
    }

    if(!Actions.ActOnForAllArrayInductionVariable(getCurScope(),
                                                  IVII[i],
                                                  IVSL[i])){
      return StmtError();
    }
  }

  StmtResult BodyResult(ParseStatement());
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_forall_body);
    return StmtError();
  }

  Stmt* Body = BodyResult.get();
  
  InsertCPPCode("^(void* m, int* i, int* j, int* k){}", ForAllLoc);
  
  BlockExpr* Block = dyn_cast<BlockExpr>(ParseExpression().get());
  assert(Block && "expected a block expression");
  Block->getBlockDecl()->setBody(cast<class CompoundStmt>(Body));
  
  StmtResult ForAllArrayResult =
  Actions.ActOnForAllArrayStmt(ForAllLoc, Body, Block);

  Stmt* stmt = ForAllArrayResult.get();

  ForAllArrayStmt* FA = dyn_cast<ForAllArrayStmt>(stmt);

  for(size_t i = 0; i < 3; ++i){
    if(!Stride[i]){
      break;
    }

    FA->setStart(i, Start[i]);
    FA->setEnd(i, End[i]);
    FA->setStride(i, Stride[i]);
    FA->setInductionVar(i, IVII[i]);
  }

  return ForAllArrayResult;
}

StmtResult Parser::ParseVolumeRenderAll(Scope* scope,
    SourceLocation VolRenLoc, ParsedAttributes &attrs,
    IdentifierInfo* MeshII, VarDecl* MVD, 
    IdentifierInfo* CameraII, SourceLocation CameraLoc, Expr* Op,
    SourceLocation OpLParenLoc, SourceLocation OpRParenLoc){

  ParseScope CompoundScope(this, Scope::DeclScope);
  StmtVector Stmts(Actions);
  StmtResult R;
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
      Tok.getLocation(),
      "in volume renderall statement ('{}')");

  // Now parse Body for transfer function closure.
  // We want to parse it here -- will give correct line numbers
  // for errors and warnings if we do so.
  StmtResult BodyResult(ParseStatement());

  // Not sure why it doesn't deem it invalid when an error is 
  // found in ParseStatement

  // We end up getting an error and warning later, since it doesn't seem to 
  // return here as I would expect.
  if(BodyResult.isInvalid()){
    Diag(Tok, diag::err_invalid_renderall_body);
    SkipUntil(tok::r_brace);
    return StmtError();
  }

  class CompoundStmt* compoundStmt;
  compoundStmt = dyn_cast<class CompoundStmt>(BodyResult.get());
  SourceLocation RBraceLoc = compoundStmt->getRBracLoc();

  // TBD do more in here

  return Actions.ActOnVolumeRenderAllStmt(scope, VolRenLoc, LBraceLoc, RBraceLoc, 
      MeshII, MVD, CameraII, CameraLoc, move_arg(Stmts), compoundStmt, false);

}
