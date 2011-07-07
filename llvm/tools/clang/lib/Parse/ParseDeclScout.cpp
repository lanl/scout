//===----------------------------------------------------------------------===//
//
//  ndm - This file implements the Declaration portions of Scout.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"

#include "clang/AST/DeclScout.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/PrettyDeclStackTrace.h"

#include <iostream>

using namespace clang;

namespace{

  typedef llvm::SmallVector<Decl*, 32> FieldVec;
  
} // end namespace

void Parser::ParseMeshSpecifier(DeclSpec &DS){
  
  // the current lookahead token is tok::kw_uniform, tok::kw_rectlinear, 
  // tok::kw_structured, or tok::kw_unstructured
  
  tok::TokenKind MeshType = Tok.getKind();
    
  SourceLocation MeshTypeLocation = ConsumeToken();
  
  if(Tok.isNot(tok::kw_mesh)){
    Diag(Tok, diag::err_expected_mesh_kw);
    
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
  }
  
  SourceLocation MeshLocation = ConsumeToken();
  
  // parse mesh name
  IdentifierInfo* Name;
  SourceLocation NameLoc;
  if(Tok.is(tok::identifier)){
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  }
  else{
    Diag(Tok, diag::err_expected_ident);
    
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return;
  }
  
  MeshDecl::MeshDimensionVec dims;
  
  // parse mesh dimensions
  if(Tok.is(tok::l_square)){
    ConsumeBracket();
    
    // how should dimensions eventually be represented on the AST?
    
    for(;;){
      
      if(Tok.isNot(tok::numeric_constant)){
        Diag(Tok, diag::err_expected_numeric_constant_in_mesh_def);
        
        DS.SetTypeSpecError();
        SkipUntil(tok::r_brace);
        SkipUntil(tok::semi);
        return;
      }
      
      dims.push_back(Actions.ActOnNumericConstant(Tok).get());
      
      ConsumeToken();
      
      if(Tok.is(tok::r_square)){
        break;
      }
      
      if(Tok.is(tok::eof)){
        Diag(Tok, diag::err_expected_lsquare);
        return;
      }
         
      if(Tok.isNot(tok::comma)){
        Diag(Tok, diag::err_expected_comma);
        DS.SetTypeSpecError();
        SkipUntil(tok::r_brace);
        SkipUntil(tok::semi);
      }
      
      ConsumeToken();
    }
  }
  else{
    Diag(Tok, diag::err_expected_lsquare);
    
    DS.SetTypeSpecError();
    SkipUntil(tok::r_square);
    SkipUntil(tok::semi);
    return;
  }
  
  ConsumeBracket();
  
  if(Tok.isNot(tok::l_brace)){
    Diag(Tok, diag::err_expected_lbrace);
    
    DS.SetTypeSpecError();
    SkipUntil(tok::r_brace);
    SkipUntil(tok::semi);
    return;
  }
  
  MeshDecl* Dec = static_cast<MeshDecl*>( 
  Actions.ActOnMeshDefinition(getCurScope(), MeshType, 
                              MeshTypeLocation, Name, NameLoc)); 
  
  ParseMeshBody(MeshLocation, Dec);
}

void Parser::ParseMeshBody(SourceLocation StartLoc, MeshDecl* Dec){
  PrettyDeclStackTraceEntry CrashInfo(Actions, Dec, StartLoc,
                                      "parsing Scout mesh body");
  
  SourceLocation LBraceLoc = ConsumeBrace();
  
  ParseScope StructScope(this, Scope::ClassScope|Scope::DeclScope);
  Actions.ActOnMeshStartDefinition(getCurScope(), Dec);
  
  FieldVec FieldDecls;
  while(Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)){
    DeclSpec DS(AttrFactory);
    
    struct ScoutFieldCallback : FieldCallback {
      Parser& P;
      Decl* MeshDecl;
      llvm::SmallVectorImpl<Decl*>& FieldDecls;
      
      ScoutFieldCallback(Parser& P, Decl* MeshDecl,
                         llvm::SmallVectorImpl<Decl*>& FieldDecls) :
      P(P), MeshDecl(MeshDecl), FieldDecls(FieldDecls) {}
      
      virtual Decl* invoke(FieldDeclarator& FD) {
        // Install the declarator into the current MeshDecl.
        Decl* Field = P.Actions.ActOnField(P.getCurScope(), MeshDecl,
                                           FD.D.getDeclSpec().getSourceRange().getBegin(),
                                           FD.D, FD.BitfieldSize);
        
        FieldDecls.push_back(Field);
        return Field;
      }
    } Callback(*this, Dec, FieldDecls);
    
    ParseStructDeclaration(DS, Callback);
    
    if(Tok.is(tok::semi)){
      ConsumeToken();
    }
    else if(Tok.is(tok::r_brace)){
      ExpectAndConsume(tok::semi, diag::ext_expected_semi_decl_list);
    }
    else{
      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list);
      SkipUntil(tok::r_brace, true, true);
      if(Tok.is(tok::semi)){
        ConsumeToken();
      }
    }
    
  }
  
  SourceLocation RBraceLoc = MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  StructScope.Exit();
}

