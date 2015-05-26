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

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scout/ASTVisitors.h"
#include "clang/AST/Scout/ImplicitMeshParamDecl.h"
#include "clang/AST/Scout/ImplicitColorParamDecl.h"
#include "clang/AST/Expr.h"

using namespace clang;
using namespace sema;


// We have to go in circles a bit here. First get the QualType
// from the VarDecl, then check if this is a MeshType and if so
// we can get the MeshDecl
MeshDecl *VarDecl2MeshDecl(VarDecl *VD) {
  QualType QT = VD->getType();

   if (const MeshType *MT = QT->getAs<MeshType>()) {
      return MT->getDecl();
   }
   return 0;
}

// Check forall mesh for shadowing
bool Sema::CheckForallMesh(Scope* S,
                                IdentifierInfo* RefVarInfo,
                                SourceLocation RefVarLoc,
                                VarDecl *VD) {

  // check if RefVar is a mesh member.
  // see test/scc/error/forall-mesh-shadow.sc
  if(MeshDecl *MD = VarDecl2MeshDecl(VD)) {

    // lookup RefVar name as a member. If we did an Ordinary
    // lookup we would be in the wrong IDNS. we need IDNS_Member
    // here not IDNS_Ordinary
    LookupResult MemberResult(*this, RefVarInfo, RefVarLoc,
            LookupMemberName);

    //lookup this in the MeshDecl
    LookupQualifiedName(MemberResult, MD);
    if(MemberResult.getResultKind() != LookupResult::NotFound) {
      Diag(RefVarLoc, diag::err_loop_variable_shadows_forall) << RefVarInfo;
      return false;
    }
  }

  // warn about shadowing see test/scc/warning/forall-mesh-shadow.sc
  // look up implicit mesh Ref variable
  LookupResult RefResult(*this, RefVarInfo, RefVarLoc,
          LookupOrdinaryName);
  LookupName(RefResult, S);
  if(RefResult.getResultKind() != LookupResult::NotFound) {
    Diag(RefVarLoc, diag::warn_loop_variable_shadows_forall) << RefVarInfo;
  }

  return true;
}


// ----- ActOnForallMeshRefVariable
// This call assumes the reference variable details have been parsed
// Given this, this member function takes steps to further determine
// the actual mesh type of the forall (passed in as a base mesh type)
// and creates the reference variable and its DeclStmt
bool Sema::ActOnForallMeshRefVariable(Scope* S,
                                  IdentifierInfo* MeshVarInfo,
                                  SourceLocation MeshVarLoc,
                                  ForallMeshStmt::MeshElementType RefVarType,
                                  IdentifierInfo* RefVarInfo,
                                  SourceLocation RefVarLoc,
                                  const MeshType *MT,
                                  VarDecl* VD,
                                  DeclStmt **Init) {

  if(!CheckForallMesh(S, RefVarInfo, RefVarLoc, VD)) {
    return false;
  }

  ImplicitMeshParamDecl* D;

  ImplicitMeshParamDecl::MeshElementType ET = (ImplicitMeshParamDecl::MeshElementType)RefVarType;

  if (MT->isUniform()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<UniformMeshType>(MT),0), VD);
  } else if (MT->isALE()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<ALEMeshType>(MT),0), VD);
  } else if (MT->isStructured()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<StructuredMeshType>(MT),0), VD);

  } else if (MT->isRectilinear()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<RectilinearMeshType>(MT),0), VD);
  } else if (MT->isUnstructured()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<UnstructuredMeshType>(MT),0), VD);

  } else {
    assert(false && "unknown mesh type");
    return false;
  }

  // build a DeclStmt for the ImplicitMeshParamDecl and return it via parameter list
  *Init = new (Context) DeclStmt(DeclGroupRef(D), RefVarLoc, RefVarLoc);

  PushOnScopeChains(D, S, true);
  SCLStack.push_back(D);  //SC_TODO: this seems like an ugly hack

  return true;
}

StmtResult Sema::ActOnForallMeshStmt(SourceLocation ForallLoc,
                                     ForallMeshStmt::MeshElementType ElementType,
                                     const MeshType *MT,
                                     VarDecl* MVD,
                                     IdentifierInfo* RefVarInfo,
                                     IdentifierInfo* MeshInfo,
                                     SourceLocation LParenLoc,
                                     Expr* Predicate, SourceLocation RParenLoc,
                                     DeclStmt* Init, VarDecl* QD, Stmt* Body) {

  SCLStack.pop_back();

  ForallMeshStmt* FS = new (Context) ForallMeshStmt(ElementType,
                                                    RefVarInfo,
                                                    MeshInfo, MVD, MT,
                                                    ForallLoc,
                                                    Init, QD, Body, Predicate,
                                                    LParenLoc, RParenLoc);

  // check that LHS mesh field assignment
  // operators do not appear as subsequent RHS values, and
  // perform other semantic checks
  ForallVisitor v(*this, FS);
  v.Visit(Body);

  if (v.error()){
    return StmtError();
  }

  return FS;
}

ExprResult Sema::ActOnQueryExpr(SourceLocation FromLoc,
                                VarDecl* MeshDecl,
                                Expr* Field,
                                Expr* Predicate){

  QueryExpr* QE = new (Context) QueryExpr(Context.getQueryType(),
                                          FromLoc,
                                          MeshDecl,
                                          Field,
                                          Predicate);
  
  return QE;
}

bool Sema::ValidateSpecExpr(SpecExpr* E, QualType t){
  SpecValueExpr* v = E->toValue();
  if(!v){
    return false;
  }
  
  ExprResult rc =
  PerformImplicitConversion(v->getExpression(), t, Sema::AA_Converting, true);
  if(rc.isInvalid()){
    return false;
  }
  
  v->setExpression(rc.get());
    
  return true;
}

bool Sema::ValidateSpecExpr(SpecExpr* E, QualType t, size_t n){
  SpecValueExpr* v = E->toValue();
  if(v){
    return ValidateSpecExpr(E,
              Context.getVectorType(t, n, VectorType::GenericVector));
  }
  
  SpecArrayExpr* a = E->toArray();
  if(a->size() != n){
    return false;
  }
  
  for(size_t i = 0; i < n; ++i){
    if(!ValidateSpecExpr(a->get(i), t)){
      return false;
    }
  }
  
  return true;
}

ExprResult Sema::ActOnSpecObjectExpr(SourceLocation BraceLoc){
  return new (Context) SpecObjectExpr(BraceLoc);
}

ExprResult Sema::ActOnSpecArrayExpr(SourceLocation BracketLoc){
  return new (Context) SpecArrayExpr(BracketLoc);
}

ExprResult Sema::ActOnSpecValueExpr(Expr* E){
  ExprResult result = CorrectDelayedTyposInExpr(E);
  
  if(result.isInvalid()){
    return ExprError();
  }
  
  return new (Context) SpecValueExpr(result.get());
}

SpecValueExpr* Sema::CreateSpecDoubleExpr(double value){
  FloatingLiteral* l =
  FloatingLiteral::Create(Context,
                          llvm::APFloat(value),
                          false,
                          Context.DoubleTy,
                          SourceLocation());
  
  return new (Context) SpecValueExpr(l);
}

SpecValueExpr* Sema::CreateSpecIntExpr(int64_t value){
  IntegerLiteral* l =
  IntegerLiteral::Create(Context,
                         llvm::APInt(64, value, true),
                         Context.LongTy,
                         SourceLocation());
  
  return new (Context) SpecValueExpr(l);
}

SpecValueExpr* Sema::CreateSpecStringExpr(const std::string& value){
  llvm::APInt len(32, value.length() + 1);
  
  QualType t = Context.CharTy.withConst();
  t = Context.getConstantArrayType(t, len, ArrayType::Normal, 0);
  
  StringLiteral* l =
  StringLiteral::Create(Context,
                        value.c_str(),
                        StringLiteral::Ascii,
                        false,
                        t,
                        SourceLocation());
  
  return new (Context) SpecValueExpr(l);
}

SpecArrayExpr* Sema::CreateSpecArrayExpr(){
  return new (Context) SpecArrayExpr(SourceLocation());
}

// Check forall array for shadowing
bool Sema::CheckForallArray(Scope* S,
                                  IdentifierInfo* InductionVarInfo,
                                  SourceLocation InductionVarLoc) {

  LookupResult LResult(*this, InductionVarInfo, InductionVarLoc,
      LookupOrdinaryName);

  LookupName(LResult, S);

  if(LResult.getResultKind() != LookupResult::NotFound){
    Diag(InductionVarLoc, diag::warn_loop_variable_shadows_forall) <<
        InductionVarInfo;

  }
  return true;
}

// ----- ActOnForallArrayInductionVariable
// This call  creates the Induction variable and its DeclStmt
bool Sema::ActOnForallArrayInductionVariable(Scope* S,
                                             IdentifierInfo* InductionVarInfo,
                                             SourceLocation InductionVarLoc,
                                             VarDecl **InductionVarDecl,
                                             DeclStmt **Init) {

  if(!CheckForallArray(S,InductionVarInfo, InductionVarLoc)) {
    return false;
  }

  // build the Induction Var. VarDecl and DeclStmt this is
  // similar to what is done in buildSingleCopyAssignRecursively()
  *InductionVarDecl = VarDecl::Create(Context, CurContext, InductionVarLoc, InductionVarLoc,
      InductionVarInfo, Context.IntTy, 0, SC_None);

  // zero initialize the induction var
  (*InductionVarDecl)->setInit(IntegerLiteral::Create(Context, llvm::APInt(32, 0),
      Context.IntTy, InductionVarLoc));

  // build a DeclStmt for the VarDecl and return both via parameter list
  *Init = new (Context) DeclStmt(DeclGroupRef(*InductionVarDecl),
      InductionVarLoc, InductionVarLoc);

  PushOnScopeChains(*InductionVarDecl, S, true);

  return true;
}

StmtResult Sema::ActOnForallArrayStmt(IdentifierInfo* InductionVarInfo[],
          VarDecl* InductionVarDecl[],
          Expr* Start[], Expr* End[], Expr* Stride[], size_t Dims,
          SourceLocation ForallLoc, DeclStmt* Init[], Stmt* Body) {


  ForallArrayStmt* FS =
  new (Context) ForallArrayStmt(InductionVarInfo, InductionVarDecl,
      Start, End, Stride, Dims, ForallLoc, Init, Body);

  return FS;
}

// ----- ActOnRenderallRefVariable
// This call assumes the reference variable details have been parsed
// (syntax checked) and issues, such as shadows, have been reported.
// Given this, this member function takes steps to further determine
// the actual mesh type of the renderall (passed in as a base mesh type)
// and creates the reference variable
bool Sema::ActOnRenderallMeshRefVariable(Scope* S,
                                         RenderallMeshStmt::MeshElementType RefVarType,
                                         IdentifierInfo* RefVarInfo,
                                         SourceLocation RefVarLoc,
                                         const MeshType *MT,
                                         VarDecl* VD) {

  ImplicitMeshParamDecl* D;

  ImplicitMeshParamDecl::MeshElementType ET = (ImplicitMeshParamDecl::MeshElementType)RefVarType;

  if (MT->isUniform()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<UniformMeshType>(MT),0), VD);
  } else if (MT->isALE()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<ALEMeshType>(MT),0), VD);
  } else if (MT->isStructured()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<StructuredMeshType>(MT),0), VD);

  } else if (MT->isRectilinear()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<RectilinearMeshType>(MT),0), VD);
  } else if (MT->isUnstructured()) {
    D = ImplicitMeshParamDecl::Create(Context,
                                      CurContext,
                                      ET,
                                      RefVarLoc,
                                      RefVarInfo,
                                      QualType(cast<UnstructuredMeshType>(MT),0), VD);

  } else {
    assert(false && "unknown mesh type");
    return false;
  }

  PushOnScopeChains(D, S, true);
  SCLStack.push_back(D);

  // add the implicit "color" parameter
  ImplicitColorParamDecl*CD = ImplicitColorParamDecl::Create(Context, CurContext, RefVarLoc);

  PushOnScopeChains(CD, S, true);

  return true;
}


StmtResult Sema::ActOnRenderallMeshStmt(SourceLocation RenderallLoc,
                                        RenderallMeshStmt::MeshElementType ElementType,
                                        const MeshType *MT,
                                        VarDecl* MVD,    // Mesh var decl. 
                                        VarDecl* RTVD,   // Render target var decl. 
                                        IdentifierInfo* RefVarInfo,
                                        IdentifierInfo* MeshInfo,
                                        IdentifierInfo* RenderTargetInfo,
                                        SourceLocation LParenLoc,
                                        Expr* Predicate, SourceLocation RParenLoc,
                                        Stmt* Body) {

  SCLStack.pop_back();

  RenderallMeshStmt* RS = new (Context) RenderallMeshStmt(ElementType,
                                                          RefVarInfo,
                                                          MeshInfo,
                                                          RenderTargetInfo,
                                                          MVD, RTVD, MT,
                                                          RenderallLoc,
                                                          Body, Predicate,
                                                          LParenLoc, RParenLoc);

  // check that LHS mesh field assignment
  // operators do not appear as subsequent RHS values, and
  // perform other semantic checks
  RenderallVisitor v(*this, RS);
  v.Visit(Body);

  if (!v.foundColorAssign()) {
    Diag(RenderallLoc, diag::err_no_color_assignment_renderall);
    return StmtError();
  }

  if (v.error()){
    return StmtError();
  }

  return RS;
}

StmtResult Sema::ActOnFrameCaptureStmt(const VarDecl* VD, SpecObjectExpr* S){
  using namespace std;
  
  const FrameType* ft = dyn_cast<FrameType>(VD->getType().getTypePtr());
  const FrameDecl* fd = ft->getDecl();
  
  auto m = S->memberMap();
  auto vm = fd->getVarMap();
  
  bool valid = true;
  
  for(auto& itr : m){
    const string& k = itr.first;
    SourceLocation loc = itr.second.first;
    SpecExpr* e = itr.second.second;
    
    auto vitr = vm.find(k);
    
    if(vitr == vm.end()){
      Diag(loc, diag::err_unknown_frame_variable) << k;
      valid = false;
      continue;
    }
    
    VarDecl* vd = vitr->second.varDecl;
    
    if(!ValidateSpecExpr(e, vd->getType())){
      Diag(loc, diag::err_invalid_frame_capture) <<
      "type mismatch";
      valid = false;
    }
  }
  
  return valid ? new (Context) FrameCaptureStmt(VD, S) : StmtError();
}

StmtResult Sema::ActOnPlotStmt(SourceLocation WithLoc,
                               SourceLocation FrameLoc,
                               VarDecl* RenderTarget,
                               FrameDecl* FD,
                               VarDecl* Frame,
                               SpecObjectExpr* Spec){
  using namespace std;
  
  bool valid = true;
  
  auto m = Spec->memberMap();
  
  for(auto& itr : m){
    const string& k = itr.first;
    SourceLocation loc = itr.second.first;
    SpecExpr* v = itr.second.second;
    
    if(k == "lines" || k == "points" || k == "area" || k == "interval"){
      SpecObjectExpr* lv = v->toObject();
      
      if(!lv){
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
        continue;
      }
      
      SpecExpr* p = lv->get("position");
      
      if(p){
        SpecArrayExpr* pa = p->toArray();
        if(pa && pa->size() == 2){
          if(!ValidateSpecExpr(pa, Context.DoubleTy, 2)){
            Diag(p->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'position'";
            valid = false;
          }
        }
        else if(k == "interval"){
          SpecObjectExpr* bv = p->toObject();
          if(!bv){
            Diag(bv->getLocStart(), diag::err_invalid_plot_spec) <<
            "expected an object specifier";
            valid = false;
            continue;
          }
          
          SpecExpr* b = bv->get("bin");
          if(!b){
            Diag(bv->getLocStart(), diag::err_invalid_plot_spec) <<
            "expected a 'bin' object specifier";
            valid = false;
            continue;
          }
          
          if(!ValidateSpecExpr(b, Context.DoubleTy)){
            Diag(bv->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'bin' object specifier";
            valid = false;
            continue;
          }
          
          if(!bv->has("n")){
            bv->put("n", CreateSpecIntExpr(0));
          }
        }
        else{
          Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'position' array of size 2";
          valid = false;
        }
      }
      else{
        Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected a 'position' key";
        valid = false;
      }
      
      if(k == "lines" || k == "points"){
        SpecExpr* s = lv->get("size");
        if(s){
          if(!ValidateSpecExpr(s, Context.DoubleTy)){
            Diag(s->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'size' key";
            valid = false;
          }
        }
        else{
          lv->put("size", CreateSpecDoubleExpr(1.0));
        }
        
        SpecExpr* label = lv->get("label");
        if(label){
          if(!ValidateSpecExpr(label, Context.getPointerType(Context.CharTy))){
            Diag(label->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'label' key, expected a string";
            valid = false;
          }
        }
      }
      
      SpecExpr* c = lv->get("color");
      if(c){
        if(!ValidateSpecExpr(c, Context.FloatTy, 4)){
          Diag(c->getLocStart(), diag::err_invalid_plot_spec) <<
          "invalid 'color'";
          valid = false;
        }
      }
      else{
        SpecArrayExpr* color = CreateSpecArrayExpr();
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(1.0));
        ValidateSpecExpr(color, Context.FloatTy, 4);
        lv->put("color", color);
      }
    }
    else if(k == "pie"){
      SpecObjectExpr* o = v->toObject();
      
      if(!o){
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
        continue;
      }
      
      SpecExpr* p = o->get("proportion");
  
      if(p){
        SpecArrayExpr* pa = p->toArray();
        
        if(pa){
          for(size_t i = 0; i < pa->size(); ++i){
            if(!ValidateSpecExpr(pa->get(i), Context.IntTy)){
              Diag(pa->get(i)->getLocStart(), diag::err_invalid_plot_spec) <<
              "invalid 'proportion' value";
              valid = false;
            }
          }
        }
        else{
          Diag(p->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'proportion' array";
          valid = false;
        }
      }
      else{
        Diag(o->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected a 'proportion' key";
        valid = false;
      }
      
      SpecExpr* c = o->get("color");
      if(c){
        if(!ValidateSpecExpr(c, Context.FloatTy, 4)){
          Diag(c->getLocStart(), diag::err_invalid_plot_spec) <<
          "invalid 'color'";
          valid = false;
        }
      }
    }
    else if(k == "line"){
      SpecObjectExpr* lv = v->toObject();
      
      if(!lv){
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
        continue;
      }
      
      SpecExpr* p = lv->get("start");
      
      if(p){
        SpecArrayExpr* pa = p->toArray();
        if(pa && pa->size() == 2){
          if(!ValidateSpecExpr(pa, Context.DoubleTy, 2)){
            Diag(p->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'start'";
            valid = false;
          }
        }
        else{
          Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'start' array of size 2";
          valid = false;
        }
      }
      else{
        Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected a 'start' key";
        valid = false;
      }
      
      p = lv->get("end");
      
      if(p){
        SpecArrayExpr* pa = p->toArray();
        if(pa && pa->size() == 2){
          if(!ValidateSpecExpr(pa, Context.DoubleTy, 2)){
            Diag(p->getLocStart(), diag::err_invalid_plot_spec) <<
            "invalid 'end'";
            valid = false;
          }
        }
        else{
          Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'end' array of size 2";
          valid = false;
        }
      }
      else{
        Diag(lv->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an 'end' key";
        valid = false;
      }
      
      SpecExpr* s = lv->get("size");
      if(s){
        if(!ValidateSpecExpr(s, Context.DoubleTy)){
          Diag(s->getLocStart(), diag::err_invalid_plot_spec) <<
          "invalid 'size' key";
          valid = false;
        }
      }
      else{
        lv->put("size", CreateSpecDoubleExpr(1.0));
      }
      
      SpecExpr* c = lv->get("color");
      if(c){
        if(!ValidateSpecExpr(c, Context.FloatTy, 4)){
          Diag(c->getLocStart(), diag::err_invalid_plot_spec) <<
          "invalid 'color'";
          valid = false;
        }
      }
      else{
        SpecArrayExpr* color = CreateSpecArrayExpr();
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(0.0));
        color->add(CreateSpecDoubleExpr(1.0));
        ValidateSpecExpr(color, Context.FloatTy, 4);
        lv->put("color", color);
      }
    }
    else if(k == "axis"){
      SpecObjectExpr* av = v->toObject();
      
      if(av){
        SpecExpr* d = av->get("dim");
        if(!(d && d->isInteger())){
          Diag(av->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'dim' integer key";
          valid = false;
        }
        
        SpecExpr* l = av->get("label");
        if(!(l && l->isString())){
          Diag(av->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected a 'label' string key";
          valid = false;
        }
        
        SpecExpr* m = av->get("major");
        if(m){
          if(!m->isInteger()){
            Diag(m->getLocStart(), diag::err_invalid_plot_spec) <<
            "'major' must be an integer";
            valid = false;
          }
        }
        else{
          av->put("major", CreateSpecIntExpr(10));
        }
        
        m = av->get("minor");
        if(m){
          if(!m->isInteger()){
            Diag(m->getLocStart(), diag::err_invalid_plot_spec) <<
            "'minor' must be an integer";
            valid = false;
          }
        }
        else{
          av->put("minor", CreateSpecIntExpr(4));
        }
      }
      else{
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
      }
    }
    else if(k == "var"){
      SpecObjectExpr* vo = v->toObject();
      if(vo){
        if(vo->size() > 0){
          auto m = vo->memberMap();
          
          for(auto& itr : m){
            SpecExpr* e = itr.second.second;
            
            if(!e->toExpr()){
              Diag(e->getLocStart(), diag::err_invalid_plot_spec) <<
              "invalid variable expression";
              valid = false;
            }
          }
        }
        else{
          Diag(vo->getLocStart(), diag::err_invalid_plot_spec) <<
          "expected one or more variable definitions";
          valid = false;
        }
      }
      else{
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
      }
    }
    else if(k == "antialiased"){
      if(!v->isBool()){
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected a bool";
        valid = false;
      }
    }
    else if(k == "output"){
      if(!v->isString()){
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected file path string";
        valid = false;
      }
    }
    else if(k == "range"){
      SpecObjectExpr* vo = v->toObject();
      
      if(vo){
        bool found = false;
        
        for(size_t i = 0; i < 2; ++i){
          SpecExpr* d = i == 0 ? vo->get("x") : vo->get("y");
          if(d){
            found = true;
            
            SpecArrayExpr* a = d->toArray();
            if(!(a && a->size() == 2)){
              Diag(d->getLocStart(), diag::err_invalid_plot_spec) <<
              "expected a 2d range array";
              valid = false;
              continue;
            }
            
            for(size_t i = 0; i < 2; ++i){
              if(!ValidateSpecExpr(a->get(i), Context.DoubleTy)){
                Diag(a->get(i)->getLocStart(), diag::err_invalid_plot_spec) <<
                "invalid range element";
                valid = false;
              }
            }
          }
          
          if(!found){
            Diag(vo->getLocStart(), diag::err_invalid_plot_spec) <<
            "expected x/y range specifier(s)";
          }
        }
      }
      else{
        Diag(v->getLocStart(), diag::err_invalid_plot_spec) <<
        "expected an object specifier";
        valid = false;
      }
    }
    else{
      Diag(loc, diag::err_invalid_plot_spec_key) << k;
      valid = false;
    }
  }
  
  return valid ? new (Context) PlotStmt(FD, Frame, RenderTarget, Spec) : StmtError();
}
