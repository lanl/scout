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
#include "clang/AST/StmtVisitor.h"
#include <map>
using namespace clang;
using namespace sema;

// ===== Scout ================================================================
// ForAllVisitor class to check that LHS mesh field assignment
// operators do not appear as subsequent RHS values, and various other
// semantic checks

namespace {

  class ForallVisitor : public StmtVisitor<ForallVisitor> {
  public:

    enum NodeType{
      NodeNone,
      NodeLHS,
      NodeRHS
    };

    ForallVisitor(Sema& sema, ForallMeshStmt* fs)
      : sema_(sema),
        fs_(fs),
        error_(false),
        nodeType_(NodeNone) {
    }

    void VisitStmt(Stmt* S) {
      VisitChildren(S);
    }

    void VisitCallExpr(CallExpr* E) {

      FunctionDecl* fd = E->getDirectCallee();

      if (fd) {
        std::string name = fd->getName();
        if (name == "cshift") {


          const MeshType* mt = fs_->getMeshType();
          unsigned args = E->getNumArgs();          

          unsigned dims = mt->rankOf();

          if (args != dims + 1) {
            sema_.Diag(E->getRParenLoc(), diag::err_cshift_args);
            error_ = true;
          } else {
            Expr* fe = E->getArg(0);

            if (ImplicitCastExpr* ce = dyn_cast<ImplicitCastExpr>(fe)) {
              fe = ce->getSubExpr();
            }

            if (MemberExpr* me = dyn_cast<MemberExpr>(fe)) {
              if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(me->getBase())) {
                ValueDecl* bd = dr->getDecl();

                if (!isa<MeshType>(bd->getType().getCanonicalType().getTypePtr())){
                  sema_.Diag(E->getRParenLoc(), diag::err_cshift_field);
                  error_ = true;
                }
              }
            } else {
              sema_.Diag(E->getRParenLoc(), diag::err_cshift_field);
              error_ = true;
            }
          }
        } else {
          sema_.Diag(E->getRParenLoc(), diag::err_cshift_not_allowed);
          error_ = true;
        }
      }

      VisitChildren(E);
    }

    void VisitChildren(Stmt* S) {
      for(Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E; ++I) {
        if (Stmt* child = *I) {
          Visit(child);
        }
      }
    }

    void VisitMemberExpr(MemberExpr* E) {

      if (DeclRefExpr* dr = dyn_cast<DeclRefExpr>(E->getBase())) {

        ValueDecl* bd = dr->getDecl();

        if (const MeshType* MT = dyn_cast<MeshType>(bd->getType().getCanonicalType().getTypePtr())){

          ValueDecl* md = E->getMemberDecl();

          // Make sure we are only accessing mesh traits that match the dimensionality 
          // of the mesh...
          if ((md->getName() == "height" ) || (md->getName() == "depth")) {

            unsigned ND = MT->rankOf();

            if (md->getName() == "height" && ND < 2) {
              sema_.Diag(E->getMemberLoc(), diag::err_invalid_height_mesh);
              error_ = true;
            } else if (md->getName() == "depth" && ND < 3) {
              sema_.Diag(E->getMemberLoc(), diag::err_invalid_depth_mesh);
              error_ = true;
            }
          }

          std::string ref = bd->getName().str() + "." + md->getName().str();

          if (nodeType_ == NodeLHS) {
            refMap_.insert(make_pair(ref, true));
          } else if (nodeType_ == NodeRHS) {
            RefMap_::iterator itr = refMap_.find(ref);
            if (itr != refMap_.end()) {
              sema_.Diag(E->getMemberLoc(), diag::err_rhs_after_lhs_forall);
              error_ = true;
            }
          }
        }
      }
    }


    void VisitDeclStmt(DeclStmt* S) {

      DeclGroupRef declGroup = S->getDeclGroup();

      for(DeclGroupRef::iterator itr = declGroup.begin(),
          itrEnd = declGroup.end(); itr != itrEnd; ++itr){
        Decl* decl = *itr;

        if (NamedDecl* nd = dyn_cast<NamedDecl>(decl)) {
          localMap_.insert(make_pair(nd->getName().str(), true));
        }
      }

      VisitChildren(S);
    }

    //SC_TODO: remove scout vectors
    void VisitScoutVectorMemberExpr(ScoutVectorMemberExpr* E){
      if(MemberExpr* ME = dyn_cast<MemberExpr>(E->getBase())){
        if(ME->getMemberDecl()->getName() == "position"){
          if(DeclRefExpr* DR = dyn_cast<DeclRefExpr>(ME->getBase())){
            if(const MeshType* MT =
               dyn_cast<MeshType>(DR->getDecl()->getType().getCanonicalType().getTypePtr())){
              if(E->getIdx() >= MT->dimensions().size()){
                sema_.Diag(E->getLocation(), diag::err_invalid_position_ref);
                error_ = true;
              }
            }
          }
        }
      }

      VisitChildren(E);
    }

    void VisitBinaryOperator(BinaryOperator* S){

      switch(S->getOpcode()){
        case BO_Assign:
        case BO_MulAssign:
        case BO_DivAssign:
        case BO_RemAssign:
        case BO_AddAssign:
        case BO_SubAssign:
        case BO_ShlAssign:
        case BO_ShrAssign:
        case BO_AndAssign:
        case BO_XorAssign:
        case BO_OrAssign:
          if(DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S->getLHS())){
            RefMap_::iterator itr = localMap_.find(DR->getDecl()->getName().str());
            if(itr == localMap_.end()){

              sema_.Diag(DR->getLocation(),
                         diag::warn_lhs_outside_forall) << DR->getDecl()->getName();
            }
          }

          nodeType_ = NodeLHS;
          break;
        default:
          break;
      }

      Visit(S->getLHS());
      nodeType_ = NodeRHS;
      Visit(S->getRHS());
      nodeType_ = NodeNone;
    }

    bool error(){
      return error_;
    }

  private:
    Sema& sema_;
    ForallMeshStmt *fs_;
    typedef std::map<std::string, bool> RefMap_;
    RefMap_ refMap_;
    RefMap_ localMap_;
    bool error_;
    NodeType nodeType_;
  };

} // end namespace


// ----- ActOnForallRefVariable 
// This call assumes the reference variable details have been parsed 
// (syntax checked) and issues, such as shadows, have been reported.
// Given this, this member function takes steps to further determine 
// the actual mesh type of the forall (passed in as a base mesh type)
// and creates the reference variable 
bool Sema::ActOnForallMeshRefVariable(Scope* S,
                                  IdentifierInfo* RefVarInfo,
                                  SourceLocation RefVarLoc,
                                  const MeshType *MT) {

  ImplicitParamDecl* D;

  if (MT->isUniform()) {
    D = ImplicitParamDecl::Create(Context, 
                                  CurContext,
                                  RefVarLoc,
                                  RefVarInfo,
                                  QualType(cast<UniformMeshType>(MT),0 ));
  } else if (MT->isStructured()) {
    D = ImplicitParamDecl::Create(Context, 
                                  CurContext,
                                  RefVarLoc,
                                  RefVarInfo,
                                  QualType(cast<StructuredMeshType>(MT),0 ));

  } else if (MT->isRectilinear()) {
    D = ImplicitParamDecl::Create(Context, 
                                  CurContext,
                                  RefVarLoc,
                                  RefVarInfo,
                                  QualType(cast<RectilinearMeshType>(MT),0 ));
  } else if (MT->isUnstructured()) {
    D = ImplicitParamDecl::Create(Context, 
                                  CurContext,
                                  RefVarLoc,
                                  RefVarInfo,
                                  QualType(cast<UnstructuredMeshType>(MT),0 ));

  } else {
    assert(false && "unknown mesh type");
    return false;
  }

  PushOnScopeChains(D, S, true);
  SCLStack.push_back(D);
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
                                     Stmt* Body) {

  SCLStack.pop_back();

  ForallMeshStmt* FS = new (Context) ForallMeshStmt(ElementType, 
                                                    RefVarInfo, 
                                                    MeshInfo, MVD, MT, 
                                                    ForallLoc, 
                                                    Body, Predicate, LParenLoc, RParenLoc);

  // check that LHS mesh field assignment
  // operators do not appear as subsequent RHS values, and
  // perform other semantic checks
  ForallVisitor v(*this, FS);
  v.Visit(Body);

  if (v.error()){
    return StmtError();
  }

  return Owned(FS);
}

/*
StmtResult Sema::ActOnForallArrayStmt(SourceLocation ForAllLoc,
                                      Stmt* Body,
                                      BlockExpr* Block){

  ForAllArrayStmt* FS =
  new (Context) ForAllArrayStmt(Context, ForAllLoc, Body, Block);

  return Owned(FS);
}

namespace{

  class RenderAllVisitor : public StmtVisitor<RenderAllVisitor> {
  public:

    RenderAllVisitor()
    : foundColorAssign_(false){

      for(size_t i = 0; i < 4; ++i){
        foundComponentAssign_[i] = false;
      }

    }

    void VisitStmt(Stmt* S){
      VisitChildren(S);
    }

    void VisitIfStmt(IfStmt* S){
      size_t ic = 0;
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){

        if(Stmt* child = *I){
          if(isa<CompoundStmt>(child) || isa<IfStmt>(child)){
            RenderAllVisitor v;
            v.Visit(child);
            if(v.foundColorAssign()){
              foundColorAssign_ = true;
            }
            else{
              foundColorAssign_ = false;
              break;
            }
          }
          else{
            Visit(child);
          }
          ++ic;
        }
      }
      if(ic == 2){
        foundColorAssign_ = false;
      }
    }

    void VisitBinaryOperator(BinaryOperator* S){
      if(S->getOpcode() == BO_Assign){
        if(DeclRefExpr* dr = dyn_cast<DeclRefExpr>(S->getLHS())){
          if(dr->getDecl()->getName().str() == "color"){
            foundColorAssign_ = true;
          }
        }
        else if(ScoutVectorMemberExpr* vm =
                dyn_cast<ScoutVectorMemberExpr>(S->getLHS())){

          if(DeclRefExpr* dr = dyn_cast<DeclRefExpr>(vm->getBase())){
            if(dr->getDecl()->getName().str() == "color"){
              foundComponentAssign_[vm->getIdx()] = true;
            }
          }
        }
      }
      else{
        VisitChildren(S);
      }
    }

    void VisitChildren(Stmt* S){
      for(Stmt::child_iterator I = S->child_begin(),
          E = S->child_end(); I != E; ++I){
        if(Stmt* child = *I){
          Visit(child);
        }
      }
    }

    bool foundColorAssign(){
      if(foundColorAssign_){
        return true;
      }

      for(size_t i = 0; i < 4; ++i){
        if(!foundComponentAssign_[i]){
          return false;
        }
      }

      return true;
    }

  private:
    bool foundColorAssign_;
    bool foundComponentAssign_[4];
  };

} // end namespace

namespace {
  // scout
  class VolumeRenderAllVisitor : public StmtVisitor<VolumeRenderAllVisitor> {
    public:
      VolumeRenderAllVisitor() {}
      void VisitStmt(Stmt* S){
        VisitChildren(S);
      }
      void VisitIfStmt(IfStmt* S){}
      void VisitBinaryOperator(BinaryOperator* S){}
      void VisitChildren(Stmt* S){}
#ifdef NOTYET
      VolumeRenderAllVisitor()
        foundColorAssign_(false){

          for(size_t i = 0; i < 4; ++i){
            foundComponentAssign_[i] = false;
          }

        }

      void VisitStmt(Stmt* S){
        VisitChildren(S);
      }

      void VisitIfStmt(IfStmt* S){
        size_t ic = 0;
        for(Stmt::child_iterator I = S->child_begin(),
            E = S->child_end(); I != E; ++I){


          if(Stmt* child = *I){
            if(isa<CompoundStmt>(child) || isa<IfStmt>(child)){
              RenderAllVisitor v;
              v.Visit(child);
              if(v.foundColorAssign()){
                foundColorAssign_ = true;
              }
              else{
                foundColorAssign_ = false;
                break;
              }
            }
            else{
              Visit(child);
            }
            ++ic;
          }
        }
        if(ic == 2){
          foundColorAssign_ = false;
        }
      }

      void VisitBinaryOperator(BinaryOperator* S){
        if(S->getOpcode() == BO_Assign){
          if(DeclRefExpr* dr = dyn_cast<DeclRefExpr>(S->getLHS())){
            if(dr->getDecl()->getName().str() == "color"){
              foundColorAssign_ = true;
            }
          }
          else if(ScoutVectorMemberExpr* vm =
              dyn_cast<ScoutVectorMemberExpr>(S->getLHS())){

            if(DeclRefExpr* dr = dyn_cast<DeclRefExpr>(vm->getBase())){
              if(dr->getDecl()->getName().str() == "color"){
                foundComponentAssign_[vm->getIdx()] = true;
              }
            }
          }
        }
        else{
          VisitChildren(S);
        }
      }

      void VisitChildren(Stmt* S){
        for(Stmt::child_iterator I = S->child_begin(),
            E = S->child_end(); I != E; ++I){
          if(Stmt* child = *I){
            Visit(child);
          }
        }
      }

      bool foundColorAssign(){
        if(foundColorAssign_){
          return true;
        }

        for(size_t i = 0; i < 4; ++i){
          if(!foundComponentAssign_[i]){
            return false;
          }
        }

        return true;
      }

    private:
      bool foundColorAssign_;
      bool foundComponentAssign_[4];
#endif
  };

} // end namespace


StmtResult Sema::ActOnRenderAllStmt(SourceLocation RenderAllLoc,
                                    ForAllStmt::ForAllType Type,
                                    const MeshType *MT,
                                    VarDecl* MVD,
                                    IdentifierInfo* LoopVariableII,
                                    IdentifierInfo* MeshII,
                                    SourceLocation LParenLoc,
                                    Expr *Op, SourceLocation RParenLoc,
                                    Stmt* Body,
                                    BlockExpr *Block){

  SCLStack.pop_back();

  RenderAllVisitor v;
  v.Visit(Body);

  if(!v.foundColorAssign()){
    Diag(RenderAllLoc, diag::err_no_color_assignment_renderall);
    return StmtError();
  }

  return Owned(new (Context) RenderAllStmt(Context, Type, MT,
                                           LoopVariableII, MeshII, MVD,
                                           Op, Body, Block,
                                           RenderAllLoc, LParenLoc,
                                           RParenLoc));
}

// scout - Scout Stmts

bool Sema::ActOnForAllLoopVariable(Scope* S,
                                   tok::TokenKind VariableType,
                                   IdentifierInfo* LoopVariableII,
                                   SourceLocation LoopVariableLoc,
                                   IdentifierInfo* MeshII,
                                   SourceLocation MeshLoc){

  // lookup result below
  LookupResult LResult(*this, LoopVariableII, LoopVariableLoc,
                       LookupOrdinaryName);

  LookupName(LResult, S);

  if(LResult.getResultKind() != LookupResult::NotFound){
    Diag(LoopVariableLoc, diag::err_loop_variable_shadows_forall) << LoopVariableII;
    return false;
  }

  LookupResult MResult(*this, MeshII, MeshLoc, LookupOrdinaryName);

  LookupName(MResult, S);

  if(MResult.getResultKind() != LookupResult::Found){
    Diag(MeshLoc, diag::err_unknown_mesh_variable_forall) << MeshII;
    return false;
  }

  NamedDecl* ND = MResult.getFoundDecl();

  if(!isa<VarDecl>(ND)){
    Diag(MeshLoc, diag::err_not_mesh_variable_forall) << MeshII;
    return false;
  }

  VarDecl* VD = cast<VarDecl>(ND);

  const Type* T = VD->getType().getCanonicalType().getTypePtr();

  if(!isa<MeshType>(T)){
    T = VD->getType().getCanonicalType().getNonReferenceType().getTypePtr();
    if(!isa<MeshType>(T)){
      Diag(MeshLoc, diag::err_not_mesh_variable_forall) << MeshII;
      return false;
    }
  }

  MeshType* MT = const_cast<MeshType *>(cast<MeshType>(T));
  UniformMeshType* UMT = cast<UniformMeshType>(MT);

  ImplicitParamDecl* D =
  ImplicitParamDecl::Create(Context, CurContext, LoopVariableLoc,
                            LoopVariableII, QualType(UMT, 0));

  PushOnScopeChains(D, S, true);

  SCLStack.push_back(D);

  return true;
}

bool Sema::ActOnForAllArrayInductionVariable(Scope* S,
                                             IdentifierInfo* InductionVarII,
                                             SourceLocation InductionVarLoc){

  // lookup result below

  LookupResult LResult(*this, InductionVarII, InductionVarLoc,
                       LookupOrdinaryName);

  LookupName(LResult, S);

  if(LResult.getResultKind() != LookupResult::NotFound){
    Diag(InductionVarLoc, diag::err_loop_variable_shadows_forall) <<
    InductionVarII;
    return false;
  }

  ImplicitParamDecl* IV =
  ImplicitParamDecl::Create(Context, CurContext,
                            InductionVarLoc, InductionVarII,
                            Context.IntTy);

  PushOnScopeChains(IV, S, true);

  return true;
}

// scout - Scout Stmts

bool Sema::ActOnRenderAllLoopVariable(Scope* S,
                                      tok::TokenKind VariableType,
                                      IdentifierInfo* LoopVariableII,
                                      SourceLocation LoopVariableLoc,
                                      IdentifierInfo* MeshII,
                                      SourceLocation MeshLoc){


  LookupResult LResult(*this, LoopVariableII, LoopVariableLoc,
                       LookupOrdinaryName);

  LookupName(LResult, S);

  if(LResult.getResultKind() != LookupResult::NotFound){
    Diag(LoopVariableLoc, diag::err_loop_variable_shadows_renderall) << LoopVariableII;
    return false;
  }

  LookupResult MResult(*this, MeshII, MeshLoc, LookupOrdinaryName);

  LookupName(MResult, S);

  if(MResult.getResultKind() != LookupResult::Found){
    Diag(MeshLoc, diag::err_unknown_mesh_variable_renderall) << MeshII;
    return false;
  }

  NamedDecl* ND = MResult.getFoundDecl();

  if(!isa<VarDecl>(ND)){
    Diag(MeshLoc, diag::err_not_mesh_variable_renderall) << MeshII;
    return false;
  }

  VarDecl* VD = cast<VarDecl>(ND);

  const Type* T = VD->getType().getCanonicalType().getNonReferenceType().getTypePtr();

  if(!isa<MeshType>(T)){
    Diag(MeshLoc, diag::err_not_mesh_variable_renderall) << MeshII;
    return false;
  }

  MeshType* MT = const_cast<MeshType *>(cast<MeshType>(T));
  UniformMeshType* UMT = cast<UniformMeshType>(MT);

  ImplicitParamDecl* D =
  ImplicitParamDecl::Create(Context, CurContext, LoopVariableLoc,
                            LoopVariableII, QualType(UMT, 0));


  PushOnScopeChains(D, S, true);

  ImplicitParamDecl* CD =
  ImplicitParamDecl::Create(Context, CurContext, MeshLoc,
                            &Context.Idents.get("color"),
                            Context.Float4Ty);

  PushOnScopeChains(CD, S, true);

  SCLStack.push_back(D);

  return true;
}

const MeshType*
Sema::ActOnRenderAllElementsVariable(Scope* S,
                                     MemberExpr* ME,
                                     tok::TokenKind VariableType,
                                     IdentifierInfo* ElementsVariableII,
                                     SourceLocation ElementsVariableLoc){

  LookupResult LResult(*this, ElementsVariableII, ElementsVariableLoc,
                       LookupOrdinaryName);

  LookupName(LResult, S);

  if(LResult.getResultKind() != LookupResult::NotFound){
    Diag(ElementsVariableLoc,
         diag::err_elements_variable_shadows_renderall) << ElementsVariableII;

    return 0;
  }

  if(SCLStack.empty()){
    Diag(ElementsVariableLoc, diag::err_elements_not_in_forall_renderall);
    return 0;
  }

  VarDecl* MD = SCLStack.back();
  const MeshType* T =
  dyn_cast<MeshType>(MD->getType().getCanonicalType().getTypePtr());

  UniformMeshType* MT = cast<UniformMeshType>(const_cast<MeshType *>(T));

  ImplicitParamDecl* D =
  ImplicitParamDecl::Create(Context, CurContext, ElementsVariableLoc,
                            ElementsVariableII, QualType(MT, 0));

  PushOnScopeChains(D, S, true);

  SCLStack.push_back(D);

  return MT;
}

StmtResult
Sema::ActOnVolumeRenderAllStmt(
        Scope* scope, SourceLocation VolRenLoc,
        SourceLocation L, SourceLocation R,
        IdentifierInfo* MeshII, VarDecl* MeshVD,
        IdentifierInfo* CameraII, SourceLocation CameraLoc,
        MultiStmtArg elts,
        CompoundStmt* body, bool isStmtExpr)
{

  // check camera if one was specified

  VarDecl* CameraVD = 0;

  if (CameraII != 0) {

    LookupResult CameraResult(*this, CameraII, CameraLoc, LookupOrdinaryName);

    LookupName(CameraResult, scope);

    if(CameraResult.getResultKind() != LookupResult::Found){
      Diag(CameraLoc, diag::err_unknown_camera_variable_renderall) << CameraII;
      return false;
    }

    NamedDecl* CameraND = CameraResult.getFoundDecl();

    if(!isa<VarDecl>(CameraND)){
      Diag(CameraLoc, diag::err_not_camera_variable_renderall) << CameraII;
      return false;
    }

    VarDecl* CameraVD = cast<VarDecl>(CameraND);

    QualType CameraVarType = CameraVD->getType();

    if (CameraVarType.getAsString() != "scout::glCamera") {
      Diag(CameraLoc, diag::err_not_camera_variable_renderall) << CameraII;
      return false;
    }

  }

  VolumeRenderAllStmt* vrs = new (Context) VolumeRenderAllStmt(Context,
      VolRenLoc, L, R, MeshII, MeshVD, CameraII, CameraVD, body);

  return Owned(vrs);
}
*/
