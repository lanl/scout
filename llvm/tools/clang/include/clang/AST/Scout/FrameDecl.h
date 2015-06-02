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
 * ###########################################################################
 */

#ifndef __SC_CLANG_FRAME_DECL_H__
#define __SC_CLANG_FRAME_DECL_H__

#include "clang/AST/APValue.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Redeclarable.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Linkage.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <map>

#include "clang/AST/Expr.h"

namespace clang {

  class FrameDecl
    : public TypeDecl, public DeclContext, public Redeclarable<FrameDecl> {

  public:
    struct Var{
      uint32_t varId;
      VarDecl* varDecl;
    };
      
  private:
      
    bool IsCompleteDefinition : 1;

    bool IsBeingDefined : 1;

    bool IsEmbeddedInDeclarator : 1;

    bool IsFreeStanding : 1;

    bool MayHaveOutOfDateDef : 1;

    bool IsCompleteDefinitionRequired : 1;

    mutable bool LoadedFieldsFromExternalStorage : 1;

    SourceLocation RBraceLoc;

    typedef QualifierInfo ExtInfo;
      
    llvm::PointerUnion<TypedefNameDecl*, ExtInfo*> 
    TypedefNameDeclOrQualifier;

    bool hasExtInfo() const{
      return TypedefNameDeclOrQualifier.is<ExtInfo*>();
    }

    ExtInfo *getExtInfo(){
      return TypedefNameDeclOrQualifier.get<ExtInfo*>();
    }

    const ExtInfo *getExtInfo() const{
      return TypedefNameDeclOrQualifier.get<ExtInfo*>();
    }

    typedef std::set<VarDecl*> VarTypeSet;
      
    VarTypeSet varTypes;
      
    typedef std::map<std::string, VarDecl*> VarTypeMap;
      
    VarTypeMap varTypeMap;
    
    SpecObjectExpr* Spec;
    
    typedef std::map<std::string, Var> VarMap;
    
    typedef std::map<VarDecl*, uint32_t> VarIdMap;
    
    using FuncSet = std::set<FunctionDecl*>;
      
    VarMap varMap;
    VarIdMap varIdMap;
      
    uint32_t nextVarId;
      
    FuncSet funcSet_;
      
  protected:
    FrameDecl(const ASTContext &ASTC,
              DeclContext    *DC,
              SourceLocation  L,
              IdentifierInfo *Id,
              FrameDecl      *PrevDecl,
              SourceLocation StartL)
      : TypeDecl(Frame, DC, L, Id, StartL), 
        DeclContext(Frame), Redeclarable(ASTC),
        TypedefNameDeclOrQualifier((TypedefNameDecl*) 0) {
      IsCompleteDefinition            = false;
      IsCompleteDefinitionRequired    = false;
      IsBeingDefined                  = false;
      IsEmbeddedInDeclarator          = false;
      IsFreeStanding                  = false;
      LoadedFieldsFromExternalStorage = false;
      setPreviousDecl(PrevDecl);
      nextVarId = 0;
    }

    typedef Redeclarable<FrameDecl> redeclarable_base;
    
    virtual FrameDecl *getPreviousDeclImpl() {
      return getPreviousDecl();
    }

  public:
    
    void setSpec(SpecObjectExpr* S){
      Spec = S;
    }
      
    void addVarType(VarDecl* D){
      varTypes.insert(D);
      varTypeMap.insert({D->getNameAsString(), D});
    }
    
    bool hasVarType(VarDecl* D){
      return varTypes.find(D) != varTypes.end();
    }
      
    VarDecl* getVarType(const std::string& name){
      auto itr = varTypeMap.find(name);
      if(itr == varTypeMap.end()){
        return 0;
      }
      
      return itr->second;
    }
    
    void addVar(const std::string& name, VarDecl* v){
      uint32_t varId = nextVarId++;
      varMap.insert({name, Var{varId, v}});
      varIdMap.insert({v, varId});
    }
      
    void addVar(const std::string& name, VarDecl* v, uint32_t varId){
      varMap.insert({name, Var{varId, v}});
      varIdMap.insert({v, varId});
    }
    
    void resetVar(const std::string& name, VarDecl* v){
      auto itr = varMap.find(name);
      assert(itr != varMap.end());
      itr->second.varDecl = v;
      varIdMap.insert({v, itr->second.varId});
    }
      
    bool hasVar(const VarDecl* v) const{
      return varIdMap.find(const_cast<VarDecl*>(v)) != varIdMap.end();
    }
    
    void addFunc(FunctionDecl* fd){
      funcSet_.insert(fd);
    }
    
    bool hasFunc(const FunctionDecl* fd) const{
      return funcSet_.find(const_cast<FunctionDecl*>(fd)) != funcSet_.end();
    }
      
    uint32_t getVarId(const VarDecl* v) const{
      auto itr = varIdMap.find(const_cast<VarDecl*>(v));
      assert(itr != varIdMap.end());
      
      return itr->second;
    }
      
    const VarMap& getVarMap() const{
      return varMap;
    }
      
    VarMap& getVarMap(){
      return varMap;
    }
      
    void completeDefinition();

    typedef redeclarable_base::redecl_iterator redecl_iterator;
    using   redeclarable_base::redecls_begin;
    using   redeclarable_base::redecls_end;
    using   redeclarable_base::getPreviousDecl;
    using   redeclarable_base::getMostRecentDecl;
    using   redeclarable_base::isFirstDecl;

    static FrameDecl *Create(const ASTContext &C,
                             DeclContext *DC,
                             SourceLocation StartLoc,
                             SourceLocation IdLoc,
                             IdentifierInfo *Id,
                             FrameDecl* PrevDecl = 0);

    static FrameDecl *CreateDeserialized(const ASTContext &C,
                                         unsigned ID);

    SourceLocation getRBraceLoc() const { return RBraceLoc; }

    void setRBraceLoc(SourceLocation L) { RBraceLoc = L; }

    SourceLocation getInnerLocStart() const { return getLocStart(); }

    SourceLocation getOuterLocStart() const;

    virtual SourceRange getSourceRange() const LLVM_READONLY;
      
    virtual FrameDecl *getNextRedeclaration() const {
      return RedeclLink.getNext(this);
    }
      
    virtual FrameDecl *getMostRecentDeclImpl() {
        return getMostRecentDecl();
    }

    virtual FrameDecl* getCanonicalDecl();

    const FrameDecl* getCanonicalDecl() const {
      return const_cast<FrameDecl*>(this)->getCanonicalDecl();
    }

    bool isThisDeclarationADefinition() const {
      return isCompleteDefinition();
    }

    bool isCompleteDefinition() const {
      return IsCompleteDefinition;
    }

    bool isCompleteDefinitionRequired() const {
      return IsCompleteDefinitionRequired;
    }

    bool isBeingDefined() const {
      return IsBeingDefined;
    }

    bool isEmbeddedInDeclarator() const {
      return IsEmbeddedInDeclarator;
    }

    void setEmbeddedInDeclarator(bool isInDeclarator) {
      IsEmbeddedInDeclarator = isInDeclarator;
    }

    bool isFreeStanding() const { return IsFreeStanding; }

    void setFreeStanding(bool isFreeStanding = true) {
      IsFreeStanding = isFreeStanding;
    }

    bool isDependentType() const { return isDependentContext(); }

    void startDefinition();

    FrameDecl *getDefinition() const;

    void setCompleteDefinition(bool V) { IsCompleteDefinition = V; }

    const char *getKindName() const;

    bool hasNameForLinkage() const {
      return (getDeclName() || getTypedefNameForAnonDecl());
    }

    TypedefNameDecl *getTypedefNameForAnonDecl() const {
      return hasExtInfo() ? 0 :
             TypedefNameDeclOrQualifier.get<TypedefNameDecl*>();
    }

    void setTypedefNameForAnonDecl(TypedefNameDecl *TDD);

    NestedNameSpecifier *getQualifier() const {
      return hasExtInfo() ? 
        getExtInfo()->QualifierLoc.getNestedNameSpecifier() : 0;
    }

    NestedNameSpecifierLoc getQualifierLoc() const {
      return hasExtInfo() ? 
        getExtInfo()->QualifierLoc : NestedNameSpecifierLoc();
    }

    void setQualifierInfo(NestedNameSpecifierLoc QualifierLoc);

    static bool classof(const Decl *D) { return classofKind(D->getKind()); }

    static bool classofKind(Kind K) { return K == Frame; }

    static DeclContext *castToDeclContext(const FrameDecl *D) {
      return static_cast<DeclContext *>(const_cast<FrameDecl*>(D));
    }
    static FrameDecl *castFromDeclContext(const DeclContext *DC) {
      return static_cast<FrameDecl *>(const_cast<DeclContext*>(DC));
    }

    friend class ASTDeclReader;
    friend class ASTDeclWriter;
  };
} // end namespace clang

#endif
