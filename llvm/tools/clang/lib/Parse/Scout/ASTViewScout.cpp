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
 * Notes: This file implements the Scout AST viewer functionality used
 * by the scc command for debugging purposes. -ndm
 *
 * #####
 */

#include "clang/Parse/scout/ASTViewScout.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;
using namespace clang;

// Stmt (and Expr) AST nodes can be dumped to text for purposes of debugging.

// When we do this we get output that looks like:

// (CompoundStmt 0x1071d89c8 <forall.sc:14:32, line:23:1>
// (CallExpr 0x1071d82a0 <line:14:32> 'void'
//  (ImplicitCastExpr 0x1071d8288 <col:32> 'void (*)(int, char **)' <FunctionToPointerDecay>
//   (DeclRefExpr 0x1071d8238 <col:32> 'void (int, char **)'
//    lvalue Function 0x106b99470 'scoutInit' 'void (int, char **)'))
//  (ImplicitCastExpr 0x1071d82d8 <col:32> 'int' <LValueToRValue>
//   (DeclRefExpr 0x1071d81e8 <col:32> 'int' lvalue ParmVar 0x1071d7f90 'argc' 'int'))
//  (ImplicitCastExpr 0x1071d82f0 <col:32> 'char **' <LValueToRValue>
// ...
// (ReturnStmt 0x1071d89a8 <line:22:3, col:10>
//  (IntegerLiteral 0x1071d8980 <col:10> 'int' 0)))
//
// SC_TODO: seems like the output has changed and is now like
// CompoundStmt 0x7ff6c3953af8 <forall.sc:65:32, line:85:1>
// |-CallExpr 0x7ff6c3951950 <line:65:32> 'void'
// | |-ImplicitCastExpr 0x7ff6c3951938 <col:32> 'void (*)(enum ScoutDeviceType)' <FunctionToPointerDecay>
// | | `-DeclRefExpr 0x7ff6c39518e8 <col:32> 'void (enum ScoutDeviceType)' lvalue Function 0x7ff6c3946380 '__scrt_init' 'void (enum ScoutDeviceType)'
// | `-DeclRefExpr 0x7ff6c39518c0 <col:32> 'enum ScoutDeviceType' EnumConstant 0x7ff6c392f370 'ScoutGPUNone' 'enum ScoutDeviceType'
//
// SC_TODO: why are we not using -ast-dump or -ast-print???
//
// This file implements a parser that parses this output, creates the
// ViewASTNodes, then walks these nodes to generate GraphViz output
// in order to view the AST graphically.

namespace{

  // A ViewASTNode has a number of children, a head label, and variable
  // number of attrs fields. These fields need to be escaped so that they
  // do not conflict with Graphviz metacharacters.

  class ViewASTNode{
  public:

    ViewASTNode(const string& head)
    : head_(head),
    id_(-1){

      string escapedHead = head;
      escapeStr_(escapedHead);
      head_ = escapedHead;

    }

    ~ViewASTNode(){
      for(ChildVec_::iterator itr = childVec_.begin(),
          itrEnd = childVec_.end(); itr != itrEnd; ++itr){
        delete *itr;
      }
    }

    ViewASTNode* child(size_t i){
      assert(i < childVec_.size() && "ViewASTNode invalid child");

      return childVec_[i];
    }

    void addChild(ViewASTNode* child){
      childVec_.push_back(child);
    }

    size_t childCount() const{
      return childVec_.size();
    }

    const string& head() const{
      return head_;
    }

    void addAttr(const string& attr){
      // to save visual space, skip pointer values which are always hex values
      if(attr.size() >= 2 && attr[0] == '0' && attr[1] == 'x'){
        return;
      }

      string escapedAttr = attr;
      escapeStr_(escapedAttr);
      attrVec_.push_back(escapedAttr);
    }

    size_t attrCount() const{
      return attrVec_.size();
    }

    const string& attr(size_t i) const{
      assert(i < attrVec_.size() && "ViewASTNode invalid attr");

      return attrVec_[i];
    }

    void setId(int id){
      id_ = id;
    }

    int id() const{
      return id_;
    }

  private:

    // escape a string so that it can safely be used in Graphviz
    void escapeStr_(string& str){
      for(size_t i = 0; i < str.size(); ++i){
        if((str[i] == '\"' || str[i] == '<' || str[i] == '>') &&
           (i == 0 || str[i-1] != '\\')){
          str.insert(i, 1, '\\');
        }
      }
    }

    typedef vector<ViewASTNode*> ChildVec_;
    typedef vector<string> AttrVec_;

    string head_;
    ChildVec_ childVec_;
    AttrVec_ attrVec_;
    int id_;
  };

  // parse the AST from text and return the top-level node
  // The following main sections needed to be handled:
  // head: e.g: CompoundStmt
  // line/character information: <...> (can contain nested '<'/'>', so
  // we need to balance these characters
  // single-quoted sequence, e.g: 'void'
  // double-quoted sequence, e.g: "..." - which can span multiple lines
  // and sometimes needs to get recursively parsed
#if 0
  ViewASTNode* viewASTParse(const string& str){
    typedef vector<ViewASTNode*> ViewASTNodeStack;
    ViewASTNodeStack stack;

    // parse the head
    for(size_t i = 0; i < str.size(); ++i){
      if(str[i] == ' ' || str[i] == '\n'){
        continue;
      }

      if(str[i] == '('){
        string head;

        ++i;

        while(str[i] != ' '){
          head += str[i];
          ++i;
        }

        ViewASTNode* n = new ViewASTNode(head);

        if(!stack.empty()){
          stack.back()->addChild(n);
        }

        stack.push_back(n);
      } // end if(str[i] == '(')
      // parse a single-quoted sequence
      else if(str[i] == '\''){
        string attr;
        char last = 0;

        for(;;){
          if((str[i] == '\n' || str[i] == ' ') && last != ' '){
            attr += ' ';
          }
          else{
            attr += str[i];
          }

          if(str[i] == '\'' && last && last != '\\'){
            break;
          }

          last = str[i];
          ++i;
        }

        stack.back()->addAttr(attr);
      } // end if(str[i] == '\'')
      // parse a double-quoted sequence and recursively parse
      // if necessary
      else if(str[i] == '\"'){
        string head;
        char last = 0;

        for(;;){
          if(str[i] == '\n'){
            break;
          }

          if(str[i] == ' '){
            if(last != ' '){
              head += ' ';
            }
          }
          else{
            head += str[i];
          }

          if(str[i] == '\"' && last && last != '\\'){
            break;
          }

          last = str[i];
          ++i;
        }

        if(str[i] == '\"'){
          stack.back()->addAttr(head);
          continue;
        }

        ViewASTNode* n = new ViewASTNode(head);

        if(!stack.empty()){
          stack.back()->addChild(n);
        }

        ++i;

        string body;
        size_t open = 0;
        last = 0;
        for(;;){
          if(str[i] == ' ' || str[i] == '\n'){
            if(last != ' ' && last != '\n'){
              body += ' ';
            }
          }
          else{
            body += str[i];
          }

          if(str[i] == '('){
            ++open;
          }
          else if(str[i] == ')'){
            --open;
            if(open == 0){
              break;
            }
          }

          last = str[i];
          ++i;
        }

        ++i;
        assert(str[i] == '\"');

        n->addChild(viewASTParse(body));
      } // end if(str[i] == '\"')
      else if(str[i] == ')'){
        if(stack.size() == 1){
          return stack[0];
        }

        stack.pop_back();
      } //end if(str[i] == ')')
      // parse a balanced angle-bracket sequence
      else if(str[i] == '<'){
        string attr;

        size_t open = 0;
        for(;;){
          attr += str[i];
          if(str[i] == '>'){
            --open;
            if(!open){
              break;
            }
          }
          else if(str[i] == '<'){
            ++open;
          }
          ++i;
        }

        stack.back()->addAttr(attr);
      } //end if(str[i] == '<')
      // other, parse a non-whitespace sequence
      else{
        string attr;

        for(;;){
          attr += str[i];

          char n = str[i+1];
          if(n == ' ' || n == '\n' || n == ')' || n == '\''){
            break;
          }

          ++i;
        }
        //SC_TODO: fails here if stack empty
        stack.back()->addAttr(attr);
      }
    }

    assert(false && "viewASTParse parse error");
  }

  // we recognize certain Scout AST nodes for purposes of
  // highlighting/coloring them
  bool isScoutASTNode(const string& name){
    return name == "ForAllStmt" || name == "RenderAllStmt" ||
    name == "ScoutVectorMemberExpr" || name == "VolumeRenderAllStmt";
  }

  // output the nodes in the GraphViz graph, initially passed
  // the root node
  void viewASTOutputNodes(ViewASTNode* n, int& id){
    n->setId(id);
    llvm::outs() << "  node" << id << " [label = \"";

    llvm::outs() << "<f0> ";

    llvm::outs() << n->head();

    for(size_t i = 0; i < n->attrCount(); ++i){
      llvm::outs() << " | <f" << i+1 << "> " << n->attr(i);
    }

    llvm::outs() << "\"";

    if(isScoutASTNode(n->head())){
      llvm::outs() << ", style=\"filled\", fillcolor=\"slategray3\"";
    }

    llvm::outs() << "];\n";

    for(size_t i = 0; i < n->childCount(); ++i){
      viewASTOutputNodes(n->child(i), ++id);
    }
  }

  // output the links in the GraphViz graph, initially passed
  // the root node
  void viewASTOutputLinks(ViewASTNode* n){
    for(size_t i = 0; i < n->childCount(); ++i){
      ViewASTNode* ni = n->child(i);

      llvm::outs() << "\"node" << n->id() << "\":f0 -> \"node" <<
      ni->id() << "\":f0;\n";

      viewASTOutputLinks(ni);
    }
  }


  // generate the top-level Graphviz output
  void viewASTOutputGraphviz(ViewASTNode* n){
    llvm::outs() << "digraph G{\n";
    llvm::outs() << "  node [shape = record];\n";
    int id = 0;
    viewASTOutputNodes(n, id);
    viewASTOutputLinks(n);
    llvm::outs() << "}\n";
  }
#endif


} // end namespace

ASTViewScout::ASTViewScout(Sema& sema)
: sema_(sema){

}

ASTViewScout::~ASTViewScout(){

}

// possibly output this DeclGroup in Graphviz
// currently, only if it is from the main file and the body
// of a function/method decl, if we outputted everything we would
// get too much information, i.e: other files included
void ASTViewScout::outputGraphviz(DeclGroupRef declGroup){

  for(DeclGroupRef::iterator itr = declGroup.begin(),
      itrEnd = declGroup.end(); itr != itrEnd; ++itr){

    Decl* decl = *itr;

    if(sema_.getSourceManager().isInMainFile(decl->getLocation())){
      if(FunctionDecl* fd = dyn_cast<FunctionDecl>(decl)){
        if(fd->hasBody()){
          Stmt* body = fd->getBody();
          string str;
          ;
          llvm::raw_string_ostream ostr(str);
          body->dump(ostr, sema_.getSourceManager());
          llvm::outs() << ostr.str() << "\n";
          //SC_TODO: disabled for now as is segfaulting
          //ViewASTNode* root = viewASTParse(ostr.str());
          //viewASTOutputGraphviz(root);
          //delete root;
        }
      }
      else{
        // any other cases to handle?
      }
    }
  }
}
// ENDSCOUTCODE

