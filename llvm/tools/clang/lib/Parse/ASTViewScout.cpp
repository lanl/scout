//===----------------------------------------------------------------------===//
//
// SCOUTCODE ndm - This file implements the Scout AST viewer functionality used
// by the scc command for debugging purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ASTViewScout.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

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

// This file implements a parser that parses this output, creates the
// ViewASTNodes, then walks these nodes to generate GraphViz output
// in order to view the AST graphically.

namespace{
  
  // A ViewASTNode has a number of children, a head label, and variable
  // number of attrs fields. These fields need to be escaped so that they
  // do not conlfict with Graphviz metacharacters.
  
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
  // double-quoated sequence, e.g: "..." - which can span multiple lines
  // and sometimes needs to get recursively parsed
  
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
      }
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
      }
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
      }
      else if(str[i] == ')'){
        if(stack.size() == 1){
          return stack[0];
        }
        
        stack.pop_back();
      }
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
      }
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
    cout << "  node" << id << " [label = \"";

    cout << "<f0> ";
    
    cout << n->head();
        
    for(size_t i = 0; i < n->attrCount(); ++i){
      cout << " | <f" << i+1 << "> " << n->attr(i);
    }
    
    cout << "\"";
    
    if(isScoutASTNode(n->head())){
      cout << ", style=\"filled\", fillcolor=\"slategray3\"";
    }
    
    cout << "];" << endl;
    
    for(size_t i = 0; i < n->childCount(); ++i){
      viewASTOutputNodes(n->child(i), ++id);
    }
  }
  
  // output the links in the GraphViz graph, initially passed
  // the root node
  void viewASTOutputLinks(ViewASTNode* n){
    for(size_t i = 0; i < n->childCount(); ++i){
      ViewASTNode* ni = n->child(i);
      
      cout << "\"node" << n->id() << "\":f0 -> \"node" << 
      ni->id() << "\":f0;" << endl;
      
      viewASTOutputLinks(ni);
    }
  }
  
  // generate the top-level Graphviz output
  void viewASTOutputGraphviz(ViewASTNode* n){
    cout << "digraph G{" << endl;
    cout << "  node [shape = record];" << endl;
    int id = 0;
    viewASTOutputNodes(n, id);
    viewASTOutputLinks(n);
    cout << "}" << endl;
  }
  
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
    
    if(sema_.getSourceManager().isFromMainFile(decl->getLocation())){
      if(FunctionDecl* fd = dyn_cast<FunctionDecl>(decl)){
        if(fd->hasBody()){
          Stmt* body = fd->getBody();
          string str;
          llvm::raw_string_ostream ostr(str);
          body->dump(ostr, sema_.getSourceManager());
          cerr << ostr.str() << endl;
          ViewASTNode* root = viewASTParse(ostr.str());
          viewASTOutputGraphviz(root);
          delete root;
        }
      }
      else{
        // any other cases to handle?
      }
    }
  }
}
// ENDSCOUTCODE

