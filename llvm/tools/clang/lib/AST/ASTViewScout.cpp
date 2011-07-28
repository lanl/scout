//===----------------------------------------------------------------------===//
//
// ndm - This file implements the Scout AST viewer functionality used
// by the scc command for debugging purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTViewScout.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

using namespace std;
using namespace clang;

namespace{
  
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
      // to save space, skip pointer values which are always hex values
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
  ViewASTNode* viewASTParse(const string& str){
    typedef vector<ViewASTNode*> ViewASTNodeStack;
    ViewASTNodeStack stack;
    
    for(size_t i = 0; i < str.size(); ++i){
      if(str[i] == ' ' || str[i] == '\n'){
        continue;
      }
      
      if(str[i] == '('){
        string head;
        
        do{
          ++i;
          head += str[i];
        } while(str[i] != ' ');
        
        ViewASTNode* n = new ViewASTNode(head);
        
        if(!stack.empty()){
          stack.back()->addChild(n);
        }
        
        stack.push_back(n);
      }
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
      else{
        string attr;
        
        for(;;){
          attr += str[i];
          
          char n = str[i+1];
          if(n == ' ' || n == '\n' || n == ')'){
            break;
          }
          
          ++i;
        }
        
        stack.back()->addAttr(attr);
      }
    }
    
    assert(false && "viewASTParse parse error");
  }
  
  void viewASTOutputNodes(ViewASTNode* n, int& id){
    n->setId(id);
    cout << "  node" << id << " [label = \"";
    cout << "<f0> " << n->head();
    
    for(size_t i = 0; i < n->attrCount(); ++i){
      cout << " | <f" << i+1 << "> " << n->attr(i);
    }
    
    cout << "\"];" << endl;
    
    for(size_t i = 0; i < n->childCount(); ++i){
      viewASTOutputNodes(n->child(i), ++id);
    }
  }
  
  void viewASTOutputLinks(ViewASTNode* n){
    for(size_t i = 0; i < n->childCount(); ++i){
      ViewASTNode* ni = n->child(i);
      
      cout << "\"node" << n->id() << "\":f0 -> \"node" << 
      ni->id() << "\":f0;" << endl;
      
      viewASTOutputLinks(ni);
    }
  }
  
  void viewASTOutputGraphviz(ViewASTNode* n){
    cout << "digraph G{" << endl;
    cout << "  node [shape = record];" << endl;
    int id = 0;
    viewASTOutputNodes(n, id);
    viewASTOutputLinks(n);
    cout << "}" << endl;
  }
  
} // end namespac

ASTViewScout::ASTViewScout(Sema& sema)
: sema_(sema){

}

ASTViewScout::~ASTViewScout(){

}

void ASTViewScout::outputGraphviz(DeclGroupRef declGroup){

  for(DeclGroupRef::iterator itr = declGroup.begin(),
      itrEnd = declGroup.end(); itr != itrEnd; ++itr){
    
    Decl* decl = *itr;
    
    if(sema_.getSourceManager().isFromMainFile(decl->getLocation())){
      if(FunctionDecl* fd = dyn_cast<FunctionDecl>(decl)){
        if(fd->hasBody()){
          Stmt* body = fd->getBody();
          std::string str;
          llvm::raw_string_ostream ostr(str);
          body->dump(ostr, sema_.getSourceManager());
          std::cerr << ostr.str() << std::endl;
          ViewASTNode* root = viewASTParse(ostr.str());
          viewASTOutputGraphviz(root);
          delete root;
        }
      }
      else{
        // any other decl types to handle?
      }
    }
  }
}
