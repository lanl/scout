#include "CGBuilder.h"
#include "CGDebugInfo.h"
#include "CGValue.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"


using namespace clang;
using namespace clang::CodeGen;

#if 0
bool CodeGen::CodeGenFunction::isMeshMember(llvm::Argument *arg, 
                                            bool& isSigned, 
                                            std::string& typeStr) {
     
  isSigned = false;

  if(arg->getName().endswith("height")) return false;
  if(arg->getName().endswith("width"))  return false;
  if(arg->getName().endswith("depth"))  return false;
  if(arg->getName().endswith("ptr"))    return false;
  if(arg->getName().endswith("dim_x"))  return false;
  if(arg->getName().endswith("dim_y"))  return false;
  if(arg->getName().endswith("dim_z"))  return false;
    
  typedef MemberMap::iterator MemberIterator;
  for(MemberIterator it = MeshMembers.begin(), end = MeshMembers.end(); it != end; ++it) {

    std::string name = it->first;
    std::string argName = arg->getName();

    size_t pos = argName.find(name);
    size_t len = name.length();
    if (pos == 0 && (argName.length() <= len || std::isdigit(argName[len]))) {
      QualType qt = it->second.second;
      isSigned = qt.getTypePtr()->isSignedIntegerType();
      typeStr = qt.getAsString() + "*";
      //llvm::outs() << "mesh: " << name << " " << typeStr << "\n";
      return true;
    }  
  }
  return false;
}
#endif
