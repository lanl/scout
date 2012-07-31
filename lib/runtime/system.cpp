#include "runtime/system.h"

#include <vector>
#include <sstream>
#include <iostream>

#include <hwloc.h>

#define SC_PREFERRED_ALIGNMENT 64

using namespace std;
using namespace scout;

namespace{

  struct NumaArrayHeader{
#if SC_PREFERRED_ALIGNMENT == 64
    uint64_t size;
#else
    uint32_t size;
#endif
  };

  class SINode{
  public:

    typedef vector<SINode*> SINodeVec;

    enum Kind{
      None,
      Any,
      System,
      NumaNode,
      Socket,
      Cache,
      Group,
      Misc,
      Bridge,
      PCIDevice,
      OSDevice,
      Machine,
      L3,
      L2,
      L1,
      Core,
      ProcessingUnit
    };

    SINode(hwloc_obj_t obj){
      switch(obj->type){
      case HWLOC_OBJ_SYSTEM:
      {
	kind_ = System;
	break;
      }
      case HWLOC_OBJ_MACHINE:
      {
	kind_ = Machine;
	break;
      }
      case HWLOC_OBJ_NODE:
      {
	kind_ = NumaNode;
	break;
      }
      case HWLOC_OBJ_SOCKET:
      {
	kind_ = Socket;
	break;
      }
      case HWLOC_OBJ_CACHE:
      {
	switch(obj->attr->cache.depth){
	  case 1:
	    kind_ = L1;
	    break;
	  case 2:
	    kind_ = L2;
	    break;
	  case 3:
	    kind_ = L3;
	    break;
	  default:
	    kind_ = Cache;
	}
	break;
      }
      case HWLOC_OBJ_CORE:
      {
	kind_ = Core;
	break;
      }
      case HWLOC_OBJ_PU:
      {
	kind_ = ProcessingUnit;
	break;
      }
      case HWLOC_OBJ_GROUP:
      {
	kind_ = Group;
	break;
      }
      case HWLOC_OBJ_MISC:
      {
	kind_ = Misc;
	break;
      }
      case HWLOC_OBJ_BRIDGE:
      {
	kind_ = Bridge;
	break;
      }
      case HWLOC_OBJ_PCI_DEVICE:
      {
	kind_ = PCIDevice;
	break;
      }
      case HWLOC_OBJ_OS_DEVICE:
      {
	kind_ = OSDevice;
	break;
      }
      default:
	kind_ = None;
      }

      if(kind_ == None){
	return;
      }

      switch(kind_){
        case L1:
        case L2:
        case L3:
        case Cache:
	  memory_ = obj->attr->cache.size;
	  break;
        default:
	  memory_ = obj->memory.local_memory;
      }
      
      totalMemory_ = obj->memory.total_memory;

      for(size_t i = 0; i < obj->arity; ++i){
	SINode* c = new SINode(obj->children[i]);
	if(c->kind() != None){
	  addChild(c);
	}
	else{
	  delete c;
	}
      }
    }

    SINode(Kind kind)
      : kind_(kind){
      
    }

    ~SINode(){
      for(size_t i = 0; i < childVec_.size(); ++i){
	delete childVec_[i];
      }
    }

    void addChild(SINode* child){
      childVec_.push_back(child);
    }

    const SINode* child(size_t i) const{
      return childVec_[i];
    }

    size_t childCount() const{
      return childVec_.size();
    }

    Kind kind() const{
      return kind_;
    }

    uint64_t memory() const{
      return memory_;
    }

    uint64_t totalMemory() const{
      return totalMemory_;
    }

    size_t count(Kind kind) const{
      size_t c = 0;
      count_(this, kind, c);
      return c;
    }

    void count_(const SINode* n,
		Kind kind,
		size_t& c) const{

      if(n->kind() == kind){
	++c;
      }

      for(size_t i = 0; i < n->childCount(); ++i){
	count_(n->child(i), kind, c);
      }
    }

    void findKinds(Kind kind, SINodeVec& v) const{
      if(kind_ == kind){
	v.push_back(const_cast<SINode*>(this));
      }

      for(size_t i = 0; i < childCount(); ++i){
	child(i)->findKinds(kind, v);
      }
    }

    size_t totalSockets() const{
      return count(Socket);
    }

    size_t totalNumaNodes() const{
      return count(NumaNode);
    }

    size_t totalCores() const{
      return count(Core);
    }

    size_t totalProcessingUnits() const{
      return count(ProcessingUnit);
    }

    size_t countPerKind(Kind countKind, Kind perKind) const{
      SINodeVec v;
      findKinds(perKind, v);
      
      size_t last = 0;
      for(size_t i = 0; i < v.size(); ++i){
	size_t count = v[i]->count(countKind);
	if(i > 0 && count != last){
	  return 0;
	}
	last = count;
      }
      return last;
    }

    size_t processingUnitsPerCore() const{
      return countPerKind(ProcessingUnit, Core);
    }

    size_t processingUnitsPerNumaNode() const{
      return countPerKind(ProcessingUnit, NumaNode);
    }

    size_t numaNodesPerSocket() const{
      return countPerKind(NumaNode, Socket);
    }

    size_t memoryPerKind(Kind kind) const{
      SINodeVec v;
      findKinds(kind, v);
      
      size_t last = 0;
      for(size_t i = 0; i < v.size(); ++i){
	if(i > 0 && v[i]->memory() != last){
	  return 0;
	}
	last = v[i]->memory();
      }
      return last;
    }

    size_t memoryPerSocket() const{
      return memoryPerKind(Socket);
    }

    size_t memoryPerNumaNode() const{
      return memoryPerKind(NumaNode);
    }

    void toStream(const std::string& indent, ostream& ostr){
      ostr << indent;

      switch(kind_){
      case None:
      case System:
      case Any:
        assert(false && "invalid kind");

      case Socket:
	ostr << "Socket";
	break;
      case Cache:
	ostr << "Cache";
	break;
      case Group:
	ostr << "Group";
	break;
      case Misc:
	ostr << "Misc";
	break;
      case Bridge:
	ostr << "Bridge";
	break;
      case PCIDevice:
	ostr << "PCIDevice";
	break;
      case OSDevice:
	ostr << "OSDevice";
	break;
      case NumaNode:
	ostr << "NumaNode";
	break;
      case Machine:
	ostr << "Machine";
	break;
      case L3:
	ostr << "L3";
	break;
      case L2:
	ostr << "L2";
	break;
      case L1:
	ostr << "L1";
	break;
      case Core:
	ostr << "Core";
	break;
      case ProcessingUnit:
	ostr << "ProcessingUnit";
	break;
      }

      if(memory_ != 0){
	ostr << ": " << memory_;
      }

      ostr << endl;

      for(size_t i = 0; i < childVec_.size(); ++i){
	childVec_[i]->toStream(indent + "  ", ostr);
      }
    }

  protected:
    Kind kind_;
    uint64_t memory_;
    uint64_t totalMemory_;

  private:
    SINodeVec childVec_;
  };

} // end namespace

namespace scout{

  class system_rt_{
  public:
    system_rt_(system_rt* o)
      : o_(o){

      hwloc_topology_init(&topology_);
      hwloc_topology_load(topology_);

      root_ = new SINode(hwloc_get_root_obj(topology_));

      // cache summary information
      totalSockets_ = root_->totalSockets();
      totalNumaNodes_ = root_->totalNumaNodes();
      totalCores_ = root_->totalCores();
      totalProcessingUnits_ = root_->totalProcessingUnits();
      numaNodesPerSocket_ = root_->numaNodesPerSocket();
      memoryPerSocket_ = root_->memoryPerSocket();
      memoryPerNumaNode_ = root_->memoryPerNumaNode();
      processingUnitsPerNumaNode_ = root_->processingUnitsPerNumaNode();
    }

    ~system_rt_(){
      delete root_;
      hwloc_topology_destroy(topology_);
    }

    size_t totalSockets() const{
      return totalSockets_;
    }
  
    size_t totalNumaNodes() const{
      return totalNumaNodes_;
    }
  
    size_t totalCores() const{
      return totalCores_;
    }
  
    size_t totalProcessingUnits() const{
      return totalProcessingUnits_;
    }
  
    size_t processingUnitsPerCore() const{
      return processingUnitsPerCore_;
    }
  
    size_t numaNodesPerSocket() const{
      return numaNodesPerSocket_;
    }
  
    size_t memoryPerSocket() const{
      return memoryPerSocket_;
    }
  
    size_t memoryPerNumaNode() const{
      return memoryPerNumaNode_;
    }

    size_t processingUnitsPerNumaNode() const{
      return processingUnitsPerNumaNode_;
    }

    std::string treeToString() const{
      stringstream ostr;
      root_->toStream("", ostr);
      return ostr.str();
    }

    void* allocArrayOnNumaNode(size_t size, size_t nodeId){
      hwloc_obj_t obj = 
	hwloc_get_obj_by_type(topology_, HWLOC_OBJ_NODE, nodeId);
      
      if(!obj){
	return 0;
      }

      void* m = hwloc_alloc_membind_nodeset(topology_,
					    size + sizeof(NumaArrayHeader),
					    obj->nodeset,
					    HWLOC_MEMBIND_DEFAULT, 0);

      ((NumaArrayHeader*)m)->size = size;

      return (char*)m + sizeof(NumaArrayHeader);
    }
    
    void freeArrayFromNumaNode(void* m){
      void* ms = (char*)m - sizeof(NumaArrayHeader);
      hwloc_free(topology_, ms, ((NumaArrayHeader*)m)->size);
    }

    bool bindThreadToNumaNode(size_t nodeId){
      // the hwloc call below does not work on Apple systems
      // so we simply return true here
#ifdef __APPLE__
      return true;
#endif

      hwloc_obj_t obj = 
	hwloc_get_obj_by_type(topology_, HWLOC_OBJ_NODE, nodeId);

      if(!obj){
	return false;
      }

      int status = hwloc_set_membind_nodeset(topology_,
					     obj->nodeset,
					     HWLOC_MEMBIND_DEFAULT,
					     HWLOC_MEMBIND_THREAD);

      return status != -1;
    }

  private:
    system_rt* o_;
    SINode* root_;
    hwloc_topology_t topology_;
    size_t totalSockets_;
    size_t totalNumaNodes_;
    size_t totalCores_;
    size_t totalProcessingUnits_;
    size_t processingUnitsPerCore_;
    size_t processingUnitsPerNumaNode_;
    size_t numaNodesPerSocket_;
    size_t memoryPerSocket_;
    size_t memoryPerNumaNode_;
  };

} // end namespace scout


system_rt::system_rt(){
  x_ = new system_rt_(this);
}

system_rt::~system_rt(){
  delete x_;
}

size_t system_rt::totalSockets() const{
  return x_->totalSockets();
}

size_t system_rt::totalNumaNodes() const{
  return x_->totalNumaNodes();
}

size_t system_rt::totalCores() const{
  return x_->totalCores();
}

size_t system_rt::totalProcessingUnits() const{
  return x_->totalProcessingUnits();
}

size_t system_rt::processingUnitsPerCore() const{
  return x_->processingUnitsPerCore();
}

size_t system_rt::numaNodesPerSocket() const{
  return x_->numaNodesPerSocket();
}

size_t system_rt::memoryPerSocket() const{
  return x_->memoryPerSocket();
}

size_t system_rt::memoryPerNumaNode() const{
  return x_->memoryPerNumaNode();
}

size_t system_rt::processingUnitsPerNumaNode() const{
  return x_->processingUnitsPerNumaNode();
}

std::string system_rt::treeToString() const{
  return x_->treeToString();
}

void* system_rt::allocArrayOnNumaNode(size_t size, size_t nodeId){
  return x_->allocArrayOnNumaNode(size, nodeId);
}

void system_rt::freeArrayFromNumaNode(void* m){
  x_->freeArrayFromNumaNode(m);
}

bool system_rt::bindThreadToNumaNode(size_t nodeId){
  return x_->bindThreadToNumaNode(nodeId);
}
