#include "runtime/sysinfo_summary.h"

#include <vector>

#include <hwloc.h>

using namespace std;
using namespace scout;

namespace{

  class SINode{
  public:

    typedef vector<SINode*> SINodeVec;

    enum Kind{
      None,
      Any,
      System,
      Node,
      Socket,
      Cache,
      Group,
      Misc,
      Bridge,
      PCIDevice,
      OSDevice,
      NumaNode,
      Machine,
      Level3,
      Level2,
      Level1,
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
	kind_ = Node;
	break;
      }
      case HWLOC_OBJ_SOCKET:
      {
	kind_ = Socket;
	break;
      }
      case HWLOC_OBJ_CACHE:
      {
	kind_ = Cache;
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

      memory_ = obj->memory.local_memory;
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

  protected:
    Kind kind_;
    uint64_t memory_;
    uint64_t totalMemory_;

  private:
    SINodeVec childVec_;
  };

} // end namespace

namespace scout{

  class sysinfo_summary_rt_{
  public:
    sysinfo_summary_rt_(sysinfo_summary_rt* o)
      : o_(o){

      hwloc_topology_t topology;
      hwloc_topology_init(&topology);
      hwloc_topology_load(topology);

      root_ = new SINode(hwloc_get_root_obj(topology)); 
    
      hwloc_topology_destroy(topology);
    }

    ~sysinfo_summary_rt_(){
      delete root_;
    }

    size_t totalSockets() const{
      return root_->totalSockets();
    }
  
    size_t totalNumaNodes() const{
      return root_->totalNumaNodes();
    }
  
    size_t totalCores() const{
      return root_->totalCores();
    }
  
    size_t totalProcessingUnits() const{
      return root_->totalProcessingUnits();
    }
  
    size_t processingUnitsPerCore() const{
      return root_->processingUnitsPerCore();
    }
  
    size_t numaNodesPerSocket() const{
      return root_->numaNodesPerSocket();
    }
  
    size_t memoryPerSocket() const{
      return root_->memoryPerSocket();
    }
  
    size_t memoryPerNumaNode() const{
      return root_->memoryPerNumaNode();
    }

  private:
    sysinfo_summary_rt* o_;
    SINode* root_;
  };

} // end namespace scout


sysinfo_summary_rt::sysinfo_summary_rt(){
  x_ = new sysinfo_summary_rt_(this);
}

sysinfo_summary_rt::~sysinfo_summary_rt(){
  delete x_;
}

size_t sysinfo_summary_rt::totalSockets() const{
  return x_->totalSockets();
}

size_t sysinfo_summary_rt::totalNumaNodes() const{
  return x_->totalNumaNodes();
}

size_t sysinfo_summary_rt::totalCores() const{
  return x_->totalCores();
}

size_t sysinfo_summary_rt::totalProcessingUnits() const{
  return x_->totalProcessingUnits();
}

size_t sysinfo_summary_rt::processingUnitsPerCore() const{
  return x_->processingUnitsPerCore();
}

size_t sysinfo_summary_rt::numaNodesPerSocket() const{
  return x_->numaNodesPerSocket();
}

size_t sysinfo_summary_rt::memoryPerSocket() const{
  return x_->memoryPerSocket();
}

size_t sysinfo_summary_rt::memoryPerNumaNode() const{
  return x_->memoryPerNumaNode();
}


