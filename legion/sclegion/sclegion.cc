/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
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

#include "sclegion.h"

#include <map>
#include <vector>
#include <string>
#include <cassert>

#include "legion.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace std;
using namespace LegionRuntime;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

namespace{

  static const uint8_t READ_MASK = 0x1;
  static const uint8_t WRITE_MASK = 0x2;  
  static const uint8_t READ_AND_WRITE = 0x3;

  static const legion_variant_id_t VARIANT_ID = 4294967295;

  struct MeshHeader{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t rank;
    uint32_t numFields;
  };
  
  struct MeshFieldInfo{
    size_t region;
    sclegion_field_kind_t fieldKind;
    size_t count;
    size_t fieldId;
  };
    
  size_t fieldKindSize(sclegion_field_kind_t fieldKind){
    switch(fieldKind){
    case SCLEGION_INT32:
      return sizeof(int32_t);
    case SCLEGION_INT64:
      return sizeof(int64_t);
    case SCLEGION_FLOAT:
      return sizeof(float);
    case SCLEGION_DOUBLE:
      return sizeof(double);
    default:
      assert(false && "invalid field kind");
    }
  }

  class Mesh{
  public:

    struct Field{
      string fieldName;
      sclegion_element_kind_t elementKind;
      sclegion_field_kind_t fieldKind;
      legion_field_id_t fieldId;
    };
    
    struct Element{
      Element()
        : count(0){}

      size_t count;
      legion_logical_region_t logicalRegion;
      legion_field_space_t fieldSpace;
      legion_domain_t domain;
      legion_index_space_t indexSpace;
      legion_field_allocator_t fieldAllocator;
    };

    Mesh(legion_runtime_t runtime,
         legion_context_t context,
         size_t rank,
         size_t width,
         size_t height,
         size_t depth)
      : runtime_(runtime),
        context_(context),
        rank_(rank),
        width_(width),
        height_(height),
        depth_(depth),
        nextFieldId_(0){

    }

    void addField(const char* fieldName,
                  sclegion_element_kind_t elementKind,
                  sclegion_field_kind_t fieldKind){
      Field field;
      field.fieldName = fieldName;
      field.elementKind = elementKind;
      field.fieldKind = fieldKind;
      field.fieldId = nextFieldId_++;

      fieldMap_.insert({string(fieldName), field});
      Element& element = elements_[elementKind];
      
      if(element.count == 0){
        element.count = numItems(elementKind);
      }
    }

    void init(){
      for(size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i){
        Element& element = elements_[i];

        size_t count = element.count;

        if(count == 0){
          continue;
        }
        
        legion_rect_1d_t rect;
        rect.lo = {0};
        rect.hi = {int(count) - 1};
        
        element.domain = legion_domain_from_rect_1d(rect);

        element.indexSpace = 
          legion_index_space_create_domain(runtime_, context_, element.domain);

        element.fieldSpace = legion_field_space_create(runtime_, context_);

        element.fieldAllocator =
          legion_field_allocator_create(runtime_, context_, element.fieldSpace);

        for(auto& itr : fieldMap_){
          Field& field = itr.second;
          size_t size = fieldKindSize(field.fieldKind) * count;
          legion_field_allocator_allocate_field(element.fieldAllocator,
                                                size,
                                                field.fieldId);
        }

        element.logicalRegion =
          legion_logical_region_create(runtime_,
                                       context_,
                                       element.indexSpace,
                                       element.fieldSpace);
      }
    }

    size_t numItems(sclegion_element_kind_t elementKind){
      switch(elementKind){
      case SCLEGION_CELL:
        return numCells();
      case SCLEGION_VERTEX:
        return numVertices();
      case SCLEGION_EDGE:
        return numEdges();
      case SCLEGION_FACE:
        return numFaces();
      default:
        assert(false && "invalid element kind");
      }
    }

    size_t numCells(){
      switch(rank_){
      case 1:
        return width_;
      case 2:
        return width_ * height_;
      case 3:
        return width_ * height_ * depth_;
      default:
        assert(false && "invalid rank");
      }
    }

    size_t numVertices(){
      switch(rank_){
      case 1:
        return width_ + 1;
      case 2:
        return (width_ + 1) * (height_ + 1);
      case 3:
        return (width_ + 1) * (height_ + 1) + (depth_ + 1);
      default:
        assert(false && "invalid rank");
      }
    }

    size_t numEdges(){
      switch(rank_){
      case 1:
        return width_;
      case 2:
        return (width_ + 1)*height_ + (height_ + 1)*width_;
      case 3:{
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        return (w1*height_ + h1*width_)*(depth_ + 1) + w1*h1*depth_;
      }
      default:
        assert(false && "invalid rank");
      }
    }

    size_t numFaces(){
      switch(rank_){
      case 1:
        return width_;
      case 2:
        return (width_ + 1)*height_ + (height_ + 1)*width_;
      case 3:{
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        size_t d1 = depth_ + 1;
        return w1*height_*depth_ + h1*width_*depth_ + d1*width_*height_;
      }
      default:
        assert(false && "invalid rank");
      }
    }

    void setMeshHeader(MeshHeader* header){
      header->width = width_;
      header->height = height_;
      header->depth = depth_;
      header->rank = rank_;
    }

    const Field& getField(const string& fieldName){
      auto itr = fieldMap_.find(fieldName);
      assert(itr != fieldMap_.end() && "invalid field");

      return itr->second;
    }

    const Element& getElement(sclegion_element_kind_t elementKind){
      return elements_[elementKind];
    }

  private:
    typedef map<string, Field> FieldMap_;

    legion_runtime_t runtime_;
    legion_context_t context_;

    legion_field_id_t nextFieldId_;

    size_t rank_;
    size_t width_;
    size_t height_;
    size_t depth_;
    
    Element elements_[SCLEGION_ELEMENT_MAX];

    FieldMap_ fieldMap_;
  };
  
  class Launcher{
  public:

    class Region{
    public:
      Region()
        : mode_(0){}

      void addField(const Mesh::Field& field, legion_privilege_mode_t mode){
        fields_.push_back(field);

        switch(mode){
        case READ_ONLY:
           mode_ |= READ_MASK;
           break;
        case WRITE_ONLY:
           mode_ |= WRITE_MASK;
           break;
        case READ_WRITE:
           mode_ = READ_AND_WRITE;
           break;
        default:
          assert(false && "invalid mode");
        }
      }
      
      legion_privilege_mode_t legionMode() const{
        switch(mode_){
        case 0:
          return NO_ACCESS;
        case READ_MASK:
          return READ_ONLY;
        case WRITE_MASK:
          return WRITE_ONLY;
        case READ_AND_WRITE:
          return READ_WRITE;
        default:
          assert(false && "invalid legion mode");
        }
      }

      char* addFieldInfo(char* args, size_t& region, size_t count){
        if(mode_ == 0){
          return args;
        }

        for(auto f : fields_){
          MeshFieldInfo* info = (MeshFieldInfo*)args;
          info->region = region;
          info->fieldKind = f.fieldKind;
          info->count = count;
          info->fieldId = f.fieldId;
          args += sizeof(MeshFieldInfo);
        }

        ++region;
        return args;
      }

      
      void addFieldsToIndexLauncher(legion_index_launcher_t launcher,
                                    unsigned region) const{
        for(auto& f : fields_){
          legion_index_launcher_add_field(launcher, region, f.fieldId, true);
        } 
      }

      void addFieldsToTaskLauncher(legion_task_launcher_t launcher,
                                   unsigned region) const{
        for(auto& f : fields_){
          legion_task_launcher_add_field(launcher, region, f.fieldId, true);
        } 
      }

    private:
      typedef vector<Mesh::Field> Fields_;

      Fields_ fields_;
      uint8_t mode_;
    };

    Launcher(Mesh* mesh, legion_task_id_t taskId)
    : mesh_(mesh),
      taskId_(taskId),
      numFields_(0){}

    void addField(const string& fieldName, legion_privilege_mode_t mode){
      const Mesh::Field& field = mesh_->getField(fieldName);
      regions_[field.elementKind].addField(field, mode);
      ++numFields_;
    }

    void execute(legion_context_t context,
                 legion_runtime_t runtime){
      size_t argsLen = sizeof(MeshHeader) + numFields_ * sizeof(MeshFieldInfo);
      void* argsPtr = malloc(argsLen);
      char* args = (char*)argsPtr;
      MeshHeader* header = (MeshHeader*)args;
      mesh_->setMeshHeader(header);
      header->numFields = numFields_;
      args += sizeof(MeshHeader);

      legion_domain_t domain;
      size_t maxCount = 0;

      size_t region = 0;
      for(size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i){
        const Mesh::Element& element = 
          mesh_->getElement(sclegion_element_kind_t(i));

        args = regions_[i].addFieldInfo(args, region, element.count);

        if(element.count > maxCount){
          maxCount = element.count;
          domain = element.domain;
        }
      }
      
      legion_task_argument_t taskArg = {argsPtr, argsLen};

      legion_argument_map_t map = {new ArgumentMap};

      /*
      legion_index_launcher_t launcher =
        legion_index_launcher_create(taskId_,
                                     domain,
                                     taskArg,
                                     map,
                                     legion_predicate_true(),
                                     false,
                                     0,
                                     0);
      */

      legion_task_launcher_t launcher =
        legion_task_launcher_create(taskId_,
                                    taskArg,
                                    legion_predicate_true(),
                                    0,
                                    0);

      for(size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i){

        const Region& region = regions_[i]; 

        if(region.legionMode() == NO_ACCESS){
          continue;
        }

        const Mesh::Element& element = 
          mesh_->getElement(sclegion_element_kind_t(i));

        /*
        legion_index_launcher_add_region_requirement_logical_region(
          launcher,
          element.logicalRegion,
          0,
          region.legionMode(),
          EXCLUSIVE,
          element.logicalRegion,
          0,
          false);
        */
        
        legion_task_launcher_add_region_requirement_logical_region(
          launcher,
          element.logicalRegion,
          region.legionMode(),
          EXCLUSIVE,
          element.logicalRegion,
          0,
          false);

        //region.addFieldsToIndexLauncher(launcher, i);
        region.addFieldsToTaskLauncher(launcher, i);
      }
      
      //legion_index_launcher_execute(runtime, context, launcher);

      legion_task_launcher_execute(runtime, context, launcher);
    }

  private:
    Mesh* mesh_;
    legion_task_id_t taskId_;
    Region regions_[SCLEGION_ELEMENT_MAX];
    size_t numFields_;
  };

} // end namespace

void
sclegion_init(const char* main_task_name,
              legion_task_pointer_void_t main_task_pointer){
  
  legion_runtime_set_top_level_task_id(0);
  
  legion_task_config_options_t options;
  options.leaf = false;
  options.inner = false;
  options.idempotent = false;

  legion_runtime_register_task_void(0, LOC_PROC, true, true,
                                    VARIANT_ID, options,
                                    main_task_name, main_task_pointer);
}

int
sclegion_start(int argc, char** argv){
  return legion_runtime_start(argc, argv, false);
}

void
sclegion_register_task(legion_task_id_t task_id,
                       const char* task_name,
                       legion_task_pointer_void_t task_pointer){

  legion_task_config_options_t options;
  options.leaf = true;
  options.inner = false;
  options.idempotent = false;

  legion_runtime_register_task_void(task_id, LOC_PROC, false, true,
                                    VARIANT_ID, options,
                                    task_name, task_pointer);
}

sclegion_uniform_mesh_t
sclegion_uniform_mesh_create(legion_runtime_t runtime,
                             legion_context_t context,
                             size_t rank,
                             size_t width,
                             size_t height,
                             size_t depth){
  return {new Mesh(runtime, context, rank, width, height, depth)};
}

void
sclegion_uniform_mesh_add_field(sclegion_uniform_mesh_t mesh,
                                const char* field_name,
                                sclegion_element_kind_t element_kind,
                                sclegion_field_kind_t field_kind){
  static_cast<Mesh*>(mesh.impl)->addField(field_name, element_kind, field_kind);
}

void
sclegion_uniform_mesh_init(sclegion_uniform_mesh_t mesh){
  static_cast<Mesh*>(mesh.impl)->init();
}

void*
sclegion_uniform_mesh_reconstruct(const legion_task_t task,
                                  const legion_physical_region_t* region,
                                  unsigned numRegions,
                                  legion_context_t context,
                                  legion_runtime_t runtime){

  HighLevelRuntime* hr = static_cast<HighLevelRuntime*>(runtime.impl);
  Context* hc = static_cast<Context*>(context.impl);
  Task* ht = static_cast<Task*>(task.impl);

  size_t argsLen = legion_task_get_arglen(task);

  char* args = (char*)legion_task_get_args(task);
  MeshHeader* header = (MeshHeader*)args;
  args += sizeof(MeshHeader);
  
  size_t size = sizeof(void*) * header->numFields + 4 * sizeof(uint32_t);
  void** meshPtr = (void**)malloc(size); 

  void* ret = meshPtr;

  MeshFieldInfo* fi;
  size_t numFields = header->numFields;
  
  for(size_t i = 0; i < numFields; ++i){
    fi = (MeshFieldInfo*)args;

    if(fi->count == 0){
      *meshPtr = 0;
    }
    else{
      PhysicalRegion* hp = 
        static_cast<PhysicalRegion*>(region[fi->region].impl);

      Domain d = 
        hr->get_index_space_domain(*hc,
          ht->regions[fi->region].region.get_index_space());

      Rect<1> r = d.get_rect<1>();
      Rect<1> sr = d.get_rect<1>();
      ByteOffset bo[1];
      
      typedef RegionAccessor<AccessorType::Generic, float> RA;
      RA fm = hp->get_field_accessor(fi->fieldId).typeify<float>();

      *meshPtr = fm.raw_rect_ptr<1>(r, sr, bo);
    }
    
    args += sizeof(MeshFieldInfo);
    ++meshPtr;
  }

  uint32_t* meshTailPtr = (uint32_t*)meshPtr;

  *meshTailPtr = header->width;
  ++meshTailPtr;

  *meshTailPtr = header->height;
  ++meshTailPtr;

  *meshTailPtr = header->depth;
  ++meshTailPtr;

  *meshTailPtr = header->rank;

  return ret;
}

sclegion_uniform_mesh_launcher_t
sclegion_uniform_mesh_create_launcher(sclegion_uniform_mesh_t mesh,
                                      legion_task_id_t task_id){
  return {new Launcher(static_cast<Mesh*>(mesh.impl), task_id)};
}

void
sclegion_uniform_mesh_launcher_add_field(
  sclegion_uniform_mesh_launcher_t launcher,
  const char* field_name,
  legion_privilege_mode_t mode){
  static_cast<Launcher*>(launcher.impl)->addField(field_name, mode);
}

void
sclegion_uniform_mesh_launcher_execute(
  legion_context_t context,
  legion_runtime_t runtime,
  sclegion_uniform_mesh_launcher_t launcher){
  static_cast<Launcher*>(launcher.impl)->execute(context, runtime);
}
