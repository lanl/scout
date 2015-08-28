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

#include "cg-mapper.h"

#include "legion.h"

#include "MeshTopology.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " <<    \
                   __PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " <<     \
                  __PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace scout;

using namespace std;
using namespace LegionRuntime;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

namespace{

  using UniformMesh1d = MeshTopology<UniformMesh1dType>;
  using UniformMesh2d = MeshTopology<UniformMesh2dType>;
  using UniformMesh3d = MeshTopology<UniformMesh3dType>;

  typedef LegionRuntime::HighLevel::HighLevelRuntime HLR;

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
    uint32_t numColors;
    uint32_t numConnections;
  };

  struct MeshFieldInfo{
    size_t region;
    sclegion_field_kind_t fieldKind;
    size_t count;
    size_t fieldId;
  };

  struct ArrayHeader{
    uint32_t fromDim;
    uint32_t toDim;
    uint64_t fromSize;
    uint64_t toSize;
  };

  size_t fieldKindSize(sclegion_field_kind_t fieldKind) {
    switch (fieldKind) {
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

  struct Field {
    string fieldName;
    sclegion_element_kind_t elementKind;
    sclegion_field_kind_t fieldKind;
    legion_field_id_t fieldId;
    legion_privilege_mode_t mode;
  };

  typedef vector<Field> FieldVec;

  size_t getEndOffset(MeshHeader *header, size_t index);

  class Mesh {
  public:

    struct Element {
      Element() :
        count(0) {
      }

      size_t count;
      LogicalRegion logicalRegion;
      LogicalPartition logicalPartition;
      IndexPartition indexPartition;
      FieldSpace fieldSpace;
      Domain domain;
      IndexSpace indexSpace;
      FieldAllocator fieldAllocator;
      Domain colorDomain;
      DomainColoring coloring;
    };

    struct TopologyArray{
      size_t size;
      LogicalRegion logicalRegion;
      LogicalPartition logicalPartition;
      IndexPartition indexPartition;
      FieldSpace fieldSpace;
      Domain domain;
      IndexSpace indexSpace;
      FieldAllocator fieldAllocator;
      Domain colorDomain;
      DomainColoring coloring;
    };

    Mesh(HighLevelRuntime* runtime,
         Context context,
         size_t rank,
         size_t width,
         size_t height,
         size_t depth,
         MeshTopologyBase* topology) :
      runtime_(runtime), context_(context), nextFieldId_(0),
      rank_(rank), width_(width), height_(height), depth_(depth),
      topology_(topology){

    }

    void addField(const char* fieldName, sclegion_element_kind_t elementKind,
                  sclegion_field_kind_t fieldKind) {
      Field field;
      field.fieldName = fieldName;
      field.elementKind = elementKind;
      field.fieldKind = fieldKind;
      field.fieldId = nextFieldId_++;

      fieldMap_.insert( { string(fieldName), field });
      Element& element = elements_[elementKind];

      if (element.count == 0) {
        element.count = numItems(elementKind);
      }
    }

    void getFields(sclegion_element_kind_t elementKind, FieldVec& fields) {
      for (auto& itr : fieldMap_) {
        const Field& field = itr.second;

        if (field.elementKind == elementKind) {
          fields.push_back(field);
        }
      }
    }

    void init() {
      for (size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i) {
        Element& element = elements_[i];

        size_t count = element.count;

        if (count == 0) {
          continue;
        }

        Rect<1> rect(Point<1>(0), Point<1>(count - 1));
        Rect<1> subRect;

        element.domain = Domain::from_rect<1>(rect);

        element.indexSpace = runtime_->create_index_space(context_,
                                                          element.domain);

        element.fieldSpace = runtime_->create_field_space(context_);

        element.fieldAllocator = runtime_->create_field_allocator(context_,
                                                                  element.fieldSpace);

        for (auto& itr : fieldMap_) {
          Field& field = itr.second;

          element.fieldAllocator.allocate_field(fieldKindSize(field.fieldKind),
                                                field.fieldId);
        }

        element.logicalRegion = 
          runtime_->create_logical_region(context_,
                                          element.indexSpace, element.fieldSpace);

        size_t numSubregions = 1;

        MeshHeader header;
        header.width = width_;
        header.height = height_;
        header.depth = depth_;
        header.rank = rank_;
        header.numColors = numSubregions;

        element.colorDomain = Domain::from_rect<1>(Rect<1>(Point<1>(0), Point<1>(numSubregions - 1)));

        element.coloring[0] = Domain::from_rect<1>(Rect<1>(Point<1>(0),
                                                           Point<1>(element.count-1)));

        element.indexPartition = 
          runtime_->create_index_partition(context_, element.indexSpace, element.colorDomain,
                                           element.coloring, true);

        element.logicalPartition = 
          runtime_->get_logical_partition(context_, element.logicalRegion, element.indexPartition);

      }
    }

    void createTopologyArray(TopologyArray& array, size_t size){
      array.size = size;

      Rect<1> rect(Point<1>(0), Point<1>(size - 1));
      Rect<1> subRect;

      array.domain = Domain::from_rect<1>(rect);
      array.indexSpace = runtime_->create_index_space(context_,
                                                      array.domain);

      array.fieldSpace = runtime_->create_field_space(context_);

      array.fieldAllocator = runtime_->create_field_allocator(context_,
                                                              array.fieldSpace);

      array.fieldAllocator.allocate_field(sizeof(uint64_t), 0);

      array.logicalRegion = 
        runtime_->create_logical_region(context_,
                                        array.indexSpace, array.fieldSpace);
        
      size_t numSubregions = 1;

      array.colorDomain = Domain::from_rect<1>(Rect<1>(Point<1>(0), Point<1>(numSubregions - 1)));

      array.coloring[0] = Domain::from_rect<1>(Rect<1>(Point<1>(0), Point<1>(array.size - 1)));

      array.indexPartition = 
        runtime_->create_index_partition(context_, array.indexSpace, array.colorDomain,
                                         array.coloring, true);
        
      array.logicalPartition = 
        runtime_->get_logical_partition(context_, array.logicalRegion, array.indexPartition);
    }

    size_t numItems(sclegion_element_kind_t elementKind) {
      switch (elementKind) {
      case SCLEGION_CELL:
        return getNumCells(rank_);
      case SCLEGION_VERTEX:
        return getNumVertices(rank_);
      case SCLEGION_EDGE:
        return getNumEdges(rank_);
      case SCLEGION_FACE:
        return getNumFaces(rank_);
      default:
        assert(false && "invalid element kind");
      }
    }

    size_t numCells() {
      return getNumCells(rank_);
    }
    size_t numVertices() {
      return getNumVertices(rank_);
    }
    size_t numEdges() {
      return getNumEdges(rank_);
    }
    size_t numFaces() {
      return getNumFaces(rank_);
    }

    //for 2-d partition we want # of elements in a col.
    //for 3-d partition we want # if elements in a row x col "face"
    size_t numSubItems(sclegion_element_kind_t elementKind) {
      if (rank_ == 1) {
        return 1;
      } else {
        switch (elementKind) {
        case SCLEGION_CELL:
          return getNumCells(rank_ - 1);
        case SCLEGION_VERTEX:
          return getNumVertices(rank_ - 1);
          // SC_TODO: these cases get messy...
        case SCLEGION_EDGE:
          assert(false && "2/3D partition not working on edges");
        case SCLEGION_FACE:
          assert(false && "2/3D partition not working on faces");
        default:
          assert(false && "invalid element kind");
        }
      }
    }

    void setMeshHeader(MeshHeader* header) {
      header->width = width_;
      header->height = height_;
      header->depth = depth_;
      header->rank = rank_;
    }

    const Field& getField(const string& fieldName) {
      auto itr = fieldMap_.find(fieldName);
      assert(itr != fieldMap_.end() && "invalid field");

      return itr->second;
    }

    Element& getElement(sclegion_element_kind_t elementKind) {
      return elements_[elementKind];
    }

    MeshTopologyBase* topology(){
      return topology_;
    }

  protected:
    size_t getNumCells(size_t rank) {
      switch (rank) {
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

    size_t getNumVertices(size_t rank) {
      switch (rank) {
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

    size_t getNumEdges(size_t rank) {
      switch (rank) {
      case 1:
        return width_;
      case 2:
        return (width_ + 1) * height_ + (height_ + 1) * width_;
      case 3: {
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        return (w1 * height_ + h1 * width_) * (depth_ + 1) + w1 * h1 * depth_;
      }
      default:
        assert(false && "invalid rank");
      }
    }

    size_t getNumFaces(size_t rank) {
      switch (rank) {
      case 1:
        return width_;
      case 2:
        return (width_ + 1) * height_ + (height_ + 1) * width_;
      case 3: {
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        size_t d1 = depth_ + 1;
        return w1 * height_ * depth_ + h1 * width_ * depth_
          + d1 * width_ * height_;
      }
      default:
        assert(false && "invalid rank");
      }
    }

  private:
    typedef map<string, Field> FieldMap_;

    HighLevelRuntime* runtime_;
    Context context_;

    legion_field_id_t nextFieldId_;

    size_t rank_;
    size_t width_;
    size_t height_;
    size_t depth_;

    Element elements_[SCLEGION_ELEMENT_MAX];

    FieldMap_ fieldMap_;
    MeshTopologyBase* topology_;
  };

  class Launcher {
  public:

    class Region {
    public:
      Region() :
        mode_(0) {
      }

      void setFields(const FieldVec& fields) {
        fields_ = fields;
      }

      void addField(const Field& field, legion_privilege_mode_t mode) {
        fieldSet_.insert(field.fieldName);

        printf("region add Field %d %d\n", field.fieldId, mode);
        fields_[field.fieldId].mode = mode;

        switch (mode) {
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

      legion_privilege_mode_t legionMode() const {
        switch (mode_) {
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

      bool isFieldUsed(size_t index) {
        auto f = fields_[index];
        if (fieldSet_.find(f.fieldName) != fieldSet_.end()) {
          return true;
        } else {
          return false;
        }
      }

      sclegion_field_kind_t getFieldKind(size_t index) {
        return fields_[index].fieldKind;
      }


      char* addFieldInfo(char* args, size_t region, size_t count){
        for(auto f : fields_){
          MeshFieldInfo* info = (MeshFieldInfo*)args;
          info->region = region;
          info->fieldKind = f.fieldKind;

          if(fieldSet_.find(f.fieldName) != fieldSet_.end()){
            info->count = count;
          }
          else{
            info->count = 0;
          }

          info->fieldId = f.fieldId;
          args += sizeof(MeshFieldInfo);
        }

        return args;
      }


      void addFieldsToIndexLauncher(IndexLauncher& launcher,
                                    unsigned region) const {
        for(auto& f : fields_){
          launcher.region_requirements[region].add_field(f.fieldId);
        }
      }

      void addFieldsToIndexLauncher(IndexLauncher& launcher,
                                    unsigned region, legion_privilege_mode_t mode) const {
        for (int i = 0; i < fields_.size(); i++) {
          if (fields_[i].mode == mode) {
            printf("addFieldsToIndexLauncher %d %d region %u\n",
                   mode, fields_[i].fieldId,region);
            launcher.region_requirements[region].add_field(fields_[i].fieldId);
          }
        }
      }


    private:
      typedef set<string> FieldSet_;
      typedef vector<Field> Fields_;

      FieldSet_ fieldSet_;
      Fields_ fields_;
      uint8_t mode_;
    };
  
    Launcher(Mesh* mesh, legion_task_id_t taskId) :
      mesh_(mesh), taskId_(taskId), numFields_(0) {

      for (size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i) {
        FieldVec fields;
        mesh->getFields(sclegion_element_kind_t(i), fields);
        regions_[i].setFields(fields);
        numFields_ += fields.size();
      }
    }

    void addField(const string& fieldName, legion_privilege_mode_t mode){
      const Field& field = mesh_->getField(fieldName);
      regions_[field.elementKind].addField(field, mode);
    }

    void execute(legion_context_t ctx, legion_runtime_t rt) {
      using TopologyArray = Mesh::TopologyArray;

      using Id = uint64_t;

      MeshTopologyBase* topology = mesh_->topology();

      size_t d = topology->topologicalDimension();

      size_t numConnections = 0;

      FieldSpace fieldSpace;
    
      using TopologyArrayVec = vector<TopologyArray>;
      TopologyArrayVec topologyArrays;
    
      using ArrayHeaderVec = vector<ArrayHeader>;
      ArrayHeaderVec arrayHeaders;

      for(size_t i = 0; i < d; ++i){
        for(size_t j = 0; j < d; ++j){
          if(i == j){
            continue;
          }
        
          Id* fromIndices;
          size_t fromSize;
        
          Id* toIndices;
          size_t toSize;

          topology->getConnectivityRaw(i, j, fromIndices, fromSize, toIndices, toSize);

          if(fromSize == 0){
            continue;
          }

          ArrayHeader arrayHeader;
          arrayHeader.fromDim = i;
          arrayHeader.toDim = j;
          arrayHeader.fromSize = fromSize;
          arrayHeader.toSize = toSize;
          arrayHeaders.emplace_back(move(arrayHeader));
        
          TopologyArray fromArray;
          mesh_->createTopologyArray(fromArray, fromSize);

          TopologyArray toArray;
          mesh_->createTopologyArray(toArray, toSize);          
        
          topologyArrays.emplace_back(move(fromArray));
          topologyArrays.emplace_back(move(toArray));
        }
      }

      HighLevelRuntime* runtime = static_cast<HighLevelRuntime*>(rt.impl);
      Context context = static_cast<Context>(ctx.impl);

      size_t argsLen = sizeof(MeshHeader) + numFields_ * sizeof(MeshFieldInfo) + 
        arrayHeaders.size() * sizeof(ArrayHeader);

      void* argsPtr = malloc(argsLen);
      char* args = (char*) argsPtr;
      MeshHeader* header = (MeshHeader*) args;
      mesh_->setMeshHeader(header);
      header->numFields = numFields_;
      header->numConnections = arrayHeaders.size();
      args += sizeof(MeshHeader);

      for(size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i){
        Mesh::Element& element = 
          mesh_->getElement(sclegion_element_kind_t(i));

        args = regions_[i].addFieldInfo(args, i, element.count);

        if(element.count == 0){
          continue;
        }
      }
     
      memcpy(args, arrayHeaders.data(), sizeof(ArrayHeader)*arrayHeaders.size());
    
      ArgumentMap argMap;

      Mesh::Element& element = mesh_->getElement(sclegion_element_kind_t(0));

      IndexLauncher launcher(taskId_, element.colorDomain,
                             TaskArgument(argsPtr, argsLen), argMap);

      for (size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i) {

        const Region& region = regions_[i];

        if (region.legionMode() == NO_ACCESS ) {
          continue;
        }

        const Mesh::Element& element = mesh_->getElement(sclegion_element_kind_t(i));

        launcher.add_region_requirement(RegionRequirement(element.logicalPartition,
                                                          0/*projection ID*/,
                                                          region.legionMode(),
                                                          EXCLUSIVE, element.logicalRegion));
             
      }

      for (size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i) {
        const Region& region = regions_[i];

        if(region.legionMode() == NO_ACCESS){
          continue;
        }

        region.addFieldsToIndexLauncher(launcher, i);

      }
    
      for(TopologyArray& array : topologyArrays){
        RegionRequirement req(array.logicalPartition,
                              0/*projection ID*/,
                              READ_ONLY,
                              EXCLUSIVE, array.logicalRegion);

        req.add_field(0);
        launcher.add_region_requirement(req);
      }

      runtime->execute_index_space(context, launcher);
    }
  
  private:
    Mesh* mesh_;
    legion_task_id_t taskId_;
    Region regions_[SCLEGION_ELEMENT_MAX];
    size_t numFields_;
  };

} // namespace

void sclegion_init(const char* main_task_name,
                   legion_task_pointer_void_t main_task_pointer) {

  HLR::set_top_level_task_id(0);

  legion_task_config_options_t options;
  options.leaf = false;
  options.inner = false;
  options.idempotent = false;

  legion_runtime_register_task_void(0, LOC_PROC, true, true, VARIANT_ID,
                                    options, main_task_name, main_task_pointer);
}

int sclegion_start(int argc, char** argv) {
  // register custom mapper
  HLR::set_registration_callback(mapperRegistration);

  return HLR::start(argc, argv);
}

void sclegion_register_task(legion_task_id_t task_id, const char* task_name,
                            legion_task_pointer_void_t task_pointer) {

  legion_task_config_options_t options;
  options.leaf = true;
  options.inner = false;
  options.idempotent = false;

  legion_runtime_register_task_void(task_id, LOC_PROC, true, true, VARIANT_ID,
                                    options, task_name, task_pointer);
}

sclegion_uniform_mesh_t
sclegion_uniform_mesh_create(legion_runtime_t rt,
                             legion_context_t ctx,
                             size_t rank,
                             size_t width,
                             size_t height,
                             size_t depth,
                             void* topology) {

  HighLevelRuntime* runtime = static_cast<HighLevelRuntime*>(rt.impl);
  Context context = static_cast<Context>(ctx.impl);

  auto tb = static_cast<MeshTopologyBase*>(topology);

  return {new Mesh(runtime, context, rank, width, height, depth, tb)};
}

void sclegion_uniform_mesh_add_field(sclegion_uniform_mesh_t mesh,
                                     const char* field_name, sclegion_element_kind_t element_kind,
                                     sclegion_field_kind_t field_kind) {
  static_cast<Mesh*>(mesh.impl)->addField(field_name, element_kind, field_kind);
}

void sclegion_uniform_mesh_init(sclegion_uniform_mesh_t mesh) {
  static_cast<Mesh*>(mesh.impl)->init();
}

void*
sclegion_uniform_mesh_reconstruct(const legion_task_t task,
                                  const legion_physical_region_t* region, unsigned numRegions,
                                  legion_context_t context, legion_runtime_t runtime) {

  HighLevelRuntime* hr = static_cast<HighLevelRuntime*>(runtime.impl);
  Context hc = static_cast<Context>(context.impl);
  Task* ht = static_cast<Task*>(task.impl);

  char* args = (char*) legion_task_get_args(task);
  MeshHeader* header = (MeshHeader*) args;
  args += sizeof(MeshHeader);

  size_t size = sizeof(void*) * (header->numFields + 1) + 10 * sizeof(uint32_t);
  void** meshPtr = (void**) malloc(size);

  void* ret = meshPtr;

  MeshFieldInfo* fi;
  size_t numFields = header->numFields;
  size_t numConnections = header->numConnections;

  for (size_t i = 0; i < numFields; ++i) {
    fi = (MeshFieldInfo*) args;

    if (fi->count == 0) {
      *meshPtr = 0;
    } else {
      printf("region %d\n",fi->region); 
      PhysicalRegion* hp = static_cast<PhysicalRegion*>(region[fi->region].impl);

      IndexSpace is = ht->regions[fi->region].region.get_index_space();
      Domain d = hr->get_index_space_domain(hc, is);

      Rect<1> r = d.get_rect<1>();
      Rect<1> sr;
      ByteOffset bo[1];

      switch(fi->fieldKind) {
      case SCLEGION_INT32: {
        RegionAccessor<AccessorType::Generic, int32_t> fm;
        fm = hp->get_field_accessor(fi->fieldId).typeify<int32_t>();
        *meshPtr = fm.raw_rect_ptr<1>(r, sr, bo);
        break;
      }
      case SCLEGION_INT64: {
        RegionAccessor<AccessorType::Generic, int64_t> fm;
        fm = hp->get_field_accessor(fi->fieldId).typeify<int64_t>();
        *meshPtr = fm.raw_rect_ptr<1>(r, sr, bo);
        break;
      }
      case SCLEGION_FLOAT: {
        RegionAccessor<AccessorType::Generic, float> fm;
        fm = hp->get_field_accessor(fi->fieldId).typeify<float>();
        *meshPtr = fm.raw_rect_ptr<1>(r, sr, bo);
        break;
      }
      case SCLEGION_DOUBLE: {
        RegionAccessor<AccessorType::Generic, double> fm;
        fm = hp->get_field_accessor(fi->fieldId).typeify<double>();
        *meshPtr = fm.raw_rect_ptr<1>(r, sr, bo);
        break;
      }
      default:
        assert(false && "invalid field kind");
      }
    }

    args += sizeof(MeshFieldInfo);
    ++meshPtr;
  }

  MeshTopologyBase* topology;

  switch(header->rank){
  case 1:
    topology = new UniformMesh1d;
    break;
  case 2:
    topology = new UniformMesh2d;
    break;
  case 3:
    topology = new UniformMesh3d;
    break;
  default:
    assert(false && "invalid rank");
  }

  ArrayHeader* ah;
  for(size_t i = 0; i < numConnections * 2; i += 2){
    ah = (ArrayHeader*)args;

    uint64_t* fromIndices;
    uint64_t* toIndices;

    for(size_t j = 0; j < 2; ++j){
      size_t regionIndex = SCLEGION_ELEMENT_MAX + i/2 + j;

      PhysicalRegion* hp = static_cast<PhysicalRegion*>(region[regionIndex].impl);
                          
      IndexSpace is = ht->regions[regionIndex].region.get_index_space();
      Domain d = hr->get_index_space_domain(hc, is);
    
      Rect<1> r = d.get_rect<1>();
      Rect<1> sr;
      ByteOffset bo[1];

      RegionAccessor<AccessorType::Generic, uint64_t> fm;
      fm = hp->get_field_accessor(fi->fieldId).typeify<uint64_t>();
      
      if(j == 0){
        fromIndices = fm.raw_rect_ptr<1>(r, sr, bo);
      }
      else{
        toIndices = fm.raw_rect_ptr<1>(r, sr, bo);
      }
    }

    topology->setConnectivityRaw(ah->fromDim, ah->toDim,
                                 fromIndices, ah->fromSize,
                                 toIndices, ah->toSize);
    
    args += sizeof(ArrayHeader);
  }

  *((MeshTopologyBase**)meshPtr) = topology;

  meshPtr++;

  uint64_t* meshTailPtr = (uint64_t*) meshPtr;

  *meshTailPtr = header->width;
  ++meshTailPtr;

  *meshTailPtr = header->height;
  ++meshTailPtr;

  *meshTailPtr = header->depth;
  ++meshTailPtr;

  *meshTailPtr = header->rank;
  ++meshTailPtr;

  *meshTailPtr = 0;
  ++meshTailPtr;

  *meshTailPtr = 0;
  ++meshTailPtr;

  *meshTailPtr = 0;
  ++meshTailPtr;

  *meshTailPtr = header->width;
  ++meshTailPtr;

  *meshTailPtr = header->height;
  ++meshTailPtr;

  *meshTailPtr = header->depth;
  ++meshTailPtr;

  return ret;
}

sclegion_uniform_mesh_launcher_t sclegion_uniform_mesh_create_launcher(sclegion_uniform_mesh_t mesh,
                                                                       legion_task_id_t task_id) {
  return {new Launcher(static_cast<Mesh*>(mesh.impl), task_id)};
}

void sclegion_uniform_mesh_launcher_add_field(sclegion_uniform_mesh_launcher_t launcher,
                                              const char* field_name,
                                              legion_privilege_mode_t mode) {
  printf("uniform_mesh_launcher_add_field %s %d\n",field_name, mode);
  static_cast<Launcher*>(launcher.impl)->addField(field_name, mode);
}

void sclegion_uniform_mesh_launcher_execute(legion_context_t context,
                                            legion_runtime_t runtime,
                                            sclegion_uniform_mesh_launcher_t launcher) {
  static_cast<Launcher*>(launcher.impl)->execute(context, runtime);
}
