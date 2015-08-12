/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2015. Los Alamos National Security, LLC. This software was
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

#ifndef __MESH_TOPOLOGY_H__
#define __MESH_TOPOLOGY_H__

#include <algorithm>
#include <iostream>
#include <array>
#include <vector>
#include <cassert>
#include <unordered_map>

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

namespace scout{

  class MeshTopologyBase{
  public:
    using Id = uint64_t;

    using IdVec = std::vector<Id>;
  
    using ConnVec = std::vector<IdVec>;
  
    struct IdVecHash{
      size_t operator()(const IdVec& v) const{
        size_t h = 0;
        for(Id id : v){
          h |= id;
        }
        return h;
      }
    };
  
    using IdVecMap = std::unordered_map<IdVec, Id, IdVecHash>;
  
    using IndexVec = std::vector<size_t>;

    class Connectivity{
    public:
      static const uint64_t INDEX_MASK = 0x00ffffffffffffff;
      static constexpr int INDEX_BITS = 56;

      Connectivity(){}
    
      void clear(){
        idVec_.clear();
        groupVec_.clear();
      }
    
      void init(){
        groupVec_.push_back(0);
      }
    
      void init(const ConnVec& cv){
        assert(idVec_.empty() && groupVec_.empty());
      
        groupVec_.push_back(0);
      
        size_t m = cv.size();
      
        for(size_t i = 0; i < m; ++i){
          const IdVec& iv = cv[i];
          
          uint64_t n = idVec_.size();

          for(Id id : iv){
            idVec_.push_back(id);
          }
          
          uint64_t n2 = idVec_.size();
          groupVec_.back() |= (n2 - n) << INDEX_BITS;
          groupVec_.push_back(n2);
        }
      }
    
      void resize(IndexVec& numConns){
        clear();
      
        size_t n = numConns.size();
        groupVec_.resize(n + 1);

        uint64_t size = 0;
        groupVec_[0] = size;

        for(size_t i = 1; i < n; ++i){
          groupVec_[i - 1] |= (size - groupVec_[i - 1]) << INDEX_BITS;
          groupVec_[i] = size;
          size += numConns[i];
        }
      
        groupVec_[n] = size;
        idVec_.resize(size);
        std::fill(idVec_.begin(), idVec_.end(), 0);
      }
    
      void endGroup(){
        uint64_t n = groupVec_.back();
        uint64_t n2 = idVec_.size();
        groupVec_.back() |= (n2 - n) << INDEX_BITS;
        groupVec_.push_back(n2);
      }
    
      void push(Id id){
        idVec_.push_back(id);
      }
    
      void dump(){
        std::cout << "=== idVec" << std::endl;
        for(Id id : idVec_){
          std::cout << id << std::endl;
        }
      
        std::cout << "=== groupVec" << std::endl;
        for(Id id : groupVec_){
          std::cout << (INDEX_MASK & id) << "(" << 
            (id >> INDEX_BITS) << ")" << std::endl;
        }
      }
    
      Id* getEntities(size_t index){
        assert(index < groupVec_.size() - 1);
        return idVec_.data() + (groupVec_[index] & INDEX_MASK);
      }

      Id* getEntities(size_t index, size_t& endIndex){
        assert(index < groupVec_.size() - 1);
        uint64_t start = groupVec_[index] & INDEX_MASK;
        endIndex = (groupVec_[index + 1] & INDEX_MASK) - start;
        return idVec_.data() + start;
      }
        
      bool empty(){
        return idVec_.empty();
      }
    
      void set(size_t fromId, size_t toId, size_t pos){
        idVec_[(groupVec_[fromId] & INDEX_MASK) + pos] = toId;
      }
    
      size_t fromSize() const{
        return groupVec_.size() - 1;
      }

      Id* rawIdVec(){
        return idVec_.data();
      }

      Id* rawGroupVec(){
        return groupVec_.data();
      }
    
      void set(ConnVec& conns){
        clear();
      
        groupVec_.resize(conns.size() + 1);
      
        size_t size = 0;
        size_t n = conns.size();
        groupVec_[0] = 0;

        for(size_t i = 1; i < n; i++){
          groupVec_[i - 1] |= (size - groupVec_[0]) << INDEX_BITS;
          groupVec_[i] = size;
          size += conns[i].size();
        }
      
        groupVec_[conns.size()] = size;
      
        idVec_.reserve(size);
        for(auto itr = conns.begin(); itr != conns.end(); ++itr){
          idVec_.insert(idVec_.end(), itr->begin(), itr->end());
        }
      }
    
      IdVec idVec_;
      IdVec groupVec_;
    };

    size_t numEntities(size_t dim){
      size_t size = size_[dim];
      if(size == 0){
        build(dim);
        return size_[dim];
      }

      return size;
    }

    Id* getToIndices(uint32_t fromDim,
                     uint32_t toDim){

      Connectivity& c = getConnectivity(fromDim, toDim);

      if(c.empty()){
        compute(fromDim, toDim);
      }

      return c.rawIdVec();
    }

    Id* getFromIndices(uint32_t fromDim,
                       uint32_t toDim){

      Connectivity& c = getConnectivity(fromDim, toDim);

      if(c.empty()){
        compute(fromDim, toDim);
      }

      return c.rawGroupVec();
    }

    virtual void build(size_t dim) = 0;
    
    virtual void compute(size_t fromDim, size_t toDim) = 0;  
  
    virtual Connectivity&
    getConnectivity(size_t fromDim, size_t toDim) = 0;

  protected:  
    std::vector<size_t> size_;  
  };

  template<class MT>
  class MeshTopology : public MeshTopologyBase{
  public:

    using Float = typename MT::Float;
      
    class Coordinate{
    public:
      Coordinate(){}
    
      Coordinate(std::initializer_list<Float> il){
        static_assert(il.size() == MT::geometricDimension(),
                      "coordinate size mismatch");
        std::copy(il.begin(), il.end(), coordinate_);
      }
    
      template<size_t I>
      Float get(){
        return coordinate_[I];
      }
    
      template<size_t I>
      void set(Float value){
        coordinate_[I] = value;
      }
    
      Coordinate& operator=(std::initializer_list<Float> il){
        size_t i = 0;
        for(Float v : il){
          coordinate_[i++] = v;
        }
      
        return *this;
      }
    
    private:
      using Coordinate_ = std::array<Float, MT::geometricDimension()>;
    
      Coordinate_ coordinate_;
    };
    
    class Geometry{
    public:
      Geometry(){}
    
      void addVertex(Id id, std::initializer_list<Float> il){
        if(id >= coordinates_.size()){
          coordinates_.resize(id + 1);
        }
      
        coordinates_[id] = il;
      }
    
    private:
      using Coordinates_ = std::vector<Coordinate>;
    
      Coordinates_ coordinates_;
    };
  
    class Entity{
    public:
      Entity(MeshTopology& mesh, size_t dim, size_t index=0)
        : mesh_(mesh),
          dim_(dim),
          index_(index),
          endIndex_(mesh_.numEntities(dim_)){
        assert(index_ < endIndex_);
      }
    
      size_t dim(){
        return dim_;
      }
    
      size_t index(){
        return index_;
      }
    
      Id* getEntities(size_t dim){
        Connectivity& c = mesh_.getConnectivity_(dim_, dim);
        assert(!c.empty());
        return c.getEntities(index_);
      }
    
      Entity& operator++(){
        assert(index_ < endIndex_);
        ++index_;
        return *this;
      }
    
      bool end() const{
        return index_ >= endIndex_;
      }
    
      MeshTopology& mesh(){
        return mesh_;
      }
    
    private:
      MeshTopology& mesh_;
      size_t dim_;
      size_t index_;
      size_t endIndex_;
    };
  
    class EntityIterator{
    public:
      EntityIterator(Entity& entity, size_t dim, size_t index=0)
        : mesh_(entity.mesh()),
          dim_(dim),
          index_(index){
        Connectivity& c = mesh_.getConnectivity_(entity.dim(), dim_);
        if(c.empty()){
          mesh_.compute(entity.dim(), dim_);
        }
      
        entities_ = c.getEntities(entity.index(), endIndex_);
        assert(index_ < endIndex_);
      }
    
      EntityIterator(EntityIterator& itr, size_t dim, size_t index=0)
        : mesh_(itr.mesh_),
          dim_(dim),
          index_(index){
        Connectivity& c = mesh_.getConnectivity_(itr.dim_, dim_);
        if(c.empty()){
          mesh_.compute(itr.dim_, dim_);
        }
      
        entities_ = c.getEntities(itr.index_, endIndex_);
        assert(index_ < endIndex_);
      }
    
      size_t dim(){
        return dim_;
      }
    
      size_t index(){
        return entities_[index_];
      }
    
      Id* getEntities(size_t dim){
        Connectivity& c = mesh_.getConnectivity_(dim_, dim);
        assert(!c.empty());
        return c.getEntities(index_);
      }
    
      EntityIterator& operator++(){
        assert(index_ < endIndex_);
        ++index_;
        return *this;
      }
    
      bool end() const{
        return index_ >= endIndex_;
      }
    
    private:
      MeshTopology& mesh_;
      size_t dim_;
      size_t index_;
      size_t endIndex_;
      Id* entities_;
    };
  
    class Cell : public Entity{
    public:
      Cell(MeshTopology& mesh, size_t index=0)
        : Entity(mesh, MT::topologicalDimension(), index){}
    };
  
    class CellIterator : public EntityIterator{
    public:
      CellIterator(Entity& entity, size_t index=0)
        : EntityIterator(entity, MT::topologicalDimension(), index){}
    };
  
    class Vertex : public Entity{
    public:
      Vertex(MeshTopology& mesh, size_t index=0)
        : Entity(mesh, 0, index){}
    };
  
    class VertexIterator : public EntityIterator{
    public:
      VertexIterator(Entity& entity, size_t index=0)
        : EntityIterator(entity, 0, index){}
    };
  
    class Edge : public Entity{
    public:
      Edge(MeshTopology& mesh, size_t index=0)
        : Entity(mesh, 1, index){}
    };
  
    class EdgeIterator : public EntityIterator{
    public:
      EdgeIterator(Entity& entity, size_t index=0)
        : EntityIterator(entity, 1, index){}
    };
  
    class Face : public Entity{
    public:
      Face(MeshTopology& mesh, size_t index=0)
        : Entity(mesh, MT::topologicalDimension() - 1, index){}
    };
  
    class FaceIterator : public EntityIterator{
    public:
      FaceIterator(Entity& entity, size_t index=0)
        : EntityIterator(entity, MT::topologicalDimension() - 1, index){}
    };
  
    MeshTopology(){
      getConnectivity_(MT::topologicalDimension(), 0).init();
      for(size_t i = 0; i <= MT::topologicalDimension(); ++i){
        size_.push_back(0);
      }
    }
  
    void addVertex(Id id, std::initializer_list<Float> il){
      geometry_.addVertex(id, il);
      ++size_[0];
    }
  
    void addCell(Id id, std::initializer_list<Id> il){
      assert(il.size() == 
             MT::numVerticesPerEntity(MT::topologicalDimension()) &&
             "invalid number of vertices per cell");
    
      auto& c = getConnectivity_(MT::topologicalDimension(), 0);
      for(Id id : il){
        c.push(id);
      }
      c.endGroup();
      ++size_[MT::topologicalDimension()];
    }
  
    void build(size_t dim) override{
      //std::cerr << "build: " << dim << std::endl;

      assert(dim <= MT::topologicalDimension());

      size_t verticesPerEntity = MT::numVerticesPerEntity(dim);
      size_t entitiesPerCell =  MT::numEntitiesPerCell(dim);
    
      Connectivity& entityToVertex = getConnectivity_(dim, 0);
    
      IdVec entityVertices(entitiesPerCell * verticesPerEntity);

      Connectivity& cellToEntity =
        getConnectivity_(MT::topologicalDimension(), dim);
    
      ConnVec entityVertexConn;

      size_t entityId = 0;
      size_t maxCellEntityConns = 1;

      Connectivity& cellToVertex =
        getConnectivity_(MT::topologicalDimension(), 0);
      assert(!cellToVertex.empty());

      size_t n = numCells();

      ConnVec cellEntityConn(n);

      IdVecMap entityVerticesMap(n * MT::numEntitiesPerCell(dim)/2);
    
      for(size_t c = 0; c < n; ++c){
        IdVec& conns = cellEntityConn[c]; 
        
        conns.reserve(maxCellEntityConns);
      
        Id* vertices = cellToVertex.getEntities(c);
      
        MT::createEntities(dim, entityVertices, vertices);

        for(size_t i = 0; i < entitiesPerCell; ++i){
          Id* a = &entityVertices[i * verticesPerEntity];
          IdVec ev(a, a + verticesPerEntity);
          std::sort(ev.begin(), ev.end());
        
          auto itr = entityVerticesMap.emplace(std::move(ev), entityId);
          conns.emplace_back(itr.first->second);
        
          if(itr.second){
            entityVertexConn.emplace_back(IdVec(a,
                                                a + verticesPerEntity));
          
            maxCellEntityConns =
              std::max(maxCellEntityConns, cellEntityConn[c].size());
          
            ++entityId;
          }
        }
      }

      cellToEntity.init(cellEntityConn);
      entityToVertex.init(entityVertexConn);

      size_[dim] = entityToVertex.fromSize();
    }
  
    void transpose(size_t fromDim, size_t toDim){
      //std::cerr << "transpose: " << fromDim << " -> " << 
      //  toDim << std::endl;
    
      IndexVec pos(numEntities(fromDim), 0);
    
      for(Entity toEntity(*this, toDim); !toEntity.end(); ++toEntity){
        for(EntityIterator fromItr(toEntity, fromDim); 
            !fromItr.end(); ++fromItr){
          pos[fromItr.index()]++;
        }
      }
    
      Connectivity& outConn = getConnectivity_(fromDim, toDim);
      outConn.resize(pos);
    
      std::fill(pos.begin(), pos.end(), 0);
    
      for(Entity toEntity(*this, toDim); !toEntity.end(); ++toEntity){
        for(EntityIterator fromItr(toEntity, fromDim); 
            !fromItr.end(); ++fromItr){
          outConn.set(fromItr.index(), toEntity.index(), 
                      pos[fromItr.index()]++);
        }
      }
    }
  
    void intersect(size_t fromDim, size_t toDim, size_t dim){
      // std::cerr << "intersect: " << fromDim << " -> " << 
      //  toDim << std::endl;

      Connectivity& outConn = getConnectivity_(fromDim, toDim);
      if(!outConn.empty()){
        return;
      }
    
      ConnVec conns(numEntities(fromDim));
    
      using VisitedVec = std::vector<bool>;
      VisitedVec visited(numEntities(fromDim));
    
      IdVec fromVerts(MT::numVerticesPerEntity(fromDim));
      IdVec toVerts(MT::numVerticesPerEntity(toDim));

      size_t maxSize = 1;    

      for(Entity fromEntity(*this, fromDim); 
          !fromEntity.end(); ++fromEntity){
        IdVec& entities = conns[fromEntity.index()];
        entities.reserve(maxSize);

        Id* ep = fromEntity.getEntities(0);

        std::copy(ep, ep + MT::numVerticesPerEntity(fromDim),
                  fromVerts.begin());
      
        std::sort(fromVerts.begin(), fromVerts.end());
      
        for(EntityIterator fromItr(fromEntity, dim);
            !fromItr.end(); ++fromItr){
          for(EntityIterator toItr(fromItr, toDim);
              !toItr.end(); ++toItr){
            visited[toItr.index()] = false;
          }
        }
      
        for(EntityIterator fromItr(fromEntity, dim);
            !fromItr.end(); ++fromItr){
          for(EntityIterator toItr(fromItr, toDim);
              !toItr.end(); ++toItr){
            if(visited[toItr.index()]){
              continue;
            }
          
            visited[toItr.index()] = true;
          
            if(fromDim == toDim){
              if(fromEntity.index() != toItr.index()){
                entities.push_back(toItr.index());
              }
            }
            else{
              Id* ep = toItr.getEntities(0);

              std::copy(ep, ep + MT::numVerticesPerEntity(toDim),
                        toVerts.begin());
            
              std::sort(toVerts.begin(), toVerts.end());
            
              if(std::includes(fromVerts.begin(), fromVerts.end(),
                               toVerts.begin(), toVerts.end())){
              
                entities.emplace_back(toItr.index());
              }
            }
          }
        }
      
        maxSize = std::max(entities.size(), maxSize);
      }
    
      outConn.init(conns);
    }
  
    void compute(size_t fromDim, size_t toDim) override{
      //std::cerr << "compute: " << fromDim << " -> " << toDim << std::endl;

      Connectivity& outConn = getConnectivity_(fromDim, toDim);
    
      if(!outConn.empty()){
        return;
      }
    
      if(numEntities(fromDim) == 0){
        build(fromDim);
      }
    
      if(numEntities(toDim) == 0){
        build(toDim);
      }
    
      if(numEntities(fromDim) == 0 && numEntities(toDim) == 0){
        return;
      }
    
      if(fromDim == toDim){
        ConnVec connVec(numEntities(fromDim), IdVec(1));
      
        for(Entity entity(*this, fromDim); !entity.end(); ++entity){
          connVec[entity.index()][0] = entity.index();
        }
      
        outConn.set(connVec);
      }
      else if(fromDim < toDim){
        compute(toDim, fromDim);
        transpose(fromDim, toDim);
      }
      else{
        compute(fromDim, 0);
        compute(0, toDim);
        intersect(fromDim, toDim, 0);
      }
    }
  
    size_t numCells(){
      return size_[MT::topologicalDimension()];
    }
  
    size_t numVertices(){
      return size_[0];
    }
  
    size_t numEdges(){
      return size_[1];
    }
  
    size_t numFaces(){
      return size_[MT::topologicalDimension() - 1];
    }

    Connectivity& getConnectivity(size_t fromDim, size_t toDim) override{
      return getConnectivity_(fromDim, toDim);
    }

    Connectivity& getConnectivity_(size_t fromDim, size_t toDim){
      assert(fromDim < topology_.size() && "invalid fromDim");
      auto& t = topology_[fromDim];
      assert(toDim < t.size() && "invalid toDim");
      return t[toDim];
    }
  
    void dump(){
      for(size_t i = 0; i < topology_.size(); ++i){
        auto& ci = topology_[i];
        for(size_t j = 0; j < ci.size(); ++j){
          auto& cj = ci[j];
          std::cout << "------------- " << i << " -> " << j << std::endl;
          cj.dump();
        }
      }
    }
  
  private:
    using Topology_ =
      std::array<std::array<Connectivity, MT::topologicalDimension() + 1>,
      MT::topologicalDimension() + 1>;
    
    Topology_ topology_;
    Geometry geometry_;
  };

} // scout

#endif // __MESH_TOPOLOGY_H__
