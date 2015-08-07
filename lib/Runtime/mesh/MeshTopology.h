#ifndef __MESH_TOPOLOGY_H__
#define __MESH_TOPOLOGY_H__

#include <algorithm>
#include <iostream>
#include <array>
#include <vector>
#include <cassert>
#include <unordered_map>

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
          groupVec_.back() |= (n2 - n) << 56;
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
          groupVec_[i - 1] |= (size - groupVec_[i - 1]) << 56;
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
        groupVec_.back() |= (n2 - n) << 56;
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
          std::cout << id << std::endl;
        }
      }
    
      Id* getEntities(size_t index){
        assert(index < groupVec_.size() - 1);
        return idVec_.data() + (groupVec_[index] & INDEX_MASK);
      }

      Id* getEntities(size_t index, size_t& endIndex){
        assert(index < groupVec_.size() - 1);
        endIndex = groupVec_[index + 1] - groupVec_[index];
        return idVec_.data() + (groupVec_[index] & INDEX_MASK);
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
          groupVec_[i - 1] |= (size - groupVec_[0]) << 56;
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
      assert(il.size() == MT::numVertices(MT::topologicalDimension()) &&
             "invalid number of vertices per cell");
    
      auto& c = getConnectivity_(MT::topologicalDimension(), 0);
      for(Id id : il){
        c.push(id);
      }
      c.endGroup();
      ++size_[MT::topologicalDimension()];
    }
  
    void build(size_t dim) override{
      assert(dim <= MT::topologicalDimension());
    
      Connectivity& entityToVertex = getConnectivity_(dim, 0);

      Connectivity& cellToEntity =
        getConnectivity_(MT::topologicalDimension(), dim);
    
      size_t numCellVertices = MT::numVertices(MT::topologicalDimension());
      size_t numEntityVertices = MT::numVertices(dim);
    
      IdVec entityVertices(MT::numEntities(dim) * 2);
    
      ConnVec entityVertexConn;
      ConnVec cellEntityConn(numCells());
    
      size_t maxCellEntityConns = 1;
      size_t entityId = 0;
    
      IdVecMap entityVerticesMap(numCells() * MT::numEntities(dim)/2);
    
      size_t n = numCells();
    
      Connectivity& cn =
        getConnectivity_(MT::topologicalDimension(), 0);
      assert(!cn.empty());
    
      for(size_t c = 0; c < n; ++c){
        cellEntityConn[c].reserve(maxCellEntityConns);
      
        Id* vertices = cn.getEntities(c);
      
        MT::createEdges(entityVertices, vertices);
      
        for(size_t i = 0; i < numCellVertices; ++i){
          Id* a = &entityVertices[i * numEntityVertices];
          IdVec ev(a, a + numEntityVertices);
          sort(ev.begin(), ev.end());
        
          auto itr = entityVerticesMap.emplace(std::move(ev), entityId);
          cellEntityConn[c].push_back(itr.first->second);
        
          if(itr.second){
            entityVertexConn.emplace_back(IdVec(a, a + numEntityVertices));
          
            maxCellEntityConns =
              std::max(maxCellEntityConns, cellEntityConn[c].size());
          
            ++entityId;
          }
        }
      }
    
      entityToVertex.init(entityVertexConn);
      cellToEntity.init(cellEntityConn);
      size_[dim] = entityToVertex.fromSize();
    }
  
    void transpose(size_t fromDim, size_t toDim){
      Connectivity& conn = getConnectivity_(fromDim, toDim);
    
      Connectivity& fromConn = getConnectivity_(toDim, fromDim);
      assert(!fromConn.empty());
    
      IndexVec pos(numEntities(fromDim), 0);
    
      for(Entity e1(*this, toDim); !e1.end(); ++e1){
        for(EntityIterator e0(e1, fromDim); !e0.end(); ++e0){
          pos[e0.index()]++;
        }
      }
    
      conn.resize(pos);
    
      std::fill(pos.begin(), pos.end(), 0);
    
      for(Entity e1(*this, toDim); !e1.end(); ++e1){
        for(EntityIterator e0(e1, fromDim); !e0.end(); ++e0){
          conn.set(e0.index(), e1.index(), pos[e0.index()]++);
        }
      }
    }
  
    void intersect(size_t fromDim, size_t toDim, size_t dim){
      Connectivity& conn = getConnectivity_(fromDim, toDim);
      if(!conn.empty()){
        return;
      }
    
      ConnVec conns(numEntities(fromDim));
    
      using VisitedVec = std::vector<bool>;
      VisitedVec visited(numEntities(fromDim));
    
      size_t maxSize = 1;
      IdVec fromVerts(MT::numVertices(fromDim));
      IdVec toVerts(MT::numVertices(toDim));
    
      for(Entity e0(*this, fromDim); !e0.end(); ++e0){
        IdVec& entities = conns[e0.index()];
        entities.reserve(maxSize);
      
        std::copy(e0.getEntities(0), e0.getEntities(0) + 
                  MT::numVertices(fromDim),
                  fromVerts.begin());
      
        std::sort(fromVerts.begin(), fromVerts.end());
      
        for(EntityIterator e1(e0, dim); !e1.end(); ++e1){
          for(EntityIterator e2(e1, toDim); !e2.end(); ++e2){
            visited[e2.index()] = false;
          }
        }
      
        for(EntityIterator e1(e0, dim); !e1.end(); ++e1){
          for(EntityIterator e2(e1, toDim); !e2.end(); ++e2){
            if(visited[e2.index()]){
              continue;
            }
          
            visited[e2.index()] = true;
          
            if (fromDim == toDim){
              if (e0.index() != e2.index()){
                entities.push_back(e2.index());
              }
            }
            else{
              copy(e2.getEntities(0),
                   e2.getEntities(0) + MT::numVertices(toDim),
                   toVerts.begin());
            
              std::sort(toVerts.begin(), toVerts.end());
            
              if(std::includes(fromVerts.begin(), fromVerts.end(),
                               toVerts.begin(), toVerts.end())){
              
                entities.push_back(e2.index());
              }
            }
          }
        }
      
        maxSize = std::max(entities.size(), maxSize);
      }
    
      conn.init(conns);
    }
  
    void compute(size_t fromDim, size_t toDim) override{
      Connectivity& conn = getConnectivity_(fromDim, toDim);
    
      if(!conn.empty()){
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
        ConnVec c(numEntities(fromDim), IdVec(1));
      
        for(Entity e(*this, fromDim); !e.end(); ++e){
          c[e.index()][0] = e.index();
        }
      
        getConnectivity_(fromDim, toDim).set(c);
      }
      else if(fromDim < toDim){
        compute(toDim, fromDim);
        transpose(fromDim, toDim);
      }
      else{
        size_t d = 0;
        compute(fromDim, d);
        compute(d, toDim);
        intersect(fromDim, toDim, d);
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
