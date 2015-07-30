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

#include "MeshTopology.h"

using namespace std;
using namespace scout;

namespace{

  class UniformMeshType{
  public:
    using Id = uint64_t;
    using Float = double;

    static constexpr size_t topologicalDimension(){
      return 2;
    }

    static constexpr size_t geometricDimension(){
      return 2;
    }

    static size_t numEntities(size_t dim){
      switch(dim){
      case 0:
        return 4;
      case 1:
        return 4;
      case 2:
        return 1;
      default:
        assert(false && "invalid dimension");
      }
    }

    static constexpr size_t verticesPerCell(){
      return 4;
    }
  
    static size_t numVertices(size_t dim){
      switch(dim){
      case 0:
        return 1;
      case 1:
        return 2;
      case 2:
        return 4;
      default:
        assert(false && "invalid dimension");
      }
    }
  
    static void create(std::vector<Id>& e, size_t dim, Id* v){
      assert(dim == 1);
      assert(e.size() == 8);
    
      e[0] = v[0];
      e[1] = v[2];

      e[2] = v[1];
      e[3] = v[3];
    
      e[4] = v[0];
      e[5] = v[1];
    
      e[6] = v[2];
      e[7] = v[3];
    }
  
  };

} // namespace

using UniformMesh = MeshTopology<UniformMeshType>;

extern "C"{

  void* __scrt_create_uniform_mesh1d(uint64_t width){
    assert(false && "unimplemented");
  }

  void* __scrt_create_uniform_mesh2d(uint64_t width, uint64_t height){
    auto mesh = new UniformMesh;

    size_t id = 0;
    for(size_t j = 0; j < height + 1; ++j){
      for(size_t i = 0; i < width + 1; ++i){
        mesh->addVertex(id++, {UniformMeshType::Float(i),
                               UniformMeshType::Float(j)}); 
      }
    }

    id = 0;
    for(size_t j = 0; j < height; ++j){
      for(size_t i = 0; i < width; ++i){
        mesh->addCell(id++,
                      {i + j*(width + 1),
                       i + (j + 1)*(width + 1),
                       i + 1 + j*(width + 1),
                       i + 1 + (j + 1)*(width + 1)}
                      );
      }
    }

    return mesh;
  }

  void* __scrt_create_uniform_mesh3d(uint64_t width,
                                     uint64_t height,
                                     uint64_t depth){
    assert(false && "unimplemented");
  }

  // return end index
  uint64_t __scrt_mesh_num_entities(void* topology, uint32_t dim){
    return static_cast<MeshTopologyBase*>(topology)->numEntities(dim);
  }

  // return end index
  uint64_t __scrt_mesh_get_entities(void* topology,
                                    uint32_t fromDim,
                                    uint32_t toDim,
                                    uint64_t index,
                                    uint64_t** entities){
    return static_cast<MeshTopologyBase*>(topology)->getEntities(fromDim,
                                                                 toDim,
                                                                 index,
                                                                 entities);
  }
  
} // extern "C"
