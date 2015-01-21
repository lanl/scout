/*
 *
 * ###########################################################################
 *
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
 *
 */

#include <iostream>
#include <vector>
#include <map>
#include <cassert>

using namespace std;

namespace{

  enum ElementKind{
    Cell,
    Vertex,
    Edge,
    Face
  };

  enum ScalarKind{
    Int32,
    Int64,
    Float,
    Double
  };

  size_t scalarSize(ScalarKind kind){
    switch(kind){
    case Float:
    case Int32:
      return 4;
    case Double:
    case Int64:
      return 8;
    default:
      assert(false && "invalid scalar kind");
    }
  }

  class MeshInfo{
  public:

    void save(const char* path){
      FILE* file = fopen(path, "wb");
      assert(file && "saveMesh: failed to open file for writing");

      uint32_t numFields = fields_.size();
      fwrite(&numFields, 1, 4, file);

      for(auto& f: fields_){
        uint32_t len = f.name.length();
        fwrite(&len, 1, 4, file);
        fwrite(f.name.c_str(), 1, len, file);
        
        uint64_t count = f.count;
        fwrite(&count, 1, 8, file);

        uint32_t elementKind = f.elementKind;
        fwrite(&elementKind, 1, 4, file);

        uint32_t scalarKind = f.scalarKind;
        fwrite(&scalarKind, 1, 4, file);
        
        fwrite(f.data, 1, f.count * scalarSize(f.scalarKind), file);
      }
      
      fclose(file);
    }

    void addField(const char* name, 
                  size_t count,
                  ElementKind elementKind,
                  ScalarKind scalarKind,
                  void* data){

      fields_.push_back({name, count, elementKind, scalarKind, data});
    }
    
  private:

    struct Field{
      string name;
      size_t count;
      ElementKind elementKind;
      ScalarKind scalarKind;
      void* data;
    };

    typedef vector<Field> FieldVec;

    FieldVec fields_;
  };

  typedef map<void*, MeshInfo*> MeshInfoMap;
  
  MeshInfoMap _meshInfoMap;

} // end namespace

extern "C" {
  
  void __scrt_save_mesh_start(void* meshPtr) {
    auto itr = _meshInfoMap.find(meshPtr);
    assert(itr == _meshInfoMap.end() && "mesh save already in progress");
    
    MeshInfo* info = new MeshInfo;

    _meshInfoMap.insert({meshPtr, info}); 
  }

  void __scrt_save_mesh_add_field(void* meshPtr,
                             const char* fieldName,
                             size_t count,
                             ElementKind elementKind,
                             ScalarKind scalarKind,
                             void* fieldData){

    auto itr = _meshInfoMap.find(meshPtr);
    assert(itr != _meshInfoMap.end() && "mesh save not in progress");
    
    MeshInfo* info = itr->second;
    info->addField(fieldName, count, elementKind, scalarKind, fieldData);
  }

  void __scrt_save_mesh_end(void* meshPtr, const char* path){
    auto itr = _meshInfoMap.find(meshPtr);
    assert(itr != _meshInfoMap.end() && "mesh save not in progress");

    MeshInfo* info = itr->second;
    info->save(path);

    delete info;
    _meshInfoMap.erase(itr);
  }  

} // extern "C"
