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
#include <string>
#include <cassert>

using namespace std;

namespace{

  class Mesh{
  public:
    struct Field{
      sclegion_element_kind_t elementKind;
      sclegion_field_kind_t fieldKind;
      legion_field_id_t fieldId;
    };
    
    struct Element{
      Element()
        : size(0){}

      size_t size;
      legion_logical_region_t logicalRegion;
      legion_field_space_t fieldSpace;
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
      field.elementKind = elementKind;
      field.fieldKind = fieldKind;
      field.fieldId = nextFieldId_++;

      fieldMap_.insert({fieldName, field});
      Element& element = elements_[elementKind];
      
      if(element.size == 0){
        element.size = numItems(elementKind);
      }
    }

    legion_index_space_t create1dIndexSpace(size_t size){
      legion_rect_1d_t rect;
      rect.lo = {0};
      rect.hi = {int(size) - 1};

      legion_domain_t domain = legion_domain_from_rect_1d(rect);

      return legion_index_space_create_domain(runtime_, context_, domain);
    }
              
    void init(){
      for(size_t i = 0; i < SCLEGION_ELEMENT_MAX; ++i){
        Element& element = elements_[i];

        size_t size = element.size;

        if(size == 0){
          continue;
        }

        element.indexSpace = create1dIndexSpace(size);
        element.fieldSpace = legion_field_space_create(runtime_, context_);

        element.fieldAllocator =
          legion_field_allocator_create(runtime_, context_, element.fieldSpace);

        for(auto& itr : fieldMap_){
          Field& field = itr.second;

          legion_field_allocator_allocate_field(element.fieldAllocator,
                                                fieldKindSize(field.fieldKind),
                                                field.fieldId);
        }
      }
    }

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
        assert(false && "default field kind");
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
      }
    }

    size_t numVertices(){
      switch(rank_){
      case 1:
        return width_ + 1;
        break;
      case 2:
        return (width_ + 1) * (height_ + 1);
        break;
      case 3:
        return (width_ + 1) * (height_ + 1) + (depth_ + 1);
        break;
      }
    }

    size_t numEdges(){
      switch(rank_){
      case 1:
        return width_;
        break;
      case 2:
        return (width_ + 1)*height_ + (height_ + 1)*width_;
        break;
      case 3:
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        return (w1*height_ + h1*width_)*(depth_ + 1) + w1*h1*depth_;
        break;
      }
    }

    size_t numFaces(){
      switch(rank_){
      case 1:
        return width_;
        break;
      case 2:
        return (width_ + 1)*height_ + (height_ + 1)*width_;
        break;
      case 3:
        size_t w1 = width_ + 1;
        size_t h1 = height_ + 1;
        size_t d1 = depth_ + 1;
        return w1*height_*depth_ + h1*width_*depth_ + d1*width_*height_;
        break;
      }
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

} // end namespace

void
sclegion_uniform_mesh_create(legion_runtime_t runtime,
                             legion_context_t context,
                             sclegion_uniform_mesh_t mesh,
                             size_t rank,
                             size_t width,
                             size_t height,
                             size_t depth){
  mesh.impl = new Mesh(runtime, context, rank, width, height, depth);
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
