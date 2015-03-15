/*
 * ###########################################################################
 * Copyright (c) 2015, Los Alamos National Security, LLC.
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

#include <cassert>
#include <vector>
#include <map>
#include <stdint.h>

using namespace std;

namespace{

  const size_t RESERVE = 1024;

  const int ELEMENT_INT32 = 0;
  const int ELEMENT_INT64 = 1;
  const int ELEMENT_FLOAT = 2;
  const int ELEMENT_DOUBLE = 3;

  class VarBase{};

  template<class T>
  class Var : public VarBase{
  public:
    Var()
      : i_(0){
      v_.reserve(RESERVE);
    }

    void capture(T value){
      v_.push_back(value);
     
      if(i_ == RESERVE){
        v_.reserve(v_.size() + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }
    }

  private:
    vector<T> v_;
    size_t i_;
  };

  class Frame{
  public:
    Frame(){}

    void addVar(uint32_t varId, int elementKind){
      while(vars_.size() <= varId){
        vars_.push_back(nullptr);
      }

      VarBase* v;

      switch(elementKind){
      case ELEMENT_INT32:
        v = new Var<int32_t>();
        break;
      case ELEMENT_INT64:
        v = new Var<int64_t>();
        break;
      case ELEMENT_FLOAT:
        v = new Var<float>();
        break;
      case ELEMENT_DOUBLE:
        v = new Var<double>();
        break;
      default:
        assert(false && "invalid element kind");
      }
      
      vars_[varId] = v;
    }

    template<class T>
    void capture(uint32_t varId, T value){
      assert(varId < vars_.size());
      
      static_cast<Var<T>*>(vars_[varId])->capture(value);
    }

  private:
    typedef vector<VarBase*> VarVec;

    VarVec vars_;
  };

} // end namespace

extern "C"{

  void* __scrt_create_frame(){
    return new Frame();
  }

  void __scrt_frame_add_var(void* f, uint32_t varId, uint32_t elementKind){
    static_cast<Frame*>(f)->addVar(varId, elementKind);
  }

  void __scrt_frame_capture_i32(void* f, uint32_t varId, int32_t value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_i64(void* f, uint32_t varId, int64_t value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_float(void* f, uint32_t varId, float value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_double(void* f, uint32_t varId, double value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void* __scrt_plot_init(void* frame, void* window){

  }

  void __scrt_plot_add_lines(void* plot, uint32_t xVarId, uint32_t yVarId){

  }

  void __scrt_plot_add_axis(void* plot, uint32_t dim, const char* label){

  }

  void __scrt_plot_render(void* plot){

  }

} // end extern "C"
