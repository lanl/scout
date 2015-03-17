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
#include <cstdint>
#include <iostream>

#include "scout/Runtime/opengl/qt/QtWindow.h"
#include "scout/Runtime/opengl/qt/PlotWidget.h"
#include "scout/Runtime/opengl/qt/PlotRenderer.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace std;
using namespace scout;

namespace{

  const size_t RESERVE = 1024;

  const int ELEMENT_INT32 = 0;
  const int ELEMENT_INT64 = 1;
  const int ELEMENT_FLOAT = 2;
  const int ELEMENT_DOUBLE = 3;

  typedef uint32_t VarId;

  class VarBase{
  public:
    virtual size_t size() const = 0;
  };

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

    size_t size() const{
      return v_.size();
    }

  private:
    vector<T> v_;
    size_t i_;
  };

  class Frame{
  public:
    Frame(){}

    void addVar(VarId varId, int elementKind){
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
    void capture(VarId varId, T value){
      assert(varId < vars_.size());
      
      static_cast<Var<T>*>(vars_[varId])->capture(value);
    }

    size_t size() const{
      return vars_[0]->size();
    }

  private:
    typedef vector<VarBase*> VarVec;

    VarVec vars_;
  };

  class Plot : public PlotRenderer{
  public:
    class Element{};

    class Lines : public Element{
    public:
      Lines(VarId xVarId, VarId yVarId)
        : xVarId(xVarId), yVarId(yVarId){}

      VarId xVarId;
      VarId yVarId;
    };

    class Axis : public Element{
    public:
      Axis(uint32_t dim, const string& label)
        : dim(dim), label(label){}

      uint32_t dim;
      string label;
    };

    Plot(Frame* frame, QtWindow* window)
      : frame_(frame),
        window_(window){}

    void addLines(VarId xVarId, VarId yVarId){
      elements_.push_back(new Lines(xVarId, yVarId)); 
    }

    void addAxis(uint32_t dim, const string& label){
      elements_.push_back(new Axis(dim, label)); 
    }

    void finalize(){
      QtWindow::init();

      //PlotWidget* widget = new PlotWidget(window_);
      //widget->setRenderer(this);

      //window_->show();
    }

    void render(){
      nlog("rendering!!!!");
    }

  private:
    typedef vector<Element*> ElementVec_;

    Frame* frame_;
    QtWindow* window_;
    
    ElementVec_ elements_;
  };

} // end namespace

extern "C"{

  void* __scrt_create_frame(){
    return new Frame();
  }

  void __scrt_frame_add_var(void* f, VarId varId, VarId elementKind){
    static_cast<Frame*>(f)->addVar(varId, elementKind);
  }

  void __scrt_frame_capture_i32(void* f, VarId varId, int32_t value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_i64(void* f, VarId varId, int64_t value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_float(void* f, VarId varId, float value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void __scrt_frame_capture_double(void* f, VarId varId, double value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void* __scrt_plot_init(void* frame, void* window){
    return new Plot(static_cast<Frame*>(frame),
                    static_cast<QtWindow*>(window));
  }

  void __scrt_plot_add_lines(void* plot, VarId xVarId, VarId yVarId){
    static_cast<Plot*>(plot)->addLines(xVarId, yVarId);
  }

  void __scrt_plot_add_axis(void* plot, uint32_t dim, const char* label){
    static_cast<Plot*>(plot)->addAxis(dim, label);
  }

  void __scrt_plot_render(void* plot){
    static_cast<Plot*>(plot)->finalize();
  }

} // end extern "C"
