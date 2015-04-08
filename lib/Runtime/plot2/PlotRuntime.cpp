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

#include <cassert>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <sstream>

#include <QtGui>

#include "scout/Runtime/opengl/qt/QtWindow.h"
#include "scout/Runtime/opengl/qt/PlotWindow.h"
#include "scout/Runtime/opengl/qt/PlotWidget.h"
#include "scout/Runtime/opengl/qt/PlotRenderer.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

using namespace std;
using namespace scout;

typedef int32_t (*__sc_plot_func_i32)(void*, uint64_t);
typedef int64_t (*__sc_plot_func_i64)(void*, uint64_t);
typedef float (*__sc_plot_func_float)(void*, uint64_t);
typedef double (*__sc_plot_func_double)(void*, uint64_t);

typedef void (*__sc_plot_func_i32_vec)(void*, uint64_t, int32_t*);
typedef void (*__sc_plot_func_i64_vec)(void*, uint64_t, int64_t*);
typedef void (*__sc_plot_func_float_vec)(void*, uint64_t, float*);
typedef void (*__sc_plot_func_double_vec)(void*, uint64_t, double*);

namespace{

  const size_t RESERVE = 1024;

  const int ELEMENT_INT32 = 0;
  const int ELEMENT_INT64 = 1;
  const int ELEMENT_FLOAT = 2;
  const int ELEMENT_DOUBLE = 3;

  const double LEFT_MARGIN = 100.0;
  const double RIGHT_MARGIN = 50.0;
  const double TOP_MARGIN = 50.0;
  const double BOTTOM_MARGIN = 50.0;

  const double MIN = numeric_limits<double>::min();
  const double MAX = numeric_limits<double>::max();

  const size_t X_LABELS = 10;
  const size_t Y_LABELS = 10;
  const size_t X_TICKS = 20;
  const size_t Y_TICKS = 20;

  typedef uint32_t VarId;

  const uint32_t PLOT_VAR_BEGIN = 65536;
  
  const uint32_t FLAG_VAR_CONSTANT = 0x00000001; 

  typedef vector<double> DoubleVec;

 template<typename T, size_t N>
  class Vec{
  public:
    Vec(){}

    Vec(T* v){
      for(size_t i = 0; i < N; ++i){
        vc_[i] = v[i];
      }
    }

    Vec& operator=(T* v){
      for(size_t i = 0; i < N; ++i){
        vc_[i] = v[i];
      }
      
      return *this;
    }

    Vec& operator=(const Vec& v){
      for(size_t i = 0; i < N; ++i){
        vc_[i] = v[i];
      }
      
      return *this;
    }

    static size_t size(){
      return N;
    }

    T& operator[](size_t i){
      return vc_[i];
    }

    const T& operator[](size_t i) const{
      return vc_[i];
    }
    
   T* raw(){
     return vc_;
   }

   void get(DoubleVec& v) const{
     v.reserve(N);
     for(size_t i = 0; i < N; ++i){
       v.push_back(vc_[i]);
     }
   }

   void dump(){
     cerr << "[";
     for(size_t i = 0; i < N; ++i){
       if(i > 0){
         cerr << ",";
       }

       cerr << vc_[i];
     }
     cerr << "]";
   }

  private:
    T vc_[N] __attribute__ ((aligned (16)));
  };

  class VarBase{
  public:    
    virtual double min() = 0;

    virtual double max() = 0;

    virtual double get(size_t i) const = 0;

    virtual void getVec(size_t i, DoubleVec& v) const = 0;

    virtual void compute(void* frame, uint64_t index) = 0;

    virtual size_t size() const = 0;
  };

  template<typename T>
  size_t maxSize(T* v){
    return v->size();
  }

  template<typename T, typename... TS>
  size_t maxSize(T* v, TS... vs){
    return max(v->size(), maxSize(vs...));
  }

  template<class T>
  class Var : public VarBase{
  public:
    Var()
      : fp_(0),
        i_(RESERVE),
        min_(numeric_limits<T>::max()),
        max_(numeric_limits<T>::min()){
    }

    Var(T (*fp)(void*, uint64_t))
      : fp_(fp),
        i_(RESERVE),
        min_(numeric_limits<T>::max()),
        max_(numeric_limits<T>::min()){
    }

    void capture(T value){
      if(i_ == RESERVE){
        v_.reserve(v_.size() + RESERVE);
        i_ = 0;

        if(value < min_){
          min_ = value;
        }
        
        if(value > max_){
          max_ = value;
        }
      }
      else{
        ++i_;

        if(value < min_){
          min_ = value;
        }
        else if(value > max_){
          max_ = value;
        }
      }

      v_.push_back(value);
    }

    void compute(void* frame, uint64_t index){
      capture((*fp_)(frame, index));
    }

    double get(size_t i) const{
      return v_[i];
    }

    void getVec(size_t i, DoubleVec& v) const{
      assert(false && "not a vector");
    }

    T get_(size_t i) const{
      return v_[i];
    }

    double min(){
      return min_;
    }

    double max(){
      return max_;
    }

    size_t size() const{
      return v_.size();
    }

  private:
    vector<T> v_;
    size_t i_;
    T min_;
    T max_;
    T (*fp_)(void*, uint64_t);
  };

  template<class T>
  class ArrayVar : public VarBase{
  public:
    ArrayVar(T* array, size_t size)
      : v_(array),
        size_(size),
        ready_(false){}

    void compute(void* frame, uint64_t index){}

    double get(size_t i) const{
      return v_[i];
    }

    void getVec(size_t i, DoubleVec& v) const{
      assert(false && "not a vector");
    }

    T get_(size_t i) const{
      return v_[i];
    }

    double min(){
      if(!ready_){
        init();
      }

      return min_;
    }

    double max(){
      if(!ready_){
        init();
      }

      return max_;
    }

    size_t size() const{
      return size_;
    }

    void init(){
      min_ = v_[0];
      max_ = v_[1];

      T vi;
      for(size_t i = 1; i < size_; ++i){
        vi = v_[i];
        
        if(vi < min_){
          min_ = vi;
        }
        else if(vi > max_){
          max_ = vi;
        }
      }

      ready_ = true;
    }

  private:
    T* v_;
    size_t size_;
    T min_;
    T max_;
    bool ready_;
  };

  template<class T, size_t N>
  class VecVar : public VarBase{
  public:
    VecVar()
      : fp_(0),
        i_(RESERVE){
    }

    VecVar(void (*fp)(void*, uint64_t, T*))
      : fp_(fp),
        i_(RESERVE){
    }

    void capture(const Vec<T, N>& value){
      if(i_ == RESERVE){
        v_.reserve(v_.size() + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }

      v_.push_back(value);
    }

    void compute(void* frame, uint64_t index){
      Vec<T, N> v;
      (*fp_)(frame, index, v.raw());

      capture(v);
    }

    double get(size_t i) const{
      assert(false && "attempt to get scalar from vector");
    }

    void getVec(size_t i, DoubleVec& v) const{
      v_[i].get(v);
    }

    double min(){
      assert(false && "attempt to get min from vector");
    }

    double max(){
      assert(false && "attempt to get max from vector");
    }

    size_t size() const{
      return v_.size();
    }

  private:
    vector<Vec<T, N>> v_;
    size_t i_;
    void (*fp_)(void*, uint64_t, T*);
  };

  template<class T>
  class ConstVar : public VarBase{
  public:
    ConstVar(T value)
      : value_(value){}

    void compute(void* frame, uint64_t index){}

    double get(size_t i) const{
      return value_;
    }

    void getVec(size_t i, DoubleVec& v) const{
      assert(false && "not a vector");
    }

    double min(){
      return value_;
    }

    double max(){
      return value_;
    }

    size_t size() const{
      return 0;
    }

  private:
    T value_;
  };

  template<class T, size_t N>
  class ConstVecVar : public VarBase{
  public:
    ConstVecVar(const Vec<T, N>& v)
      : v_(v){}

    void compute(void* frame, uint64_t index){}

    double get(size_t i) const{
      assert(false && "attempt to get scalar from vector");
    }

    void getVec(size_t i, DoubleVec& v) const{
      v_.get(v);
    }

    double min(){
      assert(false && "attempt to get min from vector");
    }

    double max(){
      assert(false && "attempt to get max from vector");
    }

    size_t size() const{
      return 0;
    }

  private:
    Vec<T, N> v_;
  };

  class Frame{
  public:
    Frame()
      : ready_(false){}

    ~Frame(){
      for(auto& itr : plotFrameMap_){
        delete itr.second;
      }
    }

    void addVar(VarId varId, VarBase* v){
      while(vars_.size() <= varId){
        vars_.push_back(nullptr);
      }

      vars_[varId] = v;
    }

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

    void addArrayVar(VarId varId, int elementKind, void* array, size_t size){
      while(vars_.size() <= varId){
        vars_.push_back(nullptr);
      }

      VarBase* v;

      switch(elementKind){
      case ELEMENT_INT32:
        v = new ArrayVar<int32_t>((int32_t*)array, size);
        break;
      case ELEMENT_INT64:
        v = new ArrayVar<int64_t>((int64_t*)array, size);
        break;
      case ELEMENT_FLOAT:
        v = new ArrayVar<float>((float*)array, size);
        break;
      case ELEMENT_DOUBLE:
        v = new ArrayVar<double>((double*)array, size);
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

    void compute(Frame* parentFrame){
      ready_ = true;

      size_t end = parentFrame->size();
      size_t n = vars_.size();
      
      for(size_t i = size(); i < end; ++i){
        for(size_t j = 0; j < n; ++j){
          vars_[j]->compute(parentFrame, i);
        }
      }
    }

    size_t size() const{
      size_t n = vars_.size();
      size_t m = 0;
      
      for(size_t i = 1; i < n; ++i){
        size_t mi = vars_[i]->size();
        if(mi > m){
          m = mi;
        }
      }

      return m;
    }

    VarBase* getVar(VarId varId){
      assert(varId < vars_.size());
      return vars_[varId];
    }

    template<class T>
    T get(VarId varId, size_t index){
      return static_cast<Var<T>*>(getVar(varId))->get_(index);
    }

    template<class T>
    void addPlotVar(uint32_t plotId,
                    VarId varId,
                    T (*fp)(void*, uint64_t),
                    uint32_t flags){
      Frame* frame;

      auto itr = plotFrameMap_.find(plotId);
      if(itr == plotFrameMap_.end()){
        frame = new Frame;
        plotFrameMap_[plotId] = frame;
      }
      else{
        frame = itr->second;
        
        if(frame->ready_){
          return;
        }
      }

      VarBase* v;
      if(flags & FLAG_VAR_CONSTANT){
        v = new ConstVar<T>((*fp)(0, 0));
      }
      else{
        v = new Var<T>(fp);
      }

      frame->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    template<class T>
    void addPlotVecVar(uint32_t plotId,
                       VarId varId,
                       void (*fp)(void*, uint64_t, T*),
                       uint32_t dim,
                       uint32_t flags){
      Frame* frame;

      auto itr = plotFrameMap_.find(plotId);
      if(itr == plotFrameMap_.end()){
        frame = new Frame;
        plotFrameMap_[plotId] = frame;
      }
      else{
        frame = itr->second;
        
        if(frame->ready_){
          return;
        }
      }

      VarBase* v;
      
      if(flags & FLAG_VAR_CONSTANT){
        switch(dim){
        case 2:{
          Vec<T, 2> x;
          (*fp)(0, 0, x.raw());
          v = new ConstVecVar<T, 2>(x);
          break;
        }
        case 3:{
          Vec<T, 3> x;
          (*fp)(0, 0, x.raw());
          v = new ConstVecVar<T, 3>(x);
          break;
        }
        case 4:{
          Vec<T, 4> x;
          (*fp)(0, 0, x.raw());
          v = new ConstVecVar<T, 4>(x);
          break;
        }
        default:
          assert(false && "invalid vector size");
        }
      }
      else{
        switch(dim){
        case 2:
          v = new VecVar<T, 2>(fp);
          break;
        case 3:
          v = new VecVar<T, 3>(fp);
          break;
        case 4:
          v = new VecVar<T, 4>(fp);
          break;
        default:
          assert(false && "invalid vector size");
        }
      }

      frame->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    Frame* initPlotFrame(uint32_t plotId){
      auto itr = plotFrameMap_.find(plotId);
      if(itr == plotFrameMap_.end()){
        return 0;
      }

      itr->second->compute(this);
      return itr->second;
    }

  private:
    typedef vector<VarBase*> VarVec;
    typedef map<uint32_t, Frame*> PlotFrameMap;

    VarVec vars_;
    PlotFrameMap plotFrameMap_;
    bool ready_;
  };

  void drawText(QPainter& painter,
                const QString& text,
                const QPointF& point,
                bool right=false,
                bool bottom=false){

    QRect bounds = painter.fontMetrics().boundingRect(text);

    QRectF frame(point.x(), point.y(), bounds.width() + 5, bounds.height());

    QTextOption textOption;

    Qt::Alignment alignment = 0;

    if(right){
      frame.translate(-frame.width(), 0.0);
      alignment |= Qt::AlignRight;
    }

    if(bottom){
      frame.translate(0.0, -frame.height());
      alignment |= Qt::AlignBottom;
    }

    textOption.setAlignment(alignment);

    painter.drawText(frame, text, textOption);
  }

  QColor toQColor(DoubleVec& v){
    assert(v.size() == 4);
    
    return QColor(v[0]*255, v[1]*255, v[2]*255, v[3]*255);
  }

  QString toLabel(double value){
    stringstream sstr;
    sstr.precision(4);
    sstr << value;

    return sstr.str().c_str();
  }

  class Plot : public PlotRenderer{
  public:
    class Element{
    public:
      virtual int order() = 0;
    };

    class RangeElement : public Element{
    public:
      virtual VarId getX() = 0;

      virtual VarId getY() = 0;
    };

    class Lines : public RangeElement{
    public:
      Lines(VarId x, VarId y, VarId size, VarId color)
        : x(x), y(y), size(size), color(color){}

      VarId x;
      VarId y;
      VarId size;
      VarId color;

      int order(){
        return 1;
      }

      VarId getX(){
        return x;
      }

      VarId getY(){
        return y;
      }
    };

    class Points : public RangeElement{
    public:
      Points(VarId x, VarId y, VarId size, VarId color)
        : x(x), y(y), size(size), color(color){}

      VarId x;
      VarId y;
      VarId size;
      VarId color;

      int order(){
        return 2;
      }

      VarId getX(){
        return x;
      }

      VarId getY(){
        return y;
      }
    };

    class Area : public RangeElement{
    public:
      Area(VarId x, VarId y, VarId color)
        : x(x), y(y), color(color){}

      VarId x;
      VarId y;
      VarId color;

      int order(){
        return 0;
      }

      VarId getX(){
        return x;
      }

      VarId getY(){
        return y;
      }
    };

    class Interval : public RangeElement{
    public:
      Interval(VarId x, VarId y, VarId color)
        : x(x), y(y), color(color){}

      VarId x;
      VarId y;
      VarId color;

      int order(){
        return 0;
      }

      VarId getX(){
        return x;
      }

      VarId getY(){
        return y;
      }
    };

    class Bins : public Element{
    public:
      Bins(VarId varIn, VarId xOut, VarId yOut, uint32_t n)
        : varIn(varIn), xOut(xOut), yOut(yOut), n(n){}

      VarId varIn;
      VarId xOut;
      VarId yOut;
      uint32_t n;

      int order(){
        return 0;
      }
    };

    class Axis : public Element{
    public:
      Axis(uint32_t dim, const string& label)
        : dim(dim), label(label){}

      uint32_t dim;
      string label;

      int order(){
        return 3;
      }
    };

    Plot(uint32_t plotId, Frame* frame, PlotWindow* window)
      : plotId_(plotId),
        frame_(frame),
        plotFrame_(0),
        window_(window){}

    ~Plot(){}

    VarBase* getVar(VarId varId){
      return varId >= PLOT_VAR_BEGIN ? 
        plotFrame_->getVar(varId - PLOT_VAR_BEGIN) : frame_->getVar(varId);
    }

    template<class T>
    void addVar(VarId varId,
                T (*fp)(void*, uint64_t),
                uint32_t flags){
      frame_->addPlotVar(plotId_, varId, fp, flags);
    }

    template<class T>
    void addVecVar(VarId varId,
                   void (*fp)(void*, uint64_t, T*),
                   uint32_t dim,
                   uint32_t flags){
      frame_->addPlotVecVar(plotId_, varId, fp, dim, flags);
    }

    void addLines(VarId x, VarId y, VarId size, VarId color){
      elements_.push_back(new Lines(x, y, size, color)); 
    }

    void addPoints(VarId x, VarId y, VarId size, VarId color){
      elements_.push_back(new Points(x, y, size, color)); 
    }

    void addArea(VarId x, VarId y, VarId color){
      elements_.push_back(new Area(x, y, color)); 
    }

    void addInterval(VarId x, VarId y, VarId color){
      elements_.push_back(new Interval(x, y, color)); 
    }

    void addBins(VarId varIn, VarId xOut, VarId yOut, uint32_t n){
      elements_.push_back(new Bins(varIn, xOut, yOut, n)); 
    }

    void addAxis(uint32_t dim, const string& label){
      elements_.push_back(new Axis(dim, label)); 
    }

    void finalize(){
      QtWindow::init();

      plotFrame_ = frame_->initPlotFrame(plotId_);
     
      widget_ = window_->getWidget();
      widget_->setRenderer(this);
      window_->show();
      window_->update();

      QtWindow::pollEvents();
    }
    
    void render(){
      size_t frameSize = frame_->size();

      if(frameSize == 0){
        return;
      }

      QPainter painter(widget_);
      painter.setRenderHint(QPainter::Antialiasing, true);

      sort(elements_.begin(), elements_.end(),
           [](Element* a, Element* b){
             return a->order() < b->order();
           });

      double xMin = MAX;
      double xMax = MIN;
      double yMin = MAX;
      double yMax = MIN;

      for(Element* e : elements_){
        if(RangeElement* r = dynamic_cast<RangeElement*>(e)){
          VarBase* x = getVar(r->getX());
          VarBase* y = getVar(r->getY());

          if(x->min() < xMin){
            xMin = x->min();
          }

          if(x->max() > xMax){
            xMax = x->max();
          }

          if(y->min() < yMin){
            yMin = y->min();
          }

          if(y->max() > yMax){
            yMax = y->max();
          }
        }
      }

      double xSpan = xMax - xMin;
      double ySpan = yMax - yMin;

      QSize frame = widget_->frameSize();
      
      double width = frame.width();
      double height = frame.height();

      QPointF origin(LEFT_MARGIN, height - BOTTOM_MARGIN);
  
      double xLen = width - LEFT_MARGIN - RIGHT_MARGIN;
      double yLen = height - TOP_MARGIN - BOTTOM_MARGIN;

      QPointF xEnd = origin;
      xEnd += QPointF(xLen, 0.0);

      QPointF yEnd = origin;
      yEnd -= QPointF(0.0, yLen);
      
      for(Element* e : elements_){
        if(Axis* a = dynamic_cast<Axis*>(e)){
          if(a->dim == 1){
            painter.drawLine(origin, xEnd);

            size_t inc = frameSize / X_LABELS;

            if(inc == 0){
              inc = 1;
            }

            double xc;
            for(size_t i = 0; i < frameSize; i += inc){
              xc = origin.x() + double(i)/frameSize * xLen;

              drawText(painter,
                       toLabel(xMin + (double(i)/frameSize)*xSpan),
                       QPointF(xc, height - BOTTOM_MARGIN + 3));
            }
            
            inc = frameSize / X_TICKS;

            if(inc == 0){
              inc = 1;
            }

            for(size_t i = 0; i < frameSize; i += inc){
              xc = origin.x() + double(i)/frameSize * xLen;

              painter.drawLine(QPointF(xc, height - BOTTOM_MARGIN + 3),
                               QPointF(xc, height - BOTTOM_MARGIN - 3));
            }
          }
          else if(a->dim == 2){
            painter.drawLine(origin, yEnd);

            size_t inc = frameSize / Y_LABELS;

            if(inc == 0){
              inc = 1;
            }

            double yc;
            double yv;
            for(size_t i = 0; i <= frameSize; i += inc){
              yv = yMin + ySpan * double(i)/frameSize;
              yc = origin.y() - double(i)/frameSize * yLen;

              drawText(painter,
                       toLabel(yv),
                       QPointF(LEFT_MARGIN - 5, yc), true);
            }

            inc = frameSize / Y_TICKS;
            
            if(inc == 0){
              inc = 1;
            }

            for(size_t i = 0; i < frameSize; i += inc){
              yc = origin.y() - double(i)/frameSize * yLen;

              painter.drawLine(QPointF(LEFT_MARGIN - 3, yc),
                               QPointF(LEFT_MARGIN + 3, yc));
            }
          }
          else{
            assert(false && "invalid axis dim");
          }
        }
      }

      for(Element* e : elements_){
        if(Lines* l = dynamic_cast<Lines*>(e)){
          VarBase* x = getVar(l->x);
          VarBase* y = getVar(l->y);
          VarBase* s = getVar(l->size);
          VarBase* c = getVar(l->color);

          size_t size = maxSize(x, y, s, c);

          QPointF lastPoint;
          QPointF point;
          QPen pen;

          for(size_t i = 0; i < size; ++i){
            point.setX(origin.x() + ((x->get(i) - xMin)/xSpan) * xLen);
            point.setY(origin.y() - ((y->get(i) - yMin)/ySpan) * yLen);

            pen.setWidthF(s->get(i));

            if(i > 0){
              DoubleVec cv;
              c->getVec(i, cv);
              pen.setColor(toQColor(cv));
              painter.setPen(pen);
              painter.drawLine(point, lastPoint);
            }

            lastPoint = point;
          }
        }
        else if(Area* a = dynamic_cast<Area*>(e)){
          VarBase* x = getVar(a->x);
          VarBase* y = getVar(a->y);
          VarBase* c = getVar(a->color);

          size_t size = maxSize(x, y, c);

          QPen noPen(Qt::NoPen);
          painter.setPen(noPen);

          QPointF lastPoint;
          QPointF point;

          for(size_t i = 0; i < size; ++i){
            point.setX(origin.x() + ((x->get(i) - xMin)/xSpan) * xLen);
            point.setY(origin.y() - ((y->get(i) - yMin)/ySpan) * yLen);

            if(i > 0){
              DoubleVec cv;
              c->getVec(i, cv);
              QBrush brush(toQColor(cv));
              painter.setBrush(brush);

              QPolygonF poly;
              poly << lastPoint << point << 
                QPointF(point.x(), origin.y()) << 
                QPointF(lastPoint.x(), origin.y());
      
              painter.drawPolygon(poly);
            }

            lastPoint = point;
          }
        }
        else if(Points* p = dynamic_cast<Points*>(e)){
          VarBase* x = getVar(p->x);
          VarBase* y = getVar(p->y);
          VarBase* s = getVar(p->size);
          VarBase* c = getVar(p->color);

          size_t size = maxSize(x, y, s, c);

          QPointF point;

          QPen noPen(Qt::NoPen);
          painter.setPen(noPen);

          for(size_t i = 0; i < size; ++i){
            point.setX(origin.x() + ((x->get(i) - xMin)/xSpan) * xLen);
            point.setY(origin.y() - ((y->get(i) - yMin)/ySpan) * yLen);

            DoubleVec cv;
            c->getVec(i, cv);

            QBrush brush(toQColor(cv));
            painter.setBrush(brush);
            painter.drawEllipse(point, s->get(i), s->get(i));
          }
        }
      }
    }

  private:
    typedef vector<Element*> ElementVec_;

    uint32_t plotId_;
    Frame* frame_;
    Frame* plotFrame_;
    PlotWindow* window_;
    PlotWidget* widget_;

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

  void __scrt_frame_add_array_var(void* f,
                                  VarId varId,
                                  VarId elementKind,
                                  void* array,
                                  size_t size){
    static_cast<Frame*>(f)->addArrayVar(varId, elementKind, array, size);
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

  int32_t __scrt_frame_get_i32(void* f, VarId varId, uint64_t index){
    return static_cast<Frame*>(f)->get<int32_t>(varId, index);
  }

  int64_t __scrt_frame_get_i64(void* f, VarId varId, uint64_t index){
    return static_cast<Frame*>(f)->get<int64_t>(varId, index);
  }

  float __scrt_frame_get_float(void* f, VarId varId, uint64_t index){
    return static_cast<Frame*>(f)->get<float>(varId, index);
  }

  double __scrt_frame_get_double(void* f, VarId varId, uint64_t index){
    return static_cast<Frame*>(f)->get<double>(varId, index);
  }

  void* __scrt_plot_init(uint32_t plotId, void* frame, void* window){
    return new Plot(plotId,
                    static_cast<Frame*>(frame),
                    static_cast<PlotWindow*>(window));
  }

  void __scrt_plot_add_var_i32(void* plot,
                               VarId varId,
                               __sc_plot_func_i32 fp,
                               uint32_t flags){
    static_cast<Plot*>(plot)->addVar(varId, fp, flags);
  }

  void __scrt_plot_add_var_i64(void* plot,
                               VarId varId,
                               __sc_plot_func_i64 fp,
                               uint32_t flags){
    static_cast<Plot*>(plot)->addVar(varId, fp, flags);
  }

  void __scrt_plot_add_var_float(void* plot,
                                 VarId varId,
                                 __sc_plot_func_float fp,
                                 uint32_t flags){
    static_cast<Plot*>(plot)->addVar(varId, fp, flags);
  }

  void __scrt_plot_add_var_double(void* plot,
                                  VarId varId,
                                  __sc_plot_func_double fp,
                                  uint32_t flags){
    static_cast<Plot*>(plot)->addVar(varId, fp, flags);
  }

  void __scrt_plot_add_var_i32_vec(void* plot,
                                   VarId varId,
                                   __sc_plot_func_i32_vec fp,
                                   uint32_t dim,
                                   uint32_t flags){
    static_cast<Plot*>(plot)->addVecVar(varId, fp, dim, flags);
  }

  void __scrt_plot_add_var_i64_vec(void* plot,
                                   VarId varId,
                                   __sc_plot_func_i64_vec fp,
                                   uint32_t dim,
                                   uint32_t flags){
    static_cast<Plot*>(plot)->addVecVar(varId, fp, dim, flags);
  }

  void __scrt_plot_add_var_float_vec(void* plot,
                                 VarId varId,
                                 __sc_plot_func_float_vec fp,
                                 uint32_t dim,
                                 uint32_t flags){
    static_cast<Plot*>(plot)->addVecVar(varId, fp, dim, flags);
  }

  void __scrt_plot_add_var_double_vec(void* plot,
                                      VarId varId,
                                      __sc_plot_func_double_vec fp,
                                      uint32_t dim,
                                      uint32_t flags){
    static_cast<Plot*>(plot)->addVecVar(varId, fp, dim, flags);
  }

  void __scrt_plot_add_lines(void* plot,
                             VarId x,
                             VarId y,
                             VarId size,
                             VarId color){
    static_cast<Plot*>(plot)->addLines(x, y, size, color);
  }

  void __scrt_plot_add_points(void* plot,
                              VarId x,
                              VarId y,
                              VarId size,
                              VarId color){
    static_cast<Plot*>(plot)->addPoints(x, y, size, color);
  }

  void __scrt_plot_add_area(void* plot,
                              VarId x,
                              VarId y,
                              VarId color){
    static_cast<Plot*>(plot)->addArea(x, y, color);
  }

  void __scrt_plot_add_interval(void* plot,
                                VarId x,
                                VarId y,
                                VarId color){
    static_cast<Plot*>(plot)->addInterval(x, y, color);
  }

  void __scrt_plot_add_bins(void* plot,
                            VarId varIn,
                            VarId xOut,
                            VarId yOut,
                            uint32_t n){
    static_cast<Plot*>(plot)->addBins(varIn, xOut, yOut, n);
  }

  void __scrt_plot_add_axis(void* plot, uint32_t dim, const char* label){
    static_cast<Plot*>(plot)->addAxis(dim, label);
  }

  void __scrt_plot_render(void* plot){
    static_cast<Plot*>(plot)->finalize();
  }

} // end extern "C"
