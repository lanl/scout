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
#include <unordered_map>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <functional>
#include <random>
#include <mutex>
#include <limits>

#include <QtGui>

#include "scout/Runtime/opengl/qt/ScoutWindow.h"
#include "scout/Runtime/opengl/qt/PlotWidget.h"
#include "scout/Runtime/opengl/qt/PlotRenderer.h"

#define ndump(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << X << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << X << std::endl

// hack around linux stupidity 
#undef major
#undef minor


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
  const int ELEMENT_STRING = 4;

  const double MARGIN = 50.0;
  const double AXIS_LABEL_SIZE = 24.0;
  const double TICK_LABEL_SIZE = 12.0;
  const double MIN_FONT_SIZE = 8.0;
  const double MAX_FONT_SIZE = 28.0;

  const double MIN = numeric_limits<double>::min();
  const double MAX = numeric_limits<double>::max();
  const double EPSILON = 0.000001;

  typedef uint32_t VarId;

  const uint32_t PLOT_VAR_BEGIN = 65536;
  
  const uint32_t FLAG_VAR_CONSTANT = 0x00000001; 
  const uint32_t FLAG_VAR_POSITION = 0x00000002; 

  const uint64_t AGG_SUM = 6716694111845535935ULL;
  const uint64_t AGG_MEAN = 8849440314945535285ULL;
  const uint64_t AGG_VARIANCE = 14523147045845051570ULL;

  VarId nullVarId = numeric_limits<VarId>::max();

  class Random{
  public:
    Random(uint64_t seed=0)
    : uniform_(0, 1){
      rng_.seed(seed);
    }
    
    ~Random(){}
    
    void setSeed(uint64_t seed){
      rng_.seed(seed);
    }
        
    double uniform(){
      return uniform_(rng_);
    }
    
    double uniform(double a, double b){
      return a + (b - a) * uniform();
    }
    
    int64_t equilikely(int64_t a, int64_t b){
      return a + int64_t((b - a + 1) * uniform());
    }
  
  private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;
  };

  class Plot;
  
  class Global{
  public:
    Global(){}

    Plot* getPlot(uint64_t plotId){
      auto itr = plotMap_.find(plotId);

      if(itr == plotMap_.end()){
        return nullptr;
      }

      return itr->second;
    }

    void putPlot(uint64_t plotId, Plot* plot){
      plotMap_[plotId] = plot;
    }

  private:
    using PlotMap_ = unordered_map<uint64_t, Plot*>;

    PlotMap_ plotMap_;
  };

  Global* _global = 0;

  mutex _mutex;

  template<typename T, size_t N>
  class Vec{
  public:
    Vec(){}

    Vec(T* v){
      for(size_t i = 0; i < N; ++i){
        vc_[i] = v[i];
      }
    }

    Vec(initializer_list<T> il){
      copy(il.begin(), il.end(), vc_);
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

    void scale(const Vec& v){
      for(size_t i = 0; i < N; ++i){
        vc_[i] *= v.vc_[i];
      }
    }

  private:
    T vc_[N] __attribute__ ((aligned (16)));
  };

  using double2 = Vec<double, 2>;

  template<typename T>
  size_t maxSize(T* v){
    return v->size();
  }

  template<typename T, typename... TS>
  size_t maxSize(T* v, TS... vs){
    return max(v->size(), maxSize(vs...));
  }

  class VarBase{
  public:    
    virtual double get(size_t i) const{
      assert(false && "not a scalar");
    }

    virtual size_t hash(size_t i) const = 0;

    virtual void* getVec(size_t i){
      assert(false && "not a vector");
    }

    virtual void compute(void* plot, uint64_t index){};

    virtual void compute(const QTransform& t, void* plot, uint64_t index){
      compute(plot, index);
    };

    virtual size_t size() const = 0;
    
    virtual bool isConst() const{
      return false;
    }
  };

  template<class T>
  class ScalarVar : public VarBase{
  public:
    virtual T at(size_t i) const = 0;

    double get(size_t i) const{
      return at(i);
    }

    size_t hash(size_t i) const{
      return std::hash<T>()(at(i));
    }

    virtual T sum(){
      T ret = 0;
      size_t n = size();

      for(size_t i = 0; i < n; ++i){
        ret += at(i);
      }

      return ret;
    }

    virtual pair<T, T> minMax(){
      T min = numeric_limits<T>::max();
      T max = numeric_limits<T>::min();
      
      size_t n = size();

      for(size_t i = 0; i < n; ++i){
        T x = at(i);
        if(x < min){
          min = x;
        }

        if(x > max){
          max = x;
        }
      }

      return {min, max};
    }
  };

  class IndexVar : public ScalarVar<uint64_t>{
  public:
    IndexVar()
      : max_(0){}

    uint64_t at(size_t i) const{
      return i;
    }
    
    size_t size() const{
      return 0;
    }

  private:
    uint64_t max_;
  };

  class XPosVar : public ScalarVar<uint32_t>{
  public:
    XPosVar(uint32_t width)
      : width_(width){}

    uint32_t at(size_t i) const{
      return i % width_;
    }
    
    size_t size() const{
      return 0;
    }

  private:
    uint32_t width_;
  };

  class YPosVar : public ScalarVar<uint32_t>{
  public:
    YPosVar(uint32_t width, uint32_t height)
      : width_(width){}

    uint32_t at(size_t i) const{
      return i / width_;
    }
    
    size_t size() const{
      return 0;
    }

  private:
    uint32_t width_;
  };

  class ZPosVar : public ScalarVar<uint32_t>{
  public:
    ZPosVar(uint32_t width, uint32_t height)
      : wh_(width * height){}

    uint32_t at(size_t i) const{
      return i / wh_;
    }
    
    size_t size() const{
      return 0;
    }

  private:
    uint32_t wh_;
    uint32_t d1_;
  };

  template<class T>
  class Var : public ScalarVar<T>{
  public:
    Var()
      : fp_(0),
        i_(RESERVE){
    }

    Var(T (*fp)(void*, uint64_t))
      : fp_(fp),
        i_(RESERVE){
    }

    void capture(T value){
      if(i_ == RESERVE){
        v_.reserve(v_.size() + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }

      v_.push_back(value);
    }

    void compute(void* plot, uint64_t index){
      if(fp_){
        capture((*fp_)(plot, index));
      }
    }

    T at(size_t i) const{
      return v_[i];
    }

    size_t hash(size_t i) const{
      return std::hash<T>()(v_[i]);
    }

    size_t size() const{
      return v_.size();
    }

    void clear(){
      v_.clear();
      i_ = RESERVE;
    }

  private:
    vector<T> v_;
    size_t i_;
    T (*fp_)(void*, uint64_t);
  };

  template<class T>
  class ArrayVar : public ScalarVar<T>{
  public:
    ArrayVar(T* array, size_t size)
      : v_(array),
        size_(size){}

    double get(size_t i) const{
      return v_[i];
    }

    T at(size_t i) const{
      return v_[i];
    }

    size_t size() const{
      return size_;
    }

  private:
    T* v_;
    size_t size_;
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

    virtual void capture(const Vec<T, N>& value){
      if(i_ == RESERVE){
        v_.reserve(v_.size() + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }

      v_.push_back(value);
    }

    void compute(void* plot, uint64_t index){
      if(fp_){
        Vec<T, N> v;
        (*fp_)(plot, index, v.raw());

        capture(v);
      }
    }

    size_t hash(size_t i) const{
      assert(false && "attempt to hash from vector");
    }

    void* getVec(size_t i){
      return v_[i].raw();
    }

    T* raw(){
      return reinterpret_cast<T*>(v_.data()); 
    }

    size_t size() const{
      return v_.size();
    }

    void clear(){
      v_.clear();
    }

  private:
    vector<Vec<T, N>> v_;
    size_t i_;
    void (*fp_)(void*, uint64_t, T*);
  };

  class PositionVar : public VarBase{
  public:
    PositionVar(void (*fp)(void*, uint64_t, double*))
      : fp_(fp),
        i_(RESERVE){}

    PositionVar(const QPointF& p)
      : fp_(nullptr),
        i_(RESERVE){
      capture(p);
    }
    
    PositionVar()
      : fp_(nullptr),
      i_(RESERVE){}

    void capture(const QPointF& p){
      if(i_ == RESERVE){
        size_t n = dataVec_.size();

        dataVec_.reserve(n + RESERVE);
        plotVec_.reserve(n + RESERVE);
        i_ = 0;

        if(p.x() < xMin_){
          xMin_ = p.x();
        }
        if(p.x() > xMax_){
          xMax_ = p.x();
        }

        if(p.y() < yMin_){
          yMin_ = p.y();
        }
        if(p.y() > yMax_){
          yMax_ = p.y();
        }
      }
      else{
        ++i_;

        if(p.x() < xMin_){
          xMin_ = p.x();
        }
        else if(p.x() > xMax_){
          xMax_ = p.x();
        }

        if(p.y() < yMin_){
          yMin_ = p.y();
        }
        else if(p.y() > yMax_){
          yMax_ = p.y();
        }
      }

      dataVec_.push_back(p);
      plotVec_.push_back(p);
    }

    void capture(const QTransform& t, const QPointF& p){
      if(i_ == RESERVE){
        size_t n = plotVec_.size();

        plotVec_.reserve(n + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }

      plotVec_.emplace_back(t.map(p));
    }

    void compute(void* plot, uint64_t index) override{
      if(fp_){
        QPointF v;
        (*fp_)(plot, index, (double*)&v);
        capture(v);
      }
    }

    void compute(const QTransform& t, void* plot, uint64_t index) override{
      if(fp_){
        QPointF v;
        (*fp_)(plot, index, (double*)&v);
        capture(t, v);
      }
    };

    size_t hash(size_t i) const{
      assert(false && "attempt to hash from vector");
    }

    double xMin() const{
      return xMin_;
    }

    double xMax() const{
      return xMax_;
    }

    double yMin() const{
      return yMin_;
    }

    double yMax() const{
      return yMax_;
    }

    QPointF* getPoints(){
      return plotVec_.data();
    }

    size_t size() const{
      return plotVec_.size();
    }

    void clear(){
      dataVec_.clear();
      plotVec_.clear();
      double xMin_ = MAX;
      double xMax_ = MIN;
      double yMin_ = MAX;
      double yMax_ = MIN;
    }  

    void transform(const QTransform& t){
      size_t n = dataVec_.size();

      for(size_t i = 0; i < n; ++i){
        plotVec_[i] = t.map(dataVec_[i]);
      }
    }

    bool isConst() const override{
      return !fp_;
    }

  private:
    using PointVec = vector<QPointF>;

    void (*fp_)(void*, uint64_t, double*);

    PointVec dataVec_;
    PointVec plotVec_;

    double xMin_ = MAX;
    double xMax_ = MIN;
    double yMin_ = MAX;
    double yMax_ = MIN;
    size_t i_;
  };

  template<class T>
  class ConstVar : public ScalarVar<T>{
  public:
    ConstVar(T value)
      : value_(value){}

    T at(size_t i) const{
      return value_;
    }

    size_t size() const{
      return 0;
    }

    void set(T v){
      value_ = v;
    }

    bool isConst() const{
      return true;
    }

  private:
    T value_;
  };

  template<class T, size_t N>
  class ConstVecVar : public VarBase{
  public:
    ConstVecVar(const Vec<T, N>& v)
      : v_(v){}

    size_t hash(size_t i) const{
      assert(false && "attempt to hash from vector");
    }

    void* getVec(size_t i){
      return v_.raw();
    }

    size_t size() const{
      return 0;
    }

    bool isConst() const{
      return true;
    }

  private:
    Vec<T, N> v_;
  };

  class StringVar : public VarBase{
  public:
    StringVar()
      : i_(RESERVE){}

    void capture(const char* s){
      if(i_ == RESERVE){
        size_t n = v_.size();
        v_.reserve(n + RESERVE);
        i_ = 0;
      }
      else{
        ++i_;
      }
      
      v_.emplace_back(s);
    }

    size_t hash(size_t i) const{
      return std::hash<string>()(v_[i]);
    }

    size_t size() const{
      return v_.size();
    }

    void clear(){
      v_.clear();
    }

    const string& getString(size_t i) const{
      return v_[i];
    }

  private:
    using StringVec = vector<string>;
    
    StringVec v_;
    size_t i_;
  };

  class Frame{
  public:
    Frame(uint32_t width, uint32_t height, uint32_t depth)
      : width_(width),
        height_(height),
        depth_(depth){
      addBuiltinVars();
      addMeshBuiltinVars(width, height, depth);
    }

    Frame()
      : width_(0),
        height_(0),
        depth_(0){
      addBuiltinVars();
    }

    Frame(bool plotFrame)
      : width_(0),
        height_(0),
        depth_(0){}

    ~Frame(){}

    void addBuiltinVars(){
      indexVar_ = new IndexVar;
      addVar(0, indexVar_);
    }

    void addMeshBuiltinVars(uint32_t width, uint32_t height, uint32_t depth){
      addVar(1, new ConstVar<uint32_t>(width));
      addVar(2, new ConstVar<uint32_t>(height));
      addVar(3, new ConstVar<uint32_t>(depth));
      addVar(4, new XPosVar(width));
      addVar(5, new YPosVar(width, height));
      addVar(6, new ZPosVar(width, height));
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
      case ELEMENT_STRING:
        v = new StringVar();
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

    void capture(VarId varId, const char* value){
      assert(varId < vars_.size());
      
      static_cast<StringVar*>(vars_[varId])->capture(value);
    }

    void compute(Plot* plot, Frame* parentFrame){ 
      size_t end = parentFrame->size();
      size_t n = vars_.size();

      for(size_t i = 0; i < n; ++i){
        VarBase* v = vars_[i];

        for(size_t j = v->size(); j < end; ++j){
          v->compute(plot, j);
        }
      }
    }

    void compute(Plot* plot, Frame* parentFrame, const QTransform& t){ 
      size_t end = parentFrame->size();
      size_t n = vars_.size();

      for(size_t i = 0; i < n; ++i){
        VarBase* v = vars_[i];

        for(size_t j = v->size(); j < end; ++j){
          v->compute(t, plot, j);
        }
      }
    }

    size_t size() const{
      size_t n = vars_.size();
      size_t m = 0;
      
      for(size_t i = 0; i < n; ++i){
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
      return static_cast<ScalarVar<T>*>(getVar(varId))->at(index);
    }
    
  private:
    typedef vector<VarBase*> VarVec;

    VarVec vars_;
    uint32_t width_;
    uint32_t height_;
    uint32_t depth_;
    IndexVar* indexVar_;
  };

  void drawText(QPainter& painter,
                const QString& text,
                const QPointF& point,
                bool right=false,
                bool bottom=false){

    QRect bounds = painter.fontMetrics().boundingRect(text);

    QRectF frame(point.x() - bounds.width()/2.0 - 2.5,
                 point.y(),
                 bounds.width() + 5.0, bounds.height());

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

  QColor toQColor(float* v){    
    return QColor(v[0]*255, v[1]*255, v[2]*255, v[3]*255);
  }

  QColor toQColor(void* v){    
    return toQColor(static_cast<float*>(v));
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
      virtual VarId getPos() = 0;
    };

    class Lines : public RangeElement{
    public:
      Lines(VarId pos, VarId size, VarId color, VarId label)
        : pos(pos), size(size), color(color), label(label){}

      VarId pos;
      VarId size;
      VarId color;
      VarId label;

      int order(){
        return 1;
      }

      VarId getPos(){
        return pos;
      }
    };

    class Line : public Element{
    public:
      Line(VarId pos1, VarId pos2, VarId size, VarId color)
        : pos1(pos1), pos2(pos2), size(size), color(color){}

      VarId pos1;
      VarId pos2;
      VarId size;
      VarId color;

      int order(){
        return 1;
      }
    };

    class Points : public RangeElement{
    public:
      Points(VarId pos, VarId size, VarId color, VarId label)
        : pos(pos), size(size), color(color), label(label){}

      VarId pos;
      VarId size;
      VarId color;
      VarId label;

      int order(){
        return 2;
      }

      VarId getPos(){
        return pos;
      }
    };

    class Area : public RangeElement{
    public:
      Area(VarId pos, VarId color)
        : pos(pos), color(color){}

      VarId pos;
      VarId color;

      int order(){
        return 0;
      }

      VarId getPos(){
        return pos;
      }
    };

    class Interval : public RangeElement{
    public:
      Interval(VarId pos, VarId color)
        : pos(pos), color(color){}

      VarId pos;
      VarId color;

      int order(){
        return 0;
      }

      VarId getPos(){
        return pos;
      }
    };

    class Pie : public Element{
    public:
      Pie(VarId count, VarId color)
        : count(count), color(color){}

      VarId count;
      VarId color;

      int order(){
        return 0;
      }
    };

    class Bins : public Element{
    public:
      Bins(VarId varIn, VarId posOut, uint32_t n)
        : varIn(varIn), posOut(posOut), n(n){}

      VarId varIn;
      VarId posOut;
      uint32_t n;

      int order(){
        return 0;
      }
    };

    class AggregateBase : public Element{
    public:
      typedef vector<VarId> VarIdVec;

      virtual void compute(Plot* plot) = 0;

      void addVar(VarId varId){
        vars.push_back(varId);
      }
      
      int order(){
        return 0;
      }

      virtual VarBase* createRetVar() = 0;

    protected:
      VarIdVec vars;
    };

    template<class T>
    class Aggregate : public AggregateBase{
    public:
      Aggregate(uint64_t type, VarId ret)
        : type(type),
          ret(ret){}

      void compute(Plot* plot){
        switch(type){
        case AGG_SUM:{
          Var<T>* r = 
            static_cast<Var<T>*>(plot->getVar(ret));
          
          ScalarVar<T>* x = 
            static_cast<ScalarVar<T>*>(plot->getVar(vars[0]));

          r->capture(x->sum());

          break;
        }
        case AGG_MEAN:{
          Var<T>* r = 
            static_cast<Var<T>*>(plot->getVar(ret));
          
          ScalarVar<T>* x = 
            static_cast<ScalarVar<T>*>(plot->getVar(vars[0]));

          size_t size = x->size();

          if(size < 1){
            r->capture(0);
            break;
          }

          r->capture(x->sum() / size);

          break;
        }
        case AGG_VARIANCE:{
          Var<T>* r = 
            static_cast<Var<T>*>(plot->getVar(ret));
          
          ScalarVar<T>* x = 
            static_cast<ScalarVar<T>*>(plot->getVar(vars[0]));

          size_t size = x->size();

          if(size < 2){
            r->capture(0);
            break;
          }

          T m = x->sum() / size;
          T t = 0;
          for(size_t i = 0; i < size; ++i){
            T d = x->at(i) - m;
            t += d * d;
          }

          r->capture(t/(size - 1));

          break;
        }
        default:
          assert(false && "invalid aggregate type");
        }
      }

      VarBase* createRetVar(){
        switch(type){
        case AGG_SUM:
          return new Var<T>();
        case AGG_MEAN:
          return new Var<T>();
        case AGG_VARIANCE:
          return new Var<T>();
        default:
          assert(false && "invalid aggregate type");
        }
      }
    
    protected:
      uint64_t type;
      VarId ret;
    };

    class Proportion : public Element{
    public:
      Proportion(VarId posOut)
        : posOut(posOut){}

      void addVar(VarId var){
        vars.push_back(var);
      }

      int order(){
        return 0;
      }

      VarId posOut;

      typedef vector<VarId> VarVec;
      VarVec vars;
    };

    class Axis : public Element{
    public:
      Axis(uint32_t dim, const string& label, uint32_t major, uint32_t minor)
        : dim(dim),
          label(label),
          major(major),
          minor(minor){}

      uint32_t dim;
      string label;
      uint32_t major;
      uint32_t minor;

      int order(){
        return 3;
      }
    };

    Plot(uint64_t plotId)
      : first_(true),
        hasXLabel_(false),
        hasYLabel_(false),
        plotId_(plotId),
        frame_(nullptr),
        plotFrame_(0),
        window_(nullptr),
        antialiased_(true){}

    ~Plot(){
      if(pdfWriter_){
        delete pdfWriter_;
      }
    }

    void init(Frame* frame, ScoutWindow* scoutWindow){
      frame_ = frame;
      window_ = scoutWindow->getPlotWindow();
    }

    bool ready(){
      return window_;
    }

    void setAntialiased(bool flag){
      antialiased_ = flag;
    }

    void setOutputPath(const string& path){
      outputPath_ = path;
    }

    void setRange(bool x, double min, double max){
      if(x){
        hasXRange_ = true;
        xRangeMin_ = min;
        xRangeMax_ = max; 
      }
      else{
        hasYRange_ = true;
        yRangeMin_ = min;
        yRangeMax_ = max;  
      }
    }

    template<class T>
    T get(VarId varId, size_t index){
      return varId >= PLOT_VAR_BEGIN ? 
        plotFrame_->get<T>(varId - PLOT_VAR_BEGIN, index) : 
        frame_->get<T>(varId, index);
    }

    VarBase* getVar(VarId varId){
      return varId >= PLOT_VAR_BEGIN ? 
        plotFrame_->getVar(varId - PLOT_VAR_BEGIN) : frame_->getVar(varId);
    }

    PositionVar* getPos(VarId varId){
      return static_cast<PositionVar*>(getVar(varId));
    }

    StringVar* getStringVar(VarId varId){
      return static_cast<StringVar*>(getVar(varId));
    }

    template<class T>
    void addVar(VarId varId){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, new Var<T>());
    }

    template<class T, size_t N>
    void addVecVar(VarId varId){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, new VecVar<T, N>());
    }

    void addVar(VarId varId, VarBase* v){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    template<class T>
    void addVar(VarId varId,
                T (*fp)(void*, uint64_t),
                uint32_t flags){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      VarBase* v;
      if(flags & FLAG_VAR_CONSTANT){
        v = new ConstVar<T>((*fp)(0, 0));
      }
      else{
        v = new Var<T>(fp);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    void addPos(VarId varId, 
                void (*fp)(void*, uint64_t, double*),
                uint32_t flags){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      VarBase* v;

      if(flags & FLAG_VAR_CONSTANT){
        QPointF p;
        (*fp)(0, 0, (double*)&p);
        v = new PositionVar(p);
      }
      else{
        v = new PositionVar(fp);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    void addPos(VarId varId){
      if(!plotFrame_){
        plotFrame_ = new Frame(true);
      }

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, new PositionVar);
    }

    template<class T>
    void addVecVar(VarId varId,
                   void (*fp)(void*, uint64_t, T*),
                   uint32_t dim,
                   uint32_t flags){
      if(flags & FLAG_VAR_POSITION){
        addPos(varId, (void (*)(void*, uint64_t, double*))fp, flags);
        return;
      }

      if(!plotFrame_){
        plotFrame_ = new Frame(true);
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

      plotFrame_->addVar(varId - PLOT_VAR_BEGIN, v);
    }

    void addVar(uint32_t kind, VarId varId){
      switch(kind){
      case ELEMENT_INT32:
        addVar<int32_t>(varId);
        break;
      case ELEMENT_INT64:
        addVar<int64_t>(varId);
        break;
      case ELEMENT_FLOAT:
        addVar<float>(varId);
        break;
      case ELEMENT_DOUBLE:
        addVar<double>(varId);
        break;
      default:
        assert(false && "invalid kind");
      }
    }

    template<class T>
    void capture(VarId varId, T value){
      assert(plotFrame_);
      
      plotFrame_->capture(varId - PLOT_VAR_BEGIN, value);
    }

    void addLines(VarId pos, VarId size, VarId color, VarId label){
      elements_.push_back(new Lines(pos, size, color, label)); 
    }

    void addLine(VarId pos1,
                 VarId pos2,
                 VarId size,
                 VarId color){
      elements_.push_back(new Line(pos1, pos2, size, color)); 
    }

    void addPoints(VarId pos, VarId size, VarId color, VarId label){
      elements_.push_back(new Points(pos, size, color, label)); 
    }

    void addArea(VarId pos, VarId color){
      elements_.push_back(new Area(pos, color)); 
    }

    void addInterval(VarId pos, VarId color){
      elements_.push_back(new Interval(pos, color)); 
    }

    void addPie(VarId count, VarId color){
      elements_.push_back(new Pie(count, color)); 
    }

    void addBins(VarId varIn, VarId posOut, uint32_t n){
      elements_.push_back(new Bins(varIn, posOut, n));
      addPos(posOut);
    }

    AggregateBase* addAggregate(uint64_t type,
                                uint32_t retKind,
                                uint32_t retVarId){
      AggregateBase* a;

      switch(retKind){
      case ELEMENT_INT32:
        a = new Aggregate<int32_t>(type, retVarId);
        break;
      case ELEMENT_INT64:
        a = new Aggregate<int64_t>(type, retVarId);
        break;
      case ELEMENT_FLOAT:
        a = new Aggregate<float>(type, retVarId);
        break;
      case ELEMENT_DOUBLE:
        a = new Aggregate<double>(type, retVarId);
        break;
      default:
        assert(false && "invalid aggregate return kind");
      }

      addVar(retVarId, a->createRetVar());

      elements_.push_back(a);
      
      return a;
    }

    Proportion* addProportion(VarId posOut){
      Proportion* p = new Proportion(posOut);
      elements_.push_back(p);

      addPos(posOut);

      return p;
    }

    void addAxis(uint32_t dim,
                 const string& label,
                 uint32_t major,
                 uint32_t minor){
      elements_.push_back(new Axis(dim, label, major, minor));
      switch(dim){
      case 1:
        hasXLabel_ = !label.empty();
        break;
      case 2:
        hasYLabel_ = !label.empty();
        break;
      default:
        assert(false && "invalid axis dim");
      }
    }

    void finalize(){
      QtWindow::init();

      if(plotFrame_){
        for(Element* e : elements_){
          if(AggregateBase* a = dynamic_cast<AggregateBase*>(e)){
            a->compute(this);
          }
        }
        
        if(hasXRange_ && hasYRange_){
          plotFrame_->compute(this, frame_ ? frame_ : plotFrame_,
                              posTransform_);
        }
        else{
          plotFrame_->compute(this, frame_ ? frame_ : plotFrame_);
        }
      }

      widget_ = window_->getWidget();
      widget_->setRenderer(this);
      window_->show();
      window_->update();

      QtWindow::pollEvents();
    }

    void refresh(){
      QtWindow::pollEvents();
    }

    void prepare(QPainter& painter){
      width_ = painter.device()->width();
      height_ = painter.device()->height();
      scale_ = min(width_, height_)/1024.0;

      sort(elements_.begin(), elements_.end(),
           [](Element* a, Element* b){
             return a->order() < b->order();
           });

      QFont prevFont = painter.font();
      QFont font = prevFont;

      tickLabelSize_ = scaleFont(TICK_LABEL_SIZE);
      font.setPointSize(tickLabelSize_);
      painter.setFont(font);
      QRect bounds = painter.fontMetrics().boundingRect("?????????");
      tickLabelWidth_ = bounds.width();
      tickLabelHeight_ = bounds.height();

      axisLabelSize_ = scaleFont(AXIS_LABEL_SIZE);
      font.setPointSize(axisLabelSize_);
      painter.setFont(font);
      bounds = painter.fontMetrics().boundingRect("???");
      axisLabelHeight_ = bounds.height();

      painter.setFont(prevFont);

      double m = scale(MARGIN);

      if(hasYLabel_){
        origin_.setX(m + tickLabelWidth_ + axisLabelHeight_);
      }
      else{
        origin_.setX(m + tickLabelWidth_);
      }

      if(hasXLabel_){
        origin_.setY(height_ - m - tickLabelHeight_ - axisLabelHeight_);
      }
      else{
        origin_.setY(height_ - m - tickLabelHeight_);
      }

      xLen_ = width_ - origin_.x() - m;
      yLen_ = origin_.y() - m;

      xEnd_ = origin_;
      xEnd_ += QPointF(xLen_, 0.0);

      yEnd_ = origin_;
      yEnd_ -= QPointF(0.0, yLen_);

      first_ = false;
    }

    double scaleFont(double x){
      x *= scale_;

      if(x < MIN_FONT_SIZE){
        return MIN_FONT_SIZE;
      }
      else if(x > MAX_FONT_SIZE){
        return MAX_FONT_SIZE;
      }

      return x;
    }

    double scale(double x){
      return x * scale_;
    }

    double scaleSize(double x){
      x *= sizeScale_;
      return x < 1.0 ? 1.0 : x;
    }

    void calculateRanges(){
      xMin_ = MAX;
      xMax_ = MIN;
      yMin_ = MAX;
      yMax_ = MIN;

      for(Element* e : elements_){
        if(RangeElement* r = dynamic_cast<RangeElement*>(e)){
          PositionVar* p = getPos(r->getPos());

          if(p->xMin() < xMin_){
            xMin_ = p->xMin();
          }

          if(p->xMax() > xMax_){
            xMax_ = p->xMax();
          }

          if(p->yMin() < yMin_){
            yMin_ = p->yMin();
          }

          if(p->yMax() > yMax_){
            yMax_ = p->yMax();
          }
        }
        else if(Line* l = dynamic_cast<Line*>(e)){
          for(size_t i = 0; i < 2; ++i){
            PositionVar* p = getPos(i == 0 ? l->pos1 : l->pos2);

            if(p->xMin() < xMin_){
              xMin_ = p->xMin();
            }

            if(p->xMax() > xMax_){
              xMax_ = p->xMax();
            }

            if(p->yMin() < yMin_){
              yMin_ = p->yMin();
            }

            if(p->yMax() > yMax_){
              yMax_ = p->yMax();
            }
          }
        }
      }
    }

    void calculateXRange(){
      xMin_ = MAX;
      xMax_ = MIN;

      for(Element* e : elements_){
        if(RangeElement* r = dynamic_cast<RangeElement*>(e)){
          PositionVar* p = getPos(r->getPos());

          if(p->xMin() < xMin_){
            xMin_ = p->xMin();
          }

          if(p->xMax() > xMax_){
            xMax_ = p->xMax();
          }
        }
        else if(Line* l = dynamic_cast<Line*>(e)){
          for(size_t i = 0; i < 2; ++i){
            PositionVar* p = getPos(i == 0 ? l->pos1 : l->pos2);

            if(p->xMin() < xMin_){
              xMin_ = p->xMin();
            }

            if(p->xMax() > xMax_){
              xMax_ = p->xMax();
            }
          }
        }
      }
    }

    void calculateYRange(){
      yMin_ = MAX;
      yMax_ = MIN;

      for(Element* e : elements_){
        if(RangeElement* r = dynamic_cast<RangeElement*>(e)){
          PositionVar* p = getPos(r->getPos());

          if(p->yMin() < yMin_){
            yMin_ = p->yMin();
          }

          if(p->yMax() > yMax_){
            yMax_ = p->yMax();
          }
        }
        else if(Line* l = dynamic_cast<Line*>(e)){
          for(size_t i = 0; i < 2; ++i){
            PositionVar* p = getPos(i == 0 ? l->pos1 : l->pos2);

            if(p->yMin() < yMin_){
              yMin_ = p->yMin();
            }

            if(p->yMax() > yMax_){
              yMax_ = p->yMax();
            }
          }
        }
      }
    }

    void render(){
      QPaintDevice* device;
      if(outputPath_.empty()){
        device = widget_;
      }
      else{
        if(!pdfWriter_){
          pdfWriter_= new QPdfWriter(outputPath_.c_str());
          QSizeF size(widget_->width(), widget_->height());
          pdfWriter_->setPageSize(QPageSize(size, QPageSize::Point));
          pdfWriter_->setResolution(74);
        }

        device = pdfWriter_;
      }

      QPainter painter(device);

      if(first_){
        prepare(painter);
      }

      size_t frameSize = frame_ ? frame_->size() : plotFrame_->size();

      if(frameSize < 2){
        return;
      }

      painter.setRenderHint(QPainter::Antialiasing, antialiased_);

      for(Element* e : elements_){
        if(Bins* b = dynamic_cast<Bins*>(e)){
          VarBase* varIn = getVar(b->varIn);

          size_t size = varIn->size();

          if(size < 2){
            continue;
          }

          PositionVar* posOut = getPos(b->posOut);

          double min = MAX;
          double max = MIN;
          for(size_t i = 0; i < size; ++i){
            double x = varIn->get(i);
            if(x < min){
              min = x;
            }

            if(x > max){
              max = x;
            }
          }

          typedef map<double, size_t> BinMap;
          BinMap binMap;

          double binWidth = (max - min)/b->n;
          double start = min + binWidth + EPSILON;

          for(size_t i = 0; i < b->n; ++i){
            binMap.insert({start, 0});
            start += binWidth;
          }

          for(size_t i = 0; i < size; ++i){
            auto itr = binMap.lower_bound(varIn->get(i));
            assert(itr != binMap.end());
            ++itr->second;
          }
            
          posOut->clear();
            
          for(auto& itr : binMap){
            posOut->capture({double(itr.first), double(itr.second)});
          }
        }
        else if(Proportion* p = dynamic_cast<Proportion*>(e)){
          typedef vector<VarBase*> VarVec;
          VarVec vs;

          for(VarId var : p->vars){
            vs.push_back(getVar(var));
          }

          size_t n = vs.size();
          size_t size = vs[0]->size();

          typedef map<size_t, size_t> PropMap;
          PropMap propMap;

          for(size_t i = 0; i < size; ++i){
            size_t h = 0;
            for(size_t j = 0; j < n; ++j){
              h ^= vs[j]->hash(i);
            }

            auto itr = propMap.find(h);
            if(itr == propMap.end()){
              propMap[h] = 1;
            }
            else{
              ++itr->second;
            }
          }

          PositionVar* posOut = getPos(p->posOut);
          posOut->clear();
            
          size_t i = 0;
          for(auto& itr : propMap){
            posOut->capture({double(i++), double(itr.second)});
          }
        }
      }

      bool shouldTransform = true;

      if(hasXRange_){
        xMin_ = xRangeMin_;
        xMax_ = xRangeMax_;
        if(hasYRange_){
          yMin_ = yRangeMin_;
          yMax_ = yRangeMax_;
          shouldTransform = false;
        }
        else{
          calculateYRange();
        }
      }
      else if(hasYRange_){
        yMin_ = yRangeMin_;
        yMax_ = yRangeMax_;
        calculateXRange();
      }
      else{
        calculateRanges();
      }

      xSpan_ = xMax_ - xMin_;
      ySpan_ = yMax_ - yMin_;
      xm_ = xLen_/xSpan_;
      ym_ = yLen_/ySpan_;
      sizeScale_ = min(width_, height_)/1024.0;

      QTransform transform;
      transform.translate(origin_.x(), origin_.y());
      painter.setWorldTransform(transform);

      posTransform_.reset();
      posTransform_.scale(xm_, -ym_);
      posTransform_.translate(-xMin_, -yMin_);
      
      for(Element* e : elements_){
        QTransform t;
        painter.setWorldTransform(t);

        if(Axis* a = dynamic_cast<Axis*>(e)){
          QFont prevFont = painter.font();

          if(a->dim == 1){
            QFont font = prevFont;
            font.setPointSize(tickLabelSize_);
            painter.setFont(font);

            painter.drawLine(origin_, xEnd_);

            size_t inc = double(frameSize) / a->major;

            if(inc == 0){
              inc = 1;
            }

            bool shouldRound = xSpan_ > a->major;

            double xc;
            double xv; 
            for(size_t i = 0; i < frameSize; i += inc){
              xv = xMin_ + (double(i)/frameSize)*xSpan_;
              xc = origin_.x() + double(i)/frameSize * xLen_;

              if(shouldRound){
                xv = round(xv);
              }
              
              drawText(painter,
                       toLabel(xv),
                       QPointF(xc, origin_.y() + 3.0));
            }
            
            inc = double(frameSize) / (a->minor * a->major);

            if(inc == 0){
              inc = 1;
            }

           for(size_t i = 0; i < frameSize; i += inc){
             xc = origin_.x() + double(i)/frameSize * xLen_;
             
             painter.drawLine(QPointF(xc, origin_.y() + 3),
                              QPointF(xc, origin_.y() - 3));
           }

            if(!a->label.empty()){
              QFont font = prevFont;
              font.setPointSize(axisLabelSize_);
              painter.setFont(font);

              drawText(painter, a->label.c_str(),
                       QPointF(origin_.x() + xLen_/2.0,
                               origin_.y() + tickLabelHeight_));
            }
          }
          else if(a->dim == 2){
            QFont font = prevFont;
            font.setPointSize(tickLabelSize_);
            painter.setFont(font);

            painter.drawLine(origin_, yEnd_);

            size_t inc = double(frameSize) / a->major;

            if(inc == 0){
              inc = 1;
            }

            bool shouldRound = ySpan_ > a->major;

            double yc;
            double yv;
            for(size_t i = 0; i <= frameSize; i += inc){
              yv = yMin_ + ySpan_ * double(i)/frameSize;
              yc = origin_.y() - double(i)/frameSize * yLen_;

              if(shouldRound){
                yv = round(yv);
              }

              drawText(painter,
                       toLabel(yv),
                       QPointF(origin_.x(), yc - scale(8.0)), true);
            }

            inc = double(frameSize) / (a->minor * a->major);
            
            if(inc == 0){
              inc = 1;
            }

            for(size_t i = 0; i < frameSize; i += inc){
              yc = origin_.y() - double(i)/frameSize * yLen_;

              painter.drawLine(QPointF(origin_.x() - 3, yc),
                               QPointF(origin_.x() + 3, yc));
            }

            if(!a->label.empty()){
              QFont font = prevFont;
              font.setPointSize(axisLabelSize_);
              painter.setFont(font);
              painter.rotate(-90);

              drawText(painter, a->label.c_str(),
                       QPointF(origin_.y() - yLen_/2 - height_,
                               origin_.x() - 
                               tickLabelWidth_ - axisLabelHeight_));
            }
          }
          else{
            assert(false && "invalid axis dim");
          }

          painter.setFont(prevFont);
        }

        painter.setWorldTransform(transform);
      }

      for(Element* e : elements_){
        if(Lines* l = dynamic_cast<Lines*>(e)){
          PositionVar* p = getPos(l->pos);

          if(shouldTransform){
            p->transform(posTransform_);
          }

          QPointF* points = p->getPoints();

          VarBase* s = getVar(l->size);
          VarBase* c = getVar(l->color);

          size_t size = p->size();

          QPen pen;

          if(s->isConst() && c->isConst()){
            pen.setWidthF(scaleSize(s->get(0)));
            pen.setColor(toQColor(c->getVec(0)));
            painter.setPen(pen);
            painter.drawPolyline(points, size);
          }
          else{
            for(size_t i = 1; i < size; ++i){
              pen.setWidthF(scaleSize(s->get(i)));
              pen.setColor(toQColor(c->getVec(i)));
              painter.setPen(pen);
              painter.drawLine(points[i - 1], points[i]);
            }
          }

          if(l->label != nullVarId){
            QFont prevFont = painter.font();
            QFont font = prevFont;
            font.setPointSize(tickLabelSize_);
            painter.setFont(font);

            StringVar* label = getStringVar(l->label);
            for(size_t i = 0; i < size; ++i){
              drawText(painter, label->getString(i).c_str(), points[i]);
            }
            
            painter.setFont(prevFont);
          }
        }
        else if(Line* l = dynamic_cast<Line*>(e)){
          PositionVar* p1 = getPos(l->pos1);
          PositionVar* p2 = getPos(l->pos2);

          if(shouldTransform){
            p1->transform(posTransform_);
            p2->transform(posTransform_);
          }
          
          VarBase* s = getVar(l->size);
          VarBase* c = getVar(l->color);

          assert(p1->size() == p2->size());
          
          size_t size = p1->size();

          QPointF* points1 = p1->getPoints();
          QPointF* points2 = p2->getPoints();
          QPen pen;

          if(s->isConst() && c->isConst()){
            pen.setWidthF(s->get(0));
            pen.setColor(toQColor(c->getVec(0)));
            painter.setPen(pen);
            
            for(size_t i = 0; i < size; ++i){
              painter.drawLine(points1[i], points2[i]);
            }
          }
          else{
            for(size_t i = 0; i < size; ++i){
              pen.setWidthF(s->get(i));
              pen.setColor(toQColor(c->getVec(i)));
              painter.setPen(pen);
              painter.drawLine(points1[i], points2[i]);
            }
          }
        }
        else if(Area* a = dynamic_cast<Area*>(e)){
          PositionVar* p = getPos(a->pos);
          
          if(shouldTransform){
            p->transform(posTransform_);
          }
          
          QPointF* points = p->getPoints();

          VarBase* c = getVar(a->color);

          size_t size = p->size();

          QPen noPen(Qt::NoPen);
          painter.setPen(noPen);

          if(c->isConst()){
            QBrush brush(toQColor(c->getVec(0)));
            painter.setBrush(brush);

            for(size_t i = 1; i < size; ++i){
              QPolygonF poly;
              poly << points[i - 1] << points[i] << 
                QPointF(points[i].x(), yMin_) << 
                QPointF(points[i - 1].x(), yMin_);
      
              painter.drawPolygon(poly);
            }
          }
          else{
            for(size_t i = 1; i < size; ++i){
              QBrush brush(toQColor(c->getVec(i)));
              painter.setBrush(brush);

              QPolygonF poly;
              poly << points[i - 1] << points[i] << 
                QPointF(points[i].x(), yMin_) << 
                QPointF(points[i - 1].x(), yMin_);
      
              painter.drawPolygon(poly);
            }
          }
        }
        else if(Points* p = dynamic_cast<Points*>(e)){
          PositionVar* pv = getPos(p->pos);
          
          if(shouldTransform){
            pv->transform(posTransform_);
          }
          
          QPointF* points = pv->getPoints();

          VarBase* s = getVar(p->size);
          VarBase* c = getVar(p->color);

          size_t size = pv->size();

          QPointF point;
          QPen pen;
          pen.setCapStyle(Qt::RoundCap);
          
          if(s->isConst() && c->isConst()){
            pen.setColor(toQColor(c->getVec(0)));
            pen.setWidth(scaleSize(s->get(0)));
            painter.setPen(pen);
            painter.drawPoints(points, size);
          }
          else{
            for(size_t i = 0; i < size; ++i){
              pen.setColor(toQColor(c->getVec(i)));
              pen.setWidth(scaleSize(s->get(i)));
              painter.setPen(pen);
              painter.drawPoint(points[i]);
            }
          }

          if(p->label != nullVarId){
            QFont prevFont = painter.font();
            QFont font = prevFont;
            font.setPointSize(tickLabelSize_);
            painter.setFont(font);

            StringVar* l = getStringVar(p->label);
            for(size_t i = 0; i < size; ++i){
              drawText(painter, l->getString(i).c_str(), points[i]);
            }

            painter.setFont(prevFont);
          }
        }
        else if(Interval* i = dynamic_cast<Interval*>(e)){
          PositionVar* p = getPos(i->pos);

          if(shouldTransform){
            p->transform(posTransform_);
          }

          VarBase* c = getVar(i->color);

          size_t size = p->size();

          if(size < 2){
            continue;
          }

          QPointF* points = p->getPoints();

          double width = xLen_/size;

          for(size_t j = 0; j < size; ++j){
            painter.fillRect(QRectF(points[j].x(), 0,
                                    width, points[j].y()),
                             toQColor(c->getVec(j)));
          }
        }
        else if(Pie* p = dynamic_cast<Pie*>(e)){
          PositionVar* pn = getPos(p->count);
          VarBase* c = p->color > 0 ? getVar(p->color) : nullptr;
          
          QPointF* points = pn->getPoints();

          size_t size = pn->size();

          double total = 0;
          for(size_t i = 0; i < size; ++i){
            total += points[i].y();
          }

          double side = min(xEnd_.x() - origin_.x(), origin_.y() - yEnd_.y());

          QRectF rect(0, 0, side, -side);
          
          Random rng;

          QPen pen;
          pen.setWidthF(3.0);
          painter.setPen(pen);

          int startAngle = 0;
          for(size_t i = 0; i < size; ++i){
            if(c){              
              QBrush brush(toQColor(c->getVec(i)));
              painter.setBrush(brush);
            }
            else{
              QColor color(rng.equilikely(0, 255),
                           rng.equilikely(0, 255),
                           rng.equilikely(0, 255));

              QBrush brush(color);
              painter.setBrush(brush);
            }

            int spanAngle = points[i].y()/total * 360 * 16;
            painter.drawPie(rect, startAngle, spanAngle);

            startAngle += spanAngle;
          }
        }
      }
    }

  private:
    typedef vector<Element*> ElementVec_;

    bool first_;
    uint64_t plotId_;
    Frame* frame_;
    Frame* plotFrame_;
    PlotWindow* window_;
    PlotWidget* widget_;
    QPdfWriter* pdfWriter_ = nullptr;

    ElementVec_ elements_;
    double xLen_;
    double yLen_;
    QPointF origin_;
    QPointF xEnd_;
    QPointF yEnd_;
    bool hasXLabel_;
    bool hasYLabel_;
    double xMin_;
    double xMax_;
    double xSpan_;
    double yMin_;
    double yMax_;
    double ySpan_;
    double xm_;
    double ym_;
    double width_;
    double height_;
    bool antialiased_;
    string outputPath_;
    double scale_;
    double sizeScale_;
    double tickLabelSize_;
    double tickLabelWidth_;
    double tickLabelHeight_;
    double axisLabelSize_;
    double axisLabelHeight_;
    bool hasXRange_ = false;
    double xRangeMin_;
    double xRangeMax_;
    bool hasYRange_ = false;
    double yRangeMin_;
    double yRangeMax_;
    QTransform posTransform_;
  };

} // end namespace

extern "C"{

  void* __scrt_create_frame(){
    return new Frame();
  }

  void* __scrt_create_mesh_frame(uint32_t width,
                                 uint32_t height,
                                 uint32_t depth){
    return new Frame(width, height, depth);
  }

  void __scrt_frame_add_var(void* f, VarId varId, uint32_t elementKind){
    static_cast<Frame*>(f)->addVar(varId, elementKind);
  }

  void __scrt_frame_add_array_var(void* f,
                                  VarId varId,
                                  uint32_t elementKind,
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

  void __scrt_frame_capture_string(void* f, VarId varId, const char* value){
    static_cast<Frame*>(f)->capture(varId, value);
  }

  void* __scrt_plot_get(uint64_t plotId){
    _mutex.lock();
    
    if(!_global){
      _global = new Global;
    }
    
    Plot* plot = _global->getPlot(plotId);
    
    if(plot){
      _mutex.unlock();
      return plot;
    }
    else{
      plot = new Plot(plotId);
      _global->putPlot(plotId, plot);
    }

    _mutex.unlock();
    
    return plot;
  }

  void __scrt_plot_init(void* plot, void* frame, void* window){
    return static_cast<Plot*>(plot)->init(static_cast<Frame*>(frame),
                                          static_cast<ScoutWindow*>(window));
  }

  bool __scrt_plot_ready(void* plot){
    return static_cast<Plot*>(plot)->ready();
  }

  int32_t __scrt_plot_get_i32(void* plot, VarId varId, uint64_t index){
    return static_cast<Plot*>(plot)->get<int32_t>(varId, index);
  }

  int64_t __scrt_plot_get_i64(void* plot, VarId varId, uint64_t index){
    return static_cast<Plot*>(plot)->get<int64_t>(varId, index);
  }

  float __scrt_plot_get_float(void* plot, VarId varId, uint64_t index){
    return static_cast<Plot*>(plot)->get<float>(varId, index);
  }

  double __scrt_plot_get_double(void* plot, VarId varId, uint64_t index){
    return static_cast<Plot*>(plot)->get<double>(varId, index);
  }

  void __scrt_plot_add_var(void* plot, VarId varId, uint32_t elementKind){
    static_cast<Plot*>(plot)->addVar(elementKind, varId);
  }

  void __scrt_plot_capture_i32(void* plot, VarId varId, int32_t value){
    static_cast<Plot*>(plot)->capture(varId, value);
  }

  void __scrt_plot_capture_i64(void* plot, VarId varId, int64_t value){
    static_cast<Plot*>(plot)->capture(varId, value);
  }

  void __scrt_plot_capture_float(void* plot, VarId varId, float value){
    static_cast<Plot*>(plot)->capture(varId, value);
  }

  void __scrt_plot_capture_double(void* plot, VarId varId, double value){
    static_cast<Plot*>(plot)->capture(varId, value);
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
                             VarId pos,
                             VarId size,
                             VarId color,
                             VarId label){
    static_cast<Plot*>(plot)->addLines(pos, size, color, label);
  }

  void __scrt_plot_set_antialiased(void* plot, bool flag){
    static_cast<Plot*>(plot)->setAntialiased(flag);
  }

  void __scrt_plot_set_output(void* plot, char* path){
    static_cast<Plot*>(plot)->setOutputPath(path);
  }

  void __scrt_plot_set_range(void* plot, bool x, double min, double max){
    static_cast<Plot*>(plot)->setRange(x, min, max);
  }

  void __scrt_plot_add_line(void* plot,
                            VarId pos1,
                            VarId pos2,
                            VarId size,
                            VarId color){
    static_cast<Plot*>(plot)->addLine(pos1, pos2, size, color);
  }

  void __scrt_plot_add_points(void* plot,
                              VarId pos,
                              VarId size,
                              VarId color,
                              VarId label){
    static_cast<Plot*>(plot)->addPoints(pos, size, color, label);
  }

  void __scrt_plot_add_area(void* plot,
                            VarId pos,
                            VarId color){
    static_cast<Plot*>(plot)->addArea(pos, color);
  }

  void __scrt_plot_add_interval(void* plot,
                                VarId pos,
                                VarId color){
    static_cast<Plot*>(plot)->addInterval(pos, color);
  }

  void __scrt_plot_add_pie(void* plot,
                           VarId count,
                           VarId color){
    static_cast<Plot*>(plot)->addPie(count, color);
  }

  void __scrt_plot_add_bins(void* plot,
                            VarId varIn,
                            VarId posOut,
                            uint32_t n){
    static_cast<Plot*>(plot)->addBins(varIn, posOut, n);
  }

  void* __scrt_plot_add_aggregate(void* plot,
                                  uint64_t type,
                                  uint32_t retKind,
                                  uint32_t retVarId){
    return static_cast<Plot*>(plot)->addAggregate(type, retKind, retVarId);
  }

  void __scrt_aggregate_add_var(void* aggregate, VarId varId){
    static_cast<Plot::AggregateBase*>(aggregate)->addVar(varId);
  }

  void* __scrt_plot_add_proportion(void* plot, VarId posOut){
    return static_cast<Plot*>(plot)->addProportion(posOut);
  }

  void __scrt_plot_proportion_add_var(void* proportion, VarId var){
    return static_cast<Plot::Proportion*>(proportion)->addVar(var);
  }

  void __scrt_plot_add_axis(void* plot,
                            uint32_t dim,
                            const char* label,
                            uint32_t major,
                            uint32_t minor){
    static_cast<Plot*>(plot)->addAxis(dim, label, major, minor);
  }

  void __scrt_plot_render(void* plot){
    static_cast<Plot*>(plot)->finalize();
  }

  void __scrt_plot_refresh(void* plot){
    static_cast<Plot*>(plot)->refresh();
  }

} // end extern "C"
