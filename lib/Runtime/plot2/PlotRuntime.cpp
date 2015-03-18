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

typedef double (*__sc_plot_func)(void*, uint64_t);

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

  const uint32_t COMPUTED_VAR_BEGIN = 65536;

  class VarBase{
  public:
    virtual size_t size() const = 0;
    
    virtual double min() const = 0;

    virtual double max() const = 0;

    virtual double get(size_t i) const = 0;
  };

  template<class T>
  class Var : public VarBase{
  public:
    Var()
      : i_(RESERVE),
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

    size_t size() const{
      return v_.size();
    }

    double get(size_t i) const{
      return v_[i];
    }

    double min() const{
      return min_;
    }

    double max() const{
      return max_;
    }

  private:
    vector<T> v_;
    size_t i_;
    T min_;
    T max_;
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

    VarBase* getVar(VarId varId){
      assert(varId < vars_.size());
      return vars_[varId];
    }

  private:
    typedef vector<VarBase*> VarVec;

    VarVec vars_;
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

    class Lines : public Element{
    public:
      Lines(VarId xVarId, VarId yVarId, double size)
        : xVarId(xVarId), yVarId(yVarId), size(size){}

      VarId xVarId;
      VarId yVarId;
      double size;

      int order(){
        return 1;
      }
    };

    class Points : public Element{
    public:
      Points(VarId xVarId, VarId yVarId, double size)
        : xVarId(xVarId), yVarId(yVarId), size(size){}

      VarId xVarId;
      VarId yVarId;
      double size;

      int order(){
        return 2;
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

    Plot(Frame* frame, PlotWindow* window)
      : frame_(frame),
        computedFrame_(0),
        window_(window){}

    ~Plot(){
      if(computedFrame_){
        delete computedFrame_;
      }
    }

    VarBase* getVar(VarId varId){
      if(varId >= COMPUTED_VAR_BEGIN){
        assert(computedFrame_);
        return computedFrame_->getVar(varId - COMPUTED_VAR_BEGIN);
      }

      return frame_->getVar(varId);
    }

    void addLines(VarId xVarId, VarId yVarId, double size){
      elements_.push_back(new Lines(xVarId, yVarId, size)); 
    }

    void addPoints(VarId xVarId, VarId yVarId, double size){
      elements_.push_back(new Points(xVarId, yVarId, size)); 
    }

    void addAxis(uint32_t dim, const string& label){
      elements_.push_back(new Axis(dim, label)); 
    }

    void addComputedVar(VarId varId, __sc_plot_func fp){
      if(!computedFrame_){
        computedFrame_ = new Frame;
      }

      computedFrame_->addVar(varId - COMPUTED_VAR_BEGIN, ELEMENT_DOUBLE);
      varFuncMap_[varId] = fp;
    }

    void finalize(){
      QtWindow::init();

      if(computedFrame_){
        size_t size = frame_->size();

        for(auto& itr : varFuncMap_){
          VarId varId = itr.first - COMPUTED_VAR_BEGIN;
          
          for(size_t i = 0; i < size; ++i){
            double value = (*itr.second)(frame_, i);
            computedFrame_->capture(varId, value);
          }
        }
      }

      widget_ = window_->getWidget();
      widget_->setRenderer(this);
      window_->show();
      window_->update();

      QtWindow::pollEvents();
    }
    
    void render(){
      QPainter painter(widget_);
      painter.setRenderHint(QPainter::Antialiasing, true);

      sort(elements_.begin(), elements_.end(),
           [](Element* a, Element* b){
             return a->order() > b->order();
           });

      double xMin = MAX;
      double xMax = MIN;
      double yMin = MAX;
      double yMax = MIN;

      size_t size;

      for(Element* e : elements_){
        if(Lines* l = dynamic_cast<Lines*>(e)){
          VarBase* x = getVar(l->xVarId);
          VarBase* y = getVar(l->yVarId);

          size = x->size();

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
        else if(Points* p = dynamic_cast<Points*>(e)){
          VarBase* x = getVar(p->xVarId);
          VarBase* y = getVar(p->yVarId);

          size = x->size();

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

            size_t inc = size / X_LABELS;
            double xc;
            for(size_t i = 0; i < size; i += inc){
              xc = origin.x() + double(i)/size * xLen;

              drawText(painter,
                       toLabel(xMin + (double(i)/size)*xSpan),
                       QPointF(xc, height - BOTTOM_MARGIN + 3));
            }
            
            inc = size / X_TICKS;
            for(size_t i = 0; i < size; i += inc){
              xc = origin.x() + double(i)/size * xLen;

              painter.drawLine(QPointF(xc, height - BOTTOM_MARGIN + 3),
                               QPointF(xc, height - BOTTOM_MARGIN - 3));
            }
          }
          else if(a->dim == 2){
            painter.drawLine(origin, yEnd);

            size_t inc = size / Y_LABELS;
            double yc;
            double yv;
            for(size_t i = 0; i <= size; i += inc){
              yv = yMin + ySpan * double(i)/size;
              yc = origin.y() - double(i)/size * yLen;

              drawText(painter,
                       toLabel(yv),
                       QPointF(LEFT_MARGIN - 5, yc), true);
            }

            inc = size / Y_TICKS;
            for(size_t i = 0; i < size; i += inc){
              yc = origin.y() - double(i)/size * yLen;

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
          VarBase* x = getVar(l->xVarId);
          VarBase* y = getVar(l->yVarId);

          QPen pen;
          pen.setWidthF(l->size);
          pen.setColor(QColor(0, 0, 0));

          QPen noPen(Qt::NoPen);

          QPointF lastPoint;
          QPointF point;

          QColor color(255, 0, 0);
          QBrush brush(color);
      
          painter.setBrush(brush);

          for(size_t i = 0; i < size; ++i){
            point.setX(origin.x() + ((x->get(i) - xMin)/xSpan) * xLen);
            point.setY(origin.y() - ((y->get(i) - yMin)/ySpan) * yLen);

            if(i > 0){
              QPolygonF poly;
              poly << lastPoint << point << 
                QPointF(point.x(), origin.y()) << 
                QPointF(lastPoint.x(), origin.y());
      
              painter.setPen(noPen);
              painter.drawPolygon(poly);
            }

            if(i > 0){
              painter.setPen(pen);
              painter.drawLine(point, lastPoint);
            }

            lastPoint = point;
          }
        }
        else if(Points* p = dynamic_cast<Points*>(e)){
          VarBase* x = getVar(p->xVarId);
          VarBase* y = getVar(p->yVarId);

          QPen pen;
          pen.setWidthF(1.0);
          pen.setColor(QColor(0, 0, 0));

          QPointF point;

          QColor color(0, 0, 0);
          QBrush brush(color);
                
          painter.setBrush(brush);

          for(size_t i = 0; i < size; ++i){
            point.setX(origin.x() + ((x->get(i) - xMin)/xSpan) * xLen);
            point.setY(origin.y() - ((y->get(i) - yMin)/ySpan) * yLen);

            painter.drawEllipse(point, p->size, p->size);
          }
        }
      }
    }

  private:
    typedef vector<Element*> ElementVec_;
    typedef map<VarId, __sc_plot_func> VarFuncMap_;

    Frame* frame_;
    
    Frame* computedFrame_;
    VarFuncMap_ varFuncMap_;

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
                    static_cast<PlotWindow*>(window));
  }

  void __scrt_plot_add_computed_var(void* plot,
                                    VarId varId,
                                    __sc_plot_func fp){
    static_cast<Plot*>(plot)->addComputedVar(varId, fp);
  }

  void __scrt_plot_add_lines(void* plot,
                             VarId xVarId,
                             VarId yVarId,
                             double size){
    static_cast<Plot*>(plot)->addLines(xVarId, yVarId, size);
  }

  void __scrt_plot_add_points(void* plot,
                              VarId xVarId,
                              VarId yVarId,
                              double size){
    static_cast<Plot*>(plot)->addPoints(xVarId, yVarId, size);
  }

  void __scrt_plot_add_axis(void* plot, uint32_t dim, const char* label){
    static_cast<Plot*>(plot)->addAxis(dim, label);
  }

  void __scrt_plot_render(void* plot){
    static_cast<Plot*>(plot)->finalize();
  }

} // end extern "C"
