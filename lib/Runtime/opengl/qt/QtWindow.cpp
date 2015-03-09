#include "scout/Runtime/opengl/qt/QtWindow.h"

#include <iostream>
#include <mutex>

#include <QtCore/QCoreApplication>

#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLPaintDevice>
#include <QtGui/QPainter>

#include <QApplication>

#include "scout/Runtime/opengl/glRenderable.h"

using namespace std;
using namespace scout;

namespace{

  class Global{
  public:
    Global(){
      argc_ = 1;
      argv_[0] = strdup("scout");
      app_ = new QApplication(argc_, argv_);
    }

    void pollEvents(){
      app_->processEvents(QEventLoop::AllEvents, 1);
    }

  private:
    int argc_;
    char* argv_[1];
    QApplication* app_;
  };
  
  Global* _global = 0;
  mutex _mutex;

} // end namespace

QtWindow::QtWindow(unsigned short width, unsigned short height, QWindow* parent)
  : QWindow(parent),
    context_(0),
    device_(0),
    currentRenderable_(0){

  setSurfaceType(QWindow::OpenGLSurface);

  QSurfaceFormat format;
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setVersion(4, 1);
  format.setSamples(16);
  setFormat(format);

  resize(width, height);
}

QtWindow::~QtWindow(){}

void QtWindow::init(){
  _mutex.lock();
  if(!_global){
    _global = new Global;
  }
  _mutex.unlock();
}

void QtWindow::pollEvents(){
  _mutex.lock();
  _global->pollEvents();
  _mutex.unlock();
}

void QtWindow::makeContextCurrent(){
  if(!context_){
    context_ = new QOpenGLContext(this);
    context_->setFormat(requestedFormat());
    context_->create();
  }

  context_->makeCurrent(this);

  if(!device_){
    device_ = new QOpenGLPaintDevice;
  }

  device_->setSize(size());
}

void QtWindow::paint(){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  for(glRenderable* r : renderables_) {
    r->render(NULL);
  }
}

void QtWindow::swapBuffers(){
  context_->swapBuffers(this);
}
