#ifndef __SC_QT_WINDOW_H__
#define __SC_QT_WINDOW_H__

#include <deque>

#include <QtGui/QWindow>

class QOpenGLContext;
class QOpenGLPaintDevice;

namespace scout{

  class glRenderable;

  class QtWindow : public QWindow{
    Q_OBJECT

  public:
    QtWindow(unsigned short width, unsigned short height, QWindow* parent=0);

    ~QtWindow();

    static void init();

    static void pollEvents();

    void makeContextCurrent();

    void addRenderable(glRenderable* renderable){ 
      renderables_.push_back(renderable);
    }

    void makeCurrentRenderable(glRenderable* rend){
      currentRenderable_ = rend;
    }
  
    glRenderable* getCurrentRenderable(){
      return currentRenderable_;
    }

    void paint();

    void swapBuffers();

  private:
    typedef std::deque<glRenderable*> Renderables;

    Renderables renderables_;
    glRenderable* currentRenderable_;
  
    QOpenGLContext* context_;
    QOpenGLPaintDevice* device_;
  };

} // end namespace scout

#endif // __SC_QT_WINDOW_H__
