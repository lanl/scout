#ifndef GL_WIDGET_H
#define GL_WIDGET_H

#include <QtWidgets>
#include <QVector3D>
#include <QMatrix4x4>

class Trackball;
class Rotation;
class StopWatchInterface;

namespace scout{
    class glRenderer;
}

using namespace scout;

class GLWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
public:
  explicit GLWidget(QWidget *parent = 0);
    ~GLWidget();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

    void SetRenderable(void* r);

    void GetWindowSize(int &w, int &h) {w = width; h = height;}

    size_t getWidth(){
      return width;
    }

    size_t getHeight(){
      return height;
    }
  
protected:
    virtual void initializeGL() Q_DECL_OVERRIDE;
    virtual void paintGL() Q_DECL_OVERRIDE;
    virtual void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    virtual void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
    virtual void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;

    uint width = 512, height = 512;

private:
    void computeFPS();

    void printModelView();

    void cleanup();

    void TimerStart();

    void TimerEnd();


    QPointF pixelPosToViewPos(const QPointF& p);

    /*****view*****/
    //transformation states
    QVector3D transVec = QVector3D(0.0f, 0.0f, -5.0f);//move it towards the front of the camera
    QMatrix4x4 transRot;
    float transScale = 1;

    Trackball *trackball;
    QPointF prevPos;//previous mouse position
    Rotation *rot;

    /****timing****/
    StopWatchInterface *timer = 0;
    int m_frame;
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 16;        // FPS limit for sampling
    int g_Index = 0;
    unsigned int frameCount = 0;

    glRenderer* renderer;
};

#endif //GL_WIDGET_H
