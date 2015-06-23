#include "glwidget.h"

#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include "glRenderer.h"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

#include <Trackball.h>
#include <Rotation.h>

using namespace std;

void GLWidget::printModelView()
{

    glMatrixMode(GL_MODELVIEW);
    GLfloat modelView[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    std::fstream fs;
    fs.open ("parameters.txt", std::fstream::out);

    fs <<"modelview matrix by glGetFloatv(GL_MODELVIEW_MATRIX, modelView):\n";
    for(int i = 0; i < 16; i++)
        fs << modelView[i]<< " ";
    fs << std::endl;

    fs <<"window size:"<< width<<" "<<height<<"\n";
    fs <<"the smaller z value is, the farther the point is from the camera"<<"\n";
    fs <<"image file format: ABGR ABGR ABGR"<<"\n";

    fs.close();
}

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_frame(0)
{
    setFocusPolicy(Qt::StrongFocus);
    sdkCreateTimer(&timer);

    trackball = new Trackball();
    rot = new Rotation();

    transRot.setToIdentity();
}

void GLWidget::SetRenderable(void* r)
{
    renderer = (glRenderer*)r;
}

GLWidget::~GLWidget()
{
    cleanup();
}

QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
    return QSize(width, height);
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    sdkCreateTimer(&timer);
    renderer->initializeGL();
}

void GLWidget::cleanup()
{
    sdkDeleteTimer(&timer);
    renderer->cleanup();
}

void GLWidget::computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        qDebug() << "Volume Render: "<<ifps;
        fpsCount = 0;
//        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

void GLWidget::TimerStart()
{
    sdkStartTimer(&timer);
}

void GLWidget::TimerEnd()
{
    sdkStopTimer(&timer);
    computeFPS();
}


void GLWidget::paintGL() {
    /****transform the view direction*****/
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(transVec[0], transVec[1], transVec[2]);
    glMultMatrixf(transRot.data());
    glScalef(transScale, transScale, transScale);

    int dataDim[3];
    renderer->GetDataDim(dataDim[0], dataDim[1], dataDim[2]); 
    int maxDim = std::max(std::max(dataDim[0], dataDim[1]), dataDim[2]);
    float scale = 2.0f / maxDim;
    glScalef(scale, scale, scale);
    glTranslatef(- dataDim[0] * 0.5, - dataDim[1] * 0.5, - dataDim[2] * 0.5);

    GLfloat modelview[16];
    GLfloat projection[16];

    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);

    renderer->draw(modelview, projection);
    //cout << "draw!" << endl;
}

void Perspective(float fovyInDegrees, float aspectRatio,
                      float znear, float zfar)
{
    float ymax, xmax;
    ymax = znear * tanf(fovyInDegrees * M_PI / 360.0);
    xmax = ymax * aspectRatio;
    glFrustum(-xmax, xmax, -ymax, ymax, znear, zfar);
}

void GLWidget::resizeGL(int w, int h)
{
    width = w;
    height = h;
    renderer->resizeGL(w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    Perspective(45, (float)width / height, 0.1,10e4);
//    glOrtho(-1.5,1.5,-1.5,1.5,-1.5,1.5);

    glMatrixMode(GL_MODELVIEW);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    QPointF pos = event->pos();

    if (event->buttons() & Qt::LeftButton) {
        QPointF from = pixelPosToViewPos(prevPos);
        QPointF to = pixelPosToViewPos(pos);
        *rot = trackball->rotate(from.x(), from.y(),
                               to.x(), to.y());
        float m[16];
        rot->matrix(m);
        QMatrix4x4 qm = QMatrix4x4(m).transposed();
        transRot = qm * transRot;

    } else if (event->buttons() & Qt::RightButton) {
        QPointF diff = pixelPosToViewPos(pos) - pixelPosToViewPos(prevPos);
        transVec[0] += diff.x();
        transVec[1] += diff.y();
    }
    prevPos = pos;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    QPointF pos = event->pos();
    prevPos = pos;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
    transScale *= exp(event->delta() * -0.003);
    update();
}

void GLWidget::keyPressEvent(QKeyEvent * event)
{
    if(event->key() == Qt::Key_S) {
        renderer->saveMultiImage();
        printModelView();
    } else if(event->key() == Qt::Key_T)
        renderer->saveImage("full_image");
}

QPointF GLWidget::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width - 1.0,
                   1.0 - 2.0 * float(p.y()) / height);
}

