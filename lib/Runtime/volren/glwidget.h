/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the documentation of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <Trackball.h>

#include <helper_timer.h>
#include <cuda_runtime.h>
#include <QtWidgets>
#include <QVector3D>

//#include <DataMgr.h>

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
  explicit GLWidget(QWidget *parent = 0);
  ~GLWidget();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

    virtual void cleanup();

protected:
    virtual void initializeGL() Q_DECL_OVERRIDE;
    virtual void paintGL() Q_DECL_OVERRIDE;
    virtual void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    virtual void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
    virtual void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;


private:
    void cuda_render();
    void initPixelBuffer();
    void updatePixelBuffer();
    void* loadRawFile(const char *filename, size_t size);
    void computeFPS();
    void saveImage(const char* filename);
    void saveMultiImage();
    void printModelView();

    QPointF pixelPosToViewPos(const QPointF& p);

    void init_();

//    GLuint m_posAttr;
//    GLuint m_colAttr;
//    GLuint m_matrixUniform;

    int m_frame;


    uint width = 512, height = 512;

    /*****cuda*****/
    dim3 blockSize;
    dim3 gridSize;

    /*****view*****/
    QVector3D viewTranslation = QVector3D(0.0, 0.0, -4.0f);
    float invViewMatrix[12];

    //rotation
    Trackball trackball;
    QPointF prevPos;//previous mouse position
    Rotation rot;

    //translation
//    float m_distExp = 0;
    QVector3D transVec = QVector3D(0.0f, 0.0f, 0.0f);

    /*****shading*****/
    float density = 0.05f;
    float brightness = 1.0f;
    float transferOffset = 0.0f;
    float transferScale = 1.0f;
    bool linearFiltering = true;

    /****opengl rendering****/
    GLuint pbo = 0;     // OpenGL pixel buffer object
    GLuint tex = 0;     // OpenGL texture object
    struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
    GLfloat modelView[16];

    /****timing****/
    StopWatchInterface *timer = 0;

    // Auto-Verification Code
//    const int frameCheckNumber = 2;
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 16;        // FPS limit for sampling
    int g_Index = 0;
    unsigned int frameCount = 0;

    cudaExtent volumeSize;
    bool initialized_;
//    TrackBall m_trackBalls[3];

  //DataMgr dataMgr;

};
//! [1]

