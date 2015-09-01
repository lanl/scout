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


#include "glwidget.h"

#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include "Tracer.h"


#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

#include <Trackball.h>
#include <Rotation.h>

#if USE_PBO
#include "cuda_helper.h"
#include <cuda_gl_interop.h>
#endif

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
    renderer = (Tracer*)r;
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

}

void GLWidget::cleanup()
{
    sdkDeleteTimer(&timer);
    renderer->cleanup();
#if USE_PBO
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    if(cuda_pbo_resource)
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
#endif
}

void GLWidget::computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        //qDebug() << "Volume Render: "<<ifps;
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
    TimerStart();
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

#if USE_PBO
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));
    renderer->SetDeviceImage(d_output);

#endif

    renderer->draw(modelview, projection);

#if USE_PBO
    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
#else
    uint* img = renderer->GetHostImage();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, img);
#endif
    TimerEnd();
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
    renderer->resize(w, h);



    if(!initialized) {
#if USE_PBO


        if (pbo)
        {
            // delete old buffer
            glDeleteBuffers(1, &pbo);
            glDeleteTextures(1, &tex);
        }


        // create pixel buffer object for display
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_DYNAMIC_DRAW_ARB);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        renderer->SetPBO(pbo);


        // create texture for display
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        // register this buffer object with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
#endif
        //make init here because the window size is not updated in InitiateGL()
        renderer->init();
        initialized = true;
    } else {
#if USE_PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height*sizeof(GLubyte)*4, 0, GL_DYNAMIC_DRAW_ARB);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
#endif
    }

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

