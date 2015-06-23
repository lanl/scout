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

#include "glwidget.h"

// CUDA Runtime, Interop, and includes
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <vector_functions.h>

#include <iostream>

/*
#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char *file, int line)
{

    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    if (glErr != GL_NO_ERROR)
    {
        printf("glError in file %s @ line %d: %s\n",
                 file, line, gluErrorString(glErr));
        retCode = 1;
    }
    return retCode;
}
*/

using namespace std;

//extern "C" uint* GetDeviceImage(int size);
extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel_host(dim3 gridSize, dim3 blockSize, thrust::host_vector<uint>* h_vec_image, uint imageW, uint imageH, uint3 vSize,
                   int3 dataSize, int3 partitionStart, int3 partitionSize,
                   float density, float brightness, float transferOffset, float transferScale);
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, uint3 vSize,
                   int3 dataSize, int3 partitionStart, int3 partitionSize,
                   float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

// render image using CUDA
void GLWidget::cuda_render()
{
  /*
    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height,
                  make_uint3(volumeSize.width, volumeSize.height, volumeSize.depth),
                  make_int3(meshDim_[0], meshDim_[1], meshDim_[2]),
                  make_int3(0, 0, 0),
                  make_int3(meshDim_[0], meshDim_[1], meshDim_[2]),
                  density, brightness, transferOffset, transferScale);


    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
  */
}

/****Save image****/
void GLWidget::saveImage(const char* filename)
{
    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));

    thrust::device_ptr<uint> d_ptr_image(d_output);
    thrust::host_vector<uint> h_vec_image(width * height);
    thrust::copy(d_ptr_image, d_ptr_image + width * height, h_vec_image.begin());
    //dataMgr.SaveImage(h_vec_image.data(), width, height, filename);
    //dataMgr.SaveRawImage(h_vec_image.data(), width, height, filename);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

}

void GLWidget::saveMultiImage()
{
  /*
    // map PBO to get CUDA device pointer

//    // map PBO to get CUDA device pointer
//    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
//    size_t num_bytes;
//    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
//                                                         cuda_pbo_resource));

//    thrust::device_vector<int> dv;
//    thrust::device_vector<uint> d_vec_image;//(width * height);
//    d_vec_image.assign(width * height, 0);
    int blockDim [] = {4,4,3};

    size_t dataDim[3];
    //dataMgr.GetDataDim(dataDim);
    int3 cellDim = make_int3(
        ceil((float)dataDim[0] / blockDim[0]),
        ceil((float)dataDim[1] / blockDim[1]),
        ceil((float)dataDim[2] / blockDim[2]));

    int3* partitionStart = new int3[blockDim[0] * blockDim[1] * blockDim[2]];
    int3* partitionSize = new int3[blockDim[0] * blockDim[1] * blockDim[2]];
    thrust::host_vector<uint> h_vec_image(width * height);
    for(int i = 0; i < blockDim[0]; i++)    {
        for(int j = 0; j < blockDim[1]; j++)    {
            for(int k = 0; k < blockDim[2]; k++)    {
                int3 p_start, p_size;
                p_start = make_int3(i * cellDim.x, j * cellDim.y, k * cellDim.z);
                p_size = cellDim;

                if(i == (blockDim[0] - 1))
                    p_size.x = dataDim[0] - (blockDim[0] - 1) * cellDim.x;
                if(j == (blockDim[1] - 1))
                    p_size.y = dataDim[1] - (blockDim[1] - 1) * cellDim.y;
                if(k == (blockDim[2] - 1))
                    p_size.z = dataDim[2] - (blockDim[2] - 1) * cellDim.z;

                render_kernel_host(gridSize, blockSize, &h_vec_image, width, height,
                              make_uint3(volumeSize.width, volumeSize.height, volumeSize.depth),
                              make_int3(dataDim[0], dataDim[1], dataDim[2]), p_start, p_size,
                              density, brightness, transferOffset, transferScale);
                QString filename = QString ("partition%1_%2_%3").arg(i).arg(j).arg(k);
                //dataMgr.SaveImage(h_vec_image.data(), width, height, filename.toStdString().c_str());
                //dataMgr.SaveRawImage(h_vec_image.data(), width, height, filename.toStdString().c_str());

                partitionStart[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k] = p_start;
                partitionSize[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k] = p_size;

                int3 bst = partitionStart[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k];
                int3 bsz = partitionSize[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k];
                std::cout<<"blockStart:"<<bst.x << " "<< bst.y << " "<< bst.z <<std::endl;
                std::cout<<"blockSize:"<<bsz.x << " "<< bsz.y << " "<< bsz.z <<std::endl;
            }
        }
    }

//    int totalNumPartition = blockDim[0] * blockDim[1] * blockDim[2];
//    //uint* d_output = GetDeviceImage(width*height);
//    thrust::host_vector<uint> h_vec_image(width * height);
//    for(int i = 0; i < totalNumPartition; i++) {
//        render_kernel_host(gridSize, blockSize, &h_vec_image, width, height,
//                      make_uint3(volumeSize.width, volumeSize.height, volumeSize.depth),
//                      make_int3(dataDim[0], dataDim[1], dataDim[2]), partitionStart[i], partitionSize[i],
//                      density, brightness, transferOffset, transferScale);
//        QString filename = QString ("partition%1.png").arg(i);
//        dataMgr.SaveImage(h_vec_image.data(), width, height, filename.toStdString().c_str());
//    }

//    uint d_output[2] = {1,2};

//    render_kernel(gridSize, blockSize, d_output, width, height,
//                  make_uint3(volumeSize.width, volumeSize.height, volumeSize.depth),
//                  density, brightness, transferOffset, transferScale);

//    thrust::host_vector<uint> h_vec_image(width * height);
//     h_vec_image = d_vec_image;
////    thrust::copy(d_ptr_image, d_ptr_image + width * height, h_vec_image.begin());
//    dataMgr.SaveImage(h_vec_image.data(), width, height, filename);

//    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
*/

}


void GLWidget::printModelView()
{
    std::fstream fs;
    fs.open ("parameters.txt", std::fstream::out);

    fs <<"modelview matrix by glGetFloatv(GL_MODELVIEW_MATRIX, modelView):\n";
    for(int i = 0; i < 16; i++)
        fs << modelView[i]<< " ";
    fs << std::endl;

    size_t dim[3];
    //dataMgr.GetDataDim(dim);
    fs <<"data dimension: "<<dim[0] << " "<<dim[1]<<" "<<dim[2]<<"\n";

    fs <<"window size:"<< width<<" "<<height<<"\n";
    fs <<"the smaller z value is, the farther the point is from the camera"<<"\n";
    fs <<"image file format: ABGR ABGR ABGR"<<"\n";

    fs.close();
}

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_frame(0)
    , initialized_(false)
{
    blockSize.x = 16;
    blockSize.y = 16;
    setFocusPolicy(Qt::StrongFocus);

  //  dataMgr = new DataMgr();

//    m_trackBalls[0] = TrackBall(0.05f, QVector3D(0, 1, 0), TrackBall::Sphere);
//    m_trackBalls[1] = TrackBall(0.005f, QVector3D(0, 0, 1), TrackBall::Sphere);
//    m_trackBalls[2] = TrackBall(0.0f, QVector3D(0, 1, 0), TrackBall::Plane);

    sdkCreateTimer(&timer);
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
    return QSize(4060, 1440);
}

void GLWidget::init_(){
  /*
    initPixelBuffer();

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    size_t extent = meshDim_[0] * meshDim_[1] * meshDim_[2];

    volumeSize = make_cudaExtent(meshDim_[0], meshDim_[1], meshDim_[2]);
    float* h_volume_test = (float*)malloc(sizeof(float) * extent);

    void *h_volume;

 
    float start = 0;
    float inc = 1.0f/extent;
    for(size_t i = 0; i < extent; ++i){
      h_volume_test[i] = start;
      start += inc;
    }

    h_volume = h_volume_test;

    initCuda(h_volume, volumeSize);

    sdkCreateTimer(&timer);
  */ 
}

void GLWidget::initializeGL()
{

    //this following line has to run after the OpenGL context is generated
    initializeOpenGLFunctions();

    //this following line has to be called after OpenGL initialization
    //glViewport(0, 0, width, height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void GLWidget::cleanup()
{
  /*
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
  */
}

void GLWidget::updatePixelBuffer()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLWidget::initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    
    // create pixel buffer object for display
    glGenBuffers(1, &pbo);

    assert(pbo != 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  
    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);  
}

// Load raw data from disk
void* GLWidget::loadRawFile(const char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif

    return data;
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

void GLWidget::paintGL() {
   if(!initialized_){
     init_();
     initialized_ = true;
   } 

  //makeCurrent();

    sdkStartTimer(&timer);

    /****transform the view direction*****/
    // use OpenGL to build view matrix

    glMatrixMode(GL_MODELVIEW);

    //the rotation accumulate
    float m[16];
    rot.matrix(m);
    glMultMatrixf(m);

    // make sure the translation will not accumulate
    glPushMatrix();
    glTranslatef(-viewTranslation[0] + transVec[0], -viewTranslation[1] + transVec[1], -viewTranslation[2] + transVec[2]);
//    float scaleFactor = 1.0f / std::max(std::max(volumeSize.width, volumeSize.height), volumeSize.depth);
//    glScalef(scaleFactor, scaleFactor, scaleFactor);
//    glTranslatef(- volumeSize.width * 0.5, - volumeSize.height * 0.5, - volumeSize.depth * 0.5);
//    float scale = pow(2, m_distExp);
//    glScalef(scale, scale, scale);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    glPopMatrix();

    /****draw the texture****/
    glPushMatrix();
    glLoadIdentity();

    cuda_render();

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

    sdkStopTimer(&timer);
    computeFPS();
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void GLWidget::resizeGL(int w, int h)
{
    width = w;//s.width();
    height = h;//s.height();
    gridSize = dim3(iDivUp(w, blockSize.x), iDivUp(h, blockSize.y));

    updatePixelBuffer();//the buffer needs update
    //The following line is done automatically already
 //   glViewport(0, 0, w, h);

}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    QPointF pos = event->pos();
    pos.setX( (width - 1 - pos.x()) / width);
    pos.setY(pos.y() / height);

    if (event->buttons() & Qt::LeftButton) {
        rot = trackball.rotate(prevPos.x() , prevPos.y(), pos.x(), pos.y());
    } else if (event->buttons() & Qt::RightButton) {
        QPointF diff = 4 * (pos - prevPos);
        transVec[0] += diff.x();
        transVec[1] += diff.y();
    }
    prevPos = pos;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    QPointF pos = event->pos();
    pos.setX( (width - 1 - pos.x()) / width);
    pos.setY(pos.y() / height);
    prevPos = pos;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
    }
    update();
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
    transVec[2] += event->delta() * 0.002;
    update();
}

void GLWidget::keyPressEvent(QKeyEvent * event)
{
    if(event->key() == Qt::Key_S) {
        //saveImage("counterflow.png");
        saveMultiImage();
        printModelView();
    } else if(event->key() == Qt::Key_T)
        saveImage("full_image");
}

QPointF GLWidget::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width - 1.0,
                   1.0 - 2.0 * float(p.y()) / height);
}
