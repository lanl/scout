#include "glVolumeRenderer.h"

// CUDA Runtime, Interop, and includes
#include <cuda_gl_interop.h>

// CUDA utilities
//#include <helper_cuda.h>
#include "cuda_helper.h"

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

//others
#include <math.h>
#include <QImage>
#include <QMatrix4x4>

#include <iostream>

using namespace std;
using namespace scout;

void SaveImage(uint* image, int width, int height, const char* filename)
{
  QImage img((uchar*)image, width, height,QImage::Format_ARGB32);
  QString filename_png = QString ("%1.png").arg(filename);
  img.save(filename_png.toLatin1().constData());
}

void SaveRawImage(uint* image, int width, int height, const char* filename)
{
  QString filename_raw = QString ("%1.raw").arg(filename);
  FILE *file = fopen(filename_raw.toLatin1().constData(), "wb");
  fwrite(image, sizeof(*image), width * height, file);
  fclose(file);
}

glVolumeRenderer::glVolumeRenderer(size_t width, size_t height, size_t depth)
  : glRenderer(width, height, depth),
    ready_(false){
}

void glVolumeRenderer::draw(float modelview[16], float projection[16])
{
  init_();
  
  QMatrix4x4 q_modelview(modelview);
  q_modelview = q_modelview.transposed();

  QMatrix4x4 q_projection(projection);
  q_projection = q_projection.transposed();

  QMatrix4x4 q_invProjMulView = (q_projection * q_modelview).inverted();

  q_invProjMulView.copyDataTo(invPVM);

  /****draw the texture****/

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  cuda_render();
  // display results
  glClear(GL_COLOR_BUFFER_BIT);

  // draw image from PBO
  glDisable(GL_DEPTH_TEST);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // copy from pbo to texture
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
  glPopMatrix();
}


void glVolumeRenderer::updatePixelBuffer()
{
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, winWidth*winHeight*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void glVolumeRenderer::initializeGL()
{

}

void glVolumeRenderer::init_(){
  if(ready_){
    return;
  }

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
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, winWidth*winHeight*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  // register this buffer object with CUDA
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

  // create texture for display
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  ready_ = true;
}

void glVolumeRenderer::resizeGL(int width, int height)//WindowResize(int w, int h)
{
  glRenderer::resizeGL(width, height);
  updatePixelBuffer();
}


// render image using CUDA
void glVolumeRenderer::cuda_render()
{
  // map PBO to get CUDA device pointer
  uint *d_output;
  // map PBO to get CUDA device pointer
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                       cuda_pbo_resource));
  //    printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

  // clear image
  checkCudaErrors(cudaMemset(d_output, 0, winWidth*winHeight*4));

  callback_->render_kernel(d_output, invPVM);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}



/****Save image****/
void glVolumeRenderer::saveImage(const char* filename)
{
  // map PBO to get CUDA device pointer
  uint *d_output;
  // map PBO to get CUDA device pointer
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                       cuda_pbo_resource));

  thrust::device_ptr<uint> d_ptr_image(d_output);
  thrust::host_vector<uint> h_vec_image(winWidth * winHeight);
  thrust::copy(d_ptr_image, d_ptr_image + winWidth * winHeight, h_vec_image.begin());
  SaveImage(h_vec_image.data(), winWidth, winHeight, filename);
  SaveRawImage(h_vec_image.data(), winWidth, winHeight, filename);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

}

void glVolumeRenderer::saveMultiImage()
{
  /*
    int blockDim [] = {4,4,3};
    size_t dataDim[3];
    dataMgr->GetDataDim(dataDim);
    int3 cellDim = make_int3(
    ceil((float)dataDim[0] / blockDim[0]),
    ceil((float)dataDim[1] / blockDim[1]),
    ceil((float)dataDim[2] / blockDim[2]));

    int3* partitionStart = new int3[blockDim[0] * blockDim[1] * blockDim[2]];
    int3* partitionSize = new int3[blockDim[0] * blockDim[1] * blockDim[2]];
    thrust::host_vector<uint> h_vec_image(winWidth * winHeight);
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

    render_kernel_host(gridSize, blockSize, &h_vec_image, winWidth, winHeight,
    p_start, p_size,
    density, brightness, transferOffset, transferScale);
    QString filename = QString ("partition%1_%2_%3").arg(i).arg(j).arg(k);
    SaveImage(h_vec_image.data(), winWidth, winHeight, filename.toStdString().c_str());
    SaveRawImage(h_vec_image.data(), winWidth, winHeight, filename.toStdString().c_str());

    partitionStart[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k] = p_start;
    partitionSize[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k] = p_size;

    int3 bst = partitionStart[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k];
    int3 bsz = partitionSize[i * blockDim[1] * blockDim[2] + j * blockDim[2] + k];
    std::cout<<"blockStart:"<<bst.x << " "<< bst.y << " "<< bst.z <<std::endl;
    std::cout<<"blockSize:"<<bsz.x << " "<< bsz.y << " "<< bsz.z <<std::endl;
    }
    }
    }
    delete [] partitionStart;
    delete [] partitionSize;
  */
}

void glVolumeRenderer::cleanup()
{
  if (pbo)
    {
      cudaGraphicsUnregisterResource(cuda_pbo_resource);
      glDeleteBuffers(1, &pbo);
      glDeleteTextures(1, &tex);
    }
  cudaDeviceReset();
}
