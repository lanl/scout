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



#include "CudaVolumeTracer.h"
//#include "CUDAMarchingCubes.h"
//#include "DataMgr.h"
#include "iostream"
#include "cuda_helper.h"

//extern "C" void setTextureFilterMode(bool bLinearFilter);
//extern "C" void initCudaVolumeRendering(void *h_volume, int nx, int ny, int nz);
//extern "C" void freeCudaBuffers();
//extern "C"void render_kernel_host(dim3 gridSize, dim3 blockSize, thrust::host_vector<uint>* h_vec_image, uint imageW, uint imageH,
//                   int3 partitionStart, int3 partitionSize,
//                   float density, float brightness, float transferOffset, float transferScale);
//extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
//                   int3 partitionStart, int3 partitionSize,
//                   float density, float brightness, float transferOffset, float transferScale);
//extern "C" void copyInvPVMMatrix(float *c_invPVMMatrix, size_t sizeofMatrix);


using namespace scout;

inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


//CudaVolumeTracer::CudaVolumeTracer(){
//
//}


void CudaVolumeTracer::init() {
//    void *volume = dataMgr->GetData();//loadRawFile(volumeFilename, size);
//    size_t dim[3];
//    dataMgr->GetDataDim(dim);
//    initCudaVolumeRendering(volume, dim[0], dim[1], dim[2]);

}

void CudaVolumeTracer::draw(float modelview[16], float projection[16]) {
    float invPVM[16];
    GetInvPVM(modelview, projection, invPVM);
    //copyInvPVMMatrix(invPVM, sizeof(float4)*4);

    size_t dataDim[3];
//    dataMgr->GetDataDim(dataDim);
    // call CUDA kernel, writing results to PBO
//    render_kernel(gridSize, blockSize, d_output, winWidth, winHeight,
//                  make_int3(0, 0, 0),
//                  make_int3(dataDim[0], dataDim[1], dataDim[2]),
//                  density, brightness, transferOffset, transferScale);
    callback_->render_kernel(d_output, invPVM);
#if USE_PBO
#else
    cudaMemcpy(h_output, d_output, winWidth * winHeight * sizeof(uint), cudaMemcpyDeviceToHost);
#endif
}

void CudaVolumeTracer::cleanup() {
#if USE_PBO
#else
    if (d_output) cudaFree(d_output);
#endif
}

void CudaVolumeTracer::resize(int width, int height)
{
    Tracer::resize(width, height);
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

#if USE_PBO
#else
    if(d_output == NULL) {
        if(cudaMalloc((void**) &d_output, sizeof(uint) * winWidth * winHeight) != cudaSuccess) {
            std::cout<<"memory allocation failed..."<<std::endl;
            exit(2);
        }
    }
    //AllocOutImage();
#endif


    //updatePixelBuffer();
}
