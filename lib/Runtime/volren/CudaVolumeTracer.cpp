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
