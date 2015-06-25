/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

//#include <helper_cuda.h>
#include "cuda_helper.h"
#include <helper_math.h>
#include <thrust/device_vector.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct
{
    float4 m[4];
} float4x4;

// transform vector by matrix with translation
__device__
float4 mul(const float4x4 &M, const float4 &v)
{
    float4 r;
    r.w = dot(v, M.m[3]);
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);

    return r;
}

__device__
float4 divW(float4 v)
{
    float invW = 1 / v.w;
    return(make_float4(v.x * invW, v.y * invW, v.z * invW, 1.0f));
}

__constant__ float4x4 c_invPVMMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         int3 partitionStart, int3 partitionSize,
         float density, float brightness,
         float transferOffset, float transferScale)
{
    const int maxSteps = max(max(partitionSize.x, partitionSize.y), partitionSize.z);
    const float tstep = 1.0f;
    const float opacityThreshold = 0.95f;

    const float3 boxMin = make_float3(partitionStart.x, partitionStart.y, partitionStart.z);
    const float3 boxMax = make_float3(boxMin.x + partitionSize.x - 1,
                                      boxMin.y + partitionSize.y - 1,
                                      boxMin.z + partitionSize.z - 1);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float)imageW)*2.0f-1.0f;
    float v = (y / (float)imageH)*2.0f-1.0f;

    //unproject eye ray from clip space to object space
    //unproject: http://gamedev.stackexchange.com/questions/8974/how-can-i-convert-a-mouse-click-to-a-ray
    Ray eyeRay;
    eyeRay.o = make_float3(divW(mul(c_invPVMMatrix, make_float4(u, v, 2.0f, 1.0f))));
    float3 eyeRay_t = make_float3(divW(mul(c_invPVMMatrix, make_float4(u, v, -1.0f, 1.0f))));
    eyeRay.d = normalize(eyeRay_t - eyeRay.o);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        float sample = tex3D(tex, pos.x, pos.y , pos.z);
        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
        col.w *= density;

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
uint* GenDeviceImage(int size)
{
    thrust::device_vector<uint> d_vec_image(size);
    return thrust::raw_pointer_cast(d_vec_image.data());
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCudaVolumeRendering(void *h_volume, int nx, int ny, int nz)//cudaExtent volumeSize)
{
    cudaExtent volumeSize = make_cudaExtent(nx, ny, nz);

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // set texture parameters
    tex.normalized = false;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   int3 partitionStart, int3 partitionSize,
                   float density, float brightness, float transferOffset, float transferScale)
{
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, partitionStart, partitionSize,
                                      density, brightness, transferOffset, transferScale);
}

//this function returns an image in host memory
extern "C"
void render_kernel_host(dim3 gridSize, dim3 blockSize, thrust::host_vector<uint>* h_vec_image, uint imageW, uint imageH,
                   int3 partitionStart, int3 partitionSize,
                   float density, float brightness, float transferOffset, float transferScale)
{
    thrust::device_vector<uint> d_vec_image(imageW * imageH);
    uint* d_output = thrust::raw_pointer_cast(d_vec_image.data());
    render_kernel(gridSize, blockSize, d_output, imageW, imageH,
                  partitionStart, partitionSize,
                  density, brightness, transferOffset, transferScale);

    thrust::copy(d_vec_image.begin(), d_vec_image.end(), h_vec_image->begin());
}

extern "C"
void copyInvPVMMatrix(float *invPVMMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invPVMMatrix, invPVMMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
