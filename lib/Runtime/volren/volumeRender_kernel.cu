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

#include <helper_cuda.h>
//#include "cuda_helper.h"
#include <helper_math.h>
//#include <thrust/device_vector.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct{
  float4 m[4];
} float4x4;

typedef float4 (*transfer_fp)(void*, uint);

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

//__constant__ float4x4 c_invPVMMatrix;  // inverse view matrix

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

extern "C" __device__ void
volume_render(uint *d_output, float* invMatPtr,
              float* test,
              uint imageW, uint imageH,
              uint startX, uint startY, uint startZ,
              uint width, uint height, uint depth,
              float density, float brightness,
              float transferOffset, float transferScale,
              void* mesh, transfer_fp tfp)
{
  float4x4 invMat = *((float4x4*)invMatPtr);

    const int maxSteps = max(max(width, height), depth);
    const float tstep = 1.0f;
    const float opacityThreshold = 0.95f;

    const float3 boxMin = make_float3(startX, startY, startZ);
    const float3 boxMax = make_float3(boxMin.x + width - 1,
                                      boxMin.y + height - 1,
                                      boxMin.z + depth - 1);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    test[y*imageW + x] = 99999.9;

    float u = (x / (float)imageW)*2.0f-1.0f;
    float v = (y / (float)imageH)*2.0f-1.0f;

    //unproject eye ray from clip space to object space
    //unproject: http://gamedev.stackexchange.com/questions/8974/how-can-i-convert-a-mouse-click-to-a-ray
    Ray eyeRay;
    eyeRay.o = make_float3(divW(mul(invMat, make_float4(u, v, 2.0f, 1.0f))));
    float3 eyeRay_t = make_float3(divW(mul(invMat, make_float4(u, v, -1.0f, 1.0f))));
    eyeRay.d = normalize(eyeRay_t - eyeRay.o);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;

    /*
    float4 testcolor;
    testcolor.x = 0.5;
    testcolor.y = 0.5;
    testcolor.z = 0.5;
    testcolor.w = 1.0;
    d_output[y*imageW + x] = rgbaFloatToInt(testcolor);
    return;
    */

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {

        //float sample = tex3D(tex, pos.x, pos.y, pos.z);
        // lookup in transfer function texture
        //float4 col = 
        //tex1D(transferTex, (sample-transferOffset)*transferScale);

      //uint index = pos.x;

      uint index = 
        width * height * uint(pos.z) + 
        width * uint(pos.y) + 
        uint(pos.x);

      //float extent = 16*16*16;

      float4 col = tfp(mesh, index);

      /*
      if(col.x == 0.0f || col.w == 0.0f){
        col = make_float4(1.0, 0.0f, 0.0f, 1.0f);
      }
      */

      /*
      float4 col;
      col.x = index/extent;
      col.y = 0;
      col.z = 0;
      col.w = 0.5f;
      */

        test[y*imageW + x] = col.x;

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

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
