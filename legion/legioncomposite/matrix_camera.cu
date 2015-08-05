
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>
#include "helpers.h"
//#include "cuda_helper.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

rtDeclareVariable(optix::Matrix4x4,        invPVM, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
//rtBuffer<uchar4, 2>              output_buffer;
//rtBuffer<float3, 2>              accum_buffer;
rtBuffer<float4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(unsigned int, frame_number, , );


//#define TIME_VIEW

__device__
float4 divW(float4 v)
{
    float invW = 1 / v.w;
    return(make_float4(v.x * invW, v.y * invW, v.z * invW, 1.0f));
}

RT_PROGRAM void matrix_camera()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = make_float3(divW(invPVM * make_float4(d.x, d.y, 2.0f, 1.0f)));
  float3 eyeRay_t = make_float3(divW(invPVM * make_float4(d.x, d.y, -1.0f, 1.0f)));
  float3 ray_direction = normalize(eyeRay_t - ray_origin);

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

//  output_buffer[launch_index] = make_float4( prd.result, 1.0f);

  // accumulation
  if (frame_number > 1)
  {
      float a = 1.0f / (float)frame_number;
      float b = ((float)frame_number - 1.0f) * a;
      const float3 old_color = make_float3(output_buffer[launch_index]);//read_output();
      output_buffer[launch_index] = make_float4(a * prd.result + b * old_color, 1.0f);//make_float4(old_color);//
  }
  else
  {
      output_buffer[launch_index] = make_float4(prd.result, 1.0f);
      //output_buffer[launch_index] = make_color(prd.result);
  }
//  output_buffer[launch_index] = make_color(accum_buffer[launch_index]);

}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_float4( bad_color, 1.0f);
}
