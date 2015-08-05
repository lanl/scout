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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include "commonStructs.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};


struct PerRayData_occlusion
{
  float occlusion;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtBuffer<BasicLight> lights;

rtDeclareVariable(rtObject,    top_object, , );
rtDeclareVariable(rtObject,     top_shadower, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(unsigned int, ao_ray_type, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(float,       occlusion_distance, , );
rtDeclareVariable(int,         sqrt_occlusion_samples, , );
//rtBuffer<unsigned int, 2>      rnd_seeds;


// Textures
rtTextureSampler<float,  3> noise_texture;
rtTextureSampler<float4, 1> color_ramp_texture;


// Material properties
//rtDeclareVariable(float, reflectivity, , );
rtDeclareVariable(float, specular_exp, , );

// Attributes
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

// Ray tracing
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );
rtDeclareVariable(PerRayData_occlusion, prd_occlusion, rtPayload, );

rtDeclareVariable(optix::Matrix3x3,        normalMatrix, , );
rtDeclareVariable(optix::Matrix4x4,        modelviewMatrix, , );
rtDeclareVariable(optix::Matrix4x4,        invModelviewMatrix, , );

rtDeclareVariable(float3,       jitter, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(int,        frame, , );
rtDeclareVariable(unsigned int, subframe_idx, rtSubframeIndex, );

// Create ONB from normal.  Resulting W is parallel to normal
static
__host__ __device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W )
{
  using namespace optix;

  W = normalize( n );
  U = cross( W, optix::make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( fabs( U.x ) < 0.001f && fabs( U.y ) < 0.001f && fabs( U.z ) < 0.001f  )
    U = cross( W, make_float3( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( W, U );
}

__device__ inline float3 phongModel(float3 pos_object, float3 dir, float3 world_normal, float3 light_world, float ao) {
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    float3 color = make_float3(1.0f, 1.0f, 1.0f);
    result += 0.2 * ao * color;
    float4 pos_world = modelviewMatrix * make_float4(pos_object.x, pos_object.y, pos_object.z, 1.0);
    float3 L_world = normalize(light_world - make_float3(pos_world));
    float NdotL = dot(world_normal, L_world);
    if (NdotL > 0.0f)
    {
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.0f);

        float4 light_object =  invModelviewMatrix * make_float4(
                    light_world.x, light_world.y, light_world.z, 1.0);

        float2 sample = optix::square_to_disk(make_float2(jitter.x, jitter.y));
        //sample = make_float2(0.0f, 0.0f);
        float3 U, V, W;
        create_onb(make_float3(light_object), U, V, W);
        light_object += make_float4(50.0f * (sample.x * U + sample.y * V), 0.0f);

        float3 L_object = make_float3(light_object) - pos_object;
        // Ray shadow_ray(pos_object, normalize(L_object), shadow_ray_type,
        //                          scene_epsilon, length(L_object));
        // rtTrace(top_shadower, shadow_ray, shadow_prd);
        if(fmaxf(shadow_prd.attenuation) > 0.5) {

          result += 0.6 * color * NdotL;

          // Specular lighting.
          float3 H = normalize(L_world - dir);
          float NdotH = dot(world_normal, H);
          if (NdotH > 0.0f)
          {
            result += 0.2 * color * pow(NdotH, specular_exp);
          }
        }
    }
    return result;
}

static __device__ __inline__ float3 LightShader(float3 point, float3 dir, float3 color, float ao)
{
  float3 result = make_float3(0.0f, 0.0f, 0.0f);

  unsigned int num_lights = lights.size();
  for (unsigned int i = 0; i < num_lights; i++)
  {
    BasicLight light = lights[i];
    
    float3 world_normal = normalize( normalMatrix * normalize(shading_normal));

    if(world_normal.z < 0)
        world_normal = - world_normal;

    result += phongModel(point, dir, world_normal, light.pos, ao);
  }
  return result;
}

static __host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}


// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

__device__ float AmbientOcclusion(float3 phit)
{
    optix::Onb onb(normalize(shading_normal));

    unsigned int seed = rot_seed(launch_index.x, 2);//frame + subframe_idx);

    float       result           = 0.0f;
    const float inv_sqrt_samples = 1.0f / float(sqrt_occlusion_samples);
    for( int i=0; i<sqrt_occlusion_samples; ++i ) {
      for( int j=0; j<sqrt_occlusion_samples; ++j ) {

        PerRayData_occlusion prd_occ;
        prd_occ.occlusion = 0.0f;

        // Stratify samples via simple jitterring
        float u1 = (float(i) + rnd( seed ) )*inv_sqrt_samples;
        float u2 = (float(j) + rnd( seed ) )*inv_sqrt_samples;

        float3 dir;
        optix::cosine_sample_hemisphere( u1, u2, dir );
        onb.inverse_transform( dir );

        optix::Ray occlusion_ray = optix::make_Ray( phit, dir, ao_ray_type, scene_epsilon,
                                                    occlusion_distance );
        rtTrace( top_object, occlusion_ray, prd_occ );

        result += 1.0f-prd_occ.occlusion;
      }
    }

    result /= (float)(sqrt_occlusion_samples*sqrt_occlusion_samples);

    return result;
}

RT_PROGRAM void closest_hit_surface()
{
  const float3 pos = ray.origin + ray.direction * isect_dist;

  float ao = 1.0;//AmbientOcclusion(pos);

//  pos_world = make_float4(pos_world.x / pos_world.w, pos_world.y / pos_world.w, pos_world.z / pos_world.w, 1.0f);
//  pos_world /= pos_world.w;

  //TODO: Fix the problem that red and blue are reversed
  float3 color = make_float3(1,1,1);//tex1D(color_ramp_texture, u));

  prd_radiance.result = LightShader(pos, ray.direction, color, ao);
}

RT_PROGRAM void any_hit_shadow()
{
  prd_shadow.attenuation = make_float3(0.0f);

  rtTerminateRay();
}

RT_PROGRAM void any_hit_occlusion()
{
  prd_occlusion.occlusion = 1.0f;

  rtTerminateRay();
}
