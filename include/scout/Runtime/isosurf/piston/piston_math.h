
#ifndef PISTON_MATH
#define PISTON_MATH

#include <thrust/detail/config.h>
#include <math.h>

#if THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA

#include "cuda_runtime.h"

#else

typedef struct psfloat3
{
  float x, y, z;
} psfloat3;

typedef struct psfloat4
{
  float x, y, z, w;
} psfloat4;

struct psuint3
{
  unsigned int x, y, z;
};


static __inline__ __host__ __device__ psfloat3 make_psfloat3(float x, float y, float z)
{
  psfloat3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ psfloat4 make_psfloat4(float x, float y, float z, float w)
{
  psfloat4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ psuint3 make_psuint3(unsigned int x, unsigned int y, unsigned int z)
{
  psuint3 t; t.x = x; t.y = y; t.z = z; return t;
}

#endif


static __inline__ __host__ __device__ psfloat3 make_psfloat3(psfloat4 a)
{
  psfloat3 t; t.x = a.x; t.y = a.y; t.z = a.z; return t;
}

static __inline__ __host__ __device__ psfloat4 make_psfloat4(psfloat3 a, float w)
{
  psfloat4 t; t.x = a.x; t.y = a.y; t.z = a.z; t.w = w; return t;
}


inline __host__ __device__ psfloat3 operator+(psfloat3 a, psfloat3 b)
{
    return make_psfloat3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ psfloat4 operator+(psfloat4 a, psfloat4 b)
{
    return make_psfloat4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ psuint3 operator+(psuint3 a, psuint3 b)
{
    return make_psuint3(a.x + b.x, a.y + b.y, a.z + b.z);
}


inline __host__ __device__ psfloat3 operator-(psfloat3 a, psfloat3 b)
{
    return make_psfloat3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ psfloat4 operator-(psfloat4 a, psfloat4 b)
{
    return make_psfloat4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}


inline __host__ __device__ psfloat3 operator*(float b, psfloat3 a)
{
    return make_psfloat3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ psfloat4 operator*(float b, psfloat4 a)
{
    return make_psfloat4(b * a.x, b * a.y, b * a.z, b * a.w);
}


inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ psfloat3 lerp(psfloat3 a, psfloat3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ psfloat4 lerp(psfloat4 a, psfloat4 b, float t)
{
    return a + t*(b-a);
}


inline __host__ __device__ psfloat3 cross(psfloat3 a, psfloat3 b)
{ 
    return make_psfloat3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}


inline __host__ __device__ float dot(psfloat3 a, psfloat3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(psfloat4 a, psfloat4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}


inline __host__ __device__ psfloat3 normalize(psfloat3 v)
{
    return (sqrt(1.0f / dot(v,v)) * v);
}
inline __host__ __device__ psfloat4 normalize(psfloat4 v)
{
    return (sqrt(1.0f / dot(v,v)) * v);
}


inline __host__ __device__ void setMatrix9Columns(float* r, psfloat3 col1, psfloat3 col2, psfloat3 col3)
{
    r[0] = col1.x;  r[3] = col1.y;  r[6] = col1.z;
    r[1] = col2.x;  r[4] = col2.y;  r[7] = col2.z;
    r[2] = col3.x;  r[5] = col3.y;  r[8] = col3.z;
}


inline __host__ __device__ psfloat4 matrix16Mul(float* r, psfloat4 v)
{
    return make_psfloat4(r[0]*v.x + r[1]*v.y + r[2]*v.z + r[3]*v.w, r[4]*v.x + r[5]*v.y + r[6]*v.z +r[7]*v.w, r[8]*v.x + r[9]*v.y + r[10]*v.z + r[11]*v.w, r[12]*v.x + r[13]*v.y + r[14]*v.z + r[15]*v.w);
}
inline __host__ __device__ psfloat3 matrix16Mul(float* r, psfloat3 v)
{
    return make_psfloat3(r[0]*v.x + r[1]*v.y + r[2]*v.z, r[4]*v.x + r[5]*v.y + r[6]*v.z, r[8]*v.x + r[9]*v.y + r[10]*v.z);
}
inline __host__ __device__ psfloat3 matrix9Mul(float* r, psfloat3 v)
{
    return make_psfloat3(r[0]*v.x + r[1]*v.y + r[2]*v.z, r[3]*v.x + r[4]*v.y + r[5]*v.z, r[6]*v.x + r[7]*v.y + r[8]*v.z);
}
inline __host__ __device__ float* matrix16Mul(float* a, float* b)
{
    float* c = new float[16];
    c[0] = a[0]*b[0]+a[1]*b[4]+a[2]*b[8]+a[3]*b[12];
    c[1] = a[0]*b[1]+a[1]*b[5]+a[2]*b[9]+a[3]*b[13];
    c[2] = a[0]*b[2]+a[1]*b[6]+a[2]*b[10]+a[3]*b[14];
    c[3] = a[0]*b[3]+a[1]*b[7]+a[2]*b[11]+a[3]*b[15];

    c[4] = a[4]*b[0]+a[5]*b[4]+a[6]*b[8]+a[7]*b[12];
    c[5] = a[4]*b[1]+a[5]*b[5]+a[6]*b[9]+a[7]*b[13];
    c[6] = a[4]*b[2]+a[5]*b[6]+a[6]*b[10]+a[7]*b[14];
    c[7] = a[4]*b[3]+a[5]*b[7]+a[6]*b[11]+a[7]*b[15];

    c[8] = a[10]*b[8]+a[11]*b[12]+a[8]*b[0]+a[9]*b[4];
    c[9] = a[10]*b[9]+a[11]*b[13]+a[8]*b[1]+a[9]*b[5];
    c[10] = a[10]*b[10]+a[11]*b[14]+a[8]*b[2]+a[9]*b[6];
    c[11] = a[10]*b[11]+a[11]*b[15]+a[8]*b[3]+a[9]*b[7];

    c[12] = a[12]*b[0]+a[13]*b[4]+a[14]*b[8]+a[15]*b[12];
    c[13] = a[12]*b[1]+a[13]*b[5]+a[14]*b[9]+a[15]*b[13];
    c[14] = a[12]*b[2]+a[13]*b[6]+a[14]*b[10]+a[15]*b[14];
    c[15] = a[12]*b[3]+a[13]*b[7]+a[14]*b[11]+a[15]*b[15];

    return c;
}

#endif
