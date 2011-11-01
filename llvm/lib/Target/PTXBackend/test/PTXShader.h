/**
 * @file   PTXShader.h
 * @date   08.08.2009
 * @author Helge Rhodin
 *
 *
 * Copyright (C) 2009, 2010 Saarland University
 *
 * This file is part of llvmptxbackend.
 *
 * llvmptxbackend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * llvmptxbackend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with llvmptxbackend.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef _PTX_SHADER_H
#define _PTX_SHADER_H

#include <math.h>
#include <xmmintrin.h>

//llvm-gcc -m32 -msse -emit-llvm -c PTXTestFunctions.cpp -o PTXTestFunctions.bc && llc -march=ptx -f PTXTestFunctions.bc

/* typedef int v4sf __attribute__ (( vector_size(4*sizeof(float)) )); */

extern float __half2float(short f);

class Vec3f {
public:
  float x,y,z;

  inline Vec3f() : x(0.0), y(0.0), z(0.0) {};
  inline Vec3f(float c) :
          x(c), y(c), z(c)
  {
  }

  inline Vec3f(float _x, float _y, float _z)
    : x(_x),y(_y),z(_z)
  {};
  inline Vec3f &operator=(const Vec3f &b) { x = b.x; y = b.y; z = b.z; return *this;};
};
/*** Multiplications ***/

  /*! multiply vector by float */
inline Vec3f operator*(const float f, const Vec3f &v)
{ return Vec3f(f*v.x, f*v.y, f*v.z); };

  /*! multiply vector by float */
inline Vec3f operator*(const Vec3f &v, const float f)
{ return Vec3f(f*v.x, f*v.y, f*v.z); };

  /*! still undocumented */
inline void operator*=(Vec3f &v, const float f)
{ v.x *= f; v.y*=f; v.z*=f; };

  /*! still undocumented */
inline void operator*=(Vec3f &v, const Vec3f &f)
{ v.x *= f.x; v.y*=f.y; v.z*=f.z; };

  /*! still undocumented */
inline Vec3f operator*(const Vec3f &a, const Vec3f &b)
{ return Vec3f(a.x*b.x, a.y*b.y, a.z*b.z); };

/*** Divisions ***/

  /*! still undocumented */
inline Vec3f operator/(const Vec3f &v, const float f)
{ return (1.f/f)*v; };

  /*! still undocumented */
inline void operator/=(Vec3f &v, const float f)
{ v *= (1.f/f); };


/*** Additions ***/

inline Vec3f operator+(const float f, const Vec3f &v) {
  return Vec3f(f+v.x, f+v.y, f+v.z);
};

inline Vec3f operator+(const Vec3f &v, const float f) {
  return Vec3f(f+v.x, f+v.y, f+v.z);
};

inline Vec3f operator+(const Vec3f &a, const Vec3f &b) {
  return Vec3f(a.x+b.x, a.y+b.y, a.z+b.z);
};

inline void operator+=(Vec3f &v, const Vec3f &f) {
  v.x += f.x; v.y += f.y; v.z += f.z;
};

inline void operator+=(Vec3f &v, const float f) {
  v.x += f; v.y += f; v.z += f;
};

/*** Substractions ***/

inline Vec3f operator-(const float f, const Vec3f &v) {
  return Vec3f(f-v.x, f-v.y, f-v.z);
};

inline Vec3f operator-(const Vec3f &v, const float f) {
  return Vec3f(v.x-f, v.y-f, v.z-f);
};

inline Vec3f operator-(const Vec3f &a, const Vec3f &b) {
  return Vec3f(a.x-b.x, a.y-b.y, a.z-b.z);
};

inline void operator-=(Vec3f &v, const Vec3f &f) {
  v.x -= f.x; v.y -= f.y; v.z -= f.z;
};

inline void operator-=(Vec3f &v, const float f) {
  v.x -= f; v.y -= f; v.z -= f;
};

inline Vec3f operator-(const Vec3f &v) {
  return Vec3f(-v.x,-v.y,-v.z);
};





typedef Vec3f                                         Point;
typedef Vec3f                                         Vector;
typedef Vec3f                                         Color;
typedef Vec3f                                         Normal;

inline float Dot(const Vec3f &a, const Vec3f &b) {
  return a.x*b.x+a.y*b.y+a.z*b.z;
};
inline Vector FaceForward(const Vector &N, const Vector &I) {
  if ((float) N.x*I.x+N.y*I.y+N.z*I.z > 0.0f) {
    return -N;
  }
  return N;
}
inline float Length(const Vec3f &v) {
  return sqrtf(Dot(v,v));
};
inline float LengthSq(const Vec3f &v) {
  return Dot(v,v);
};
inline void Normalize(Vec3f &v) {
  v *= (1.f/Length(v));
};
inline float Mod(float a, float b) {
  // From Texturing & Modeling (a procedural approach) David S. Ebert
  int n = (int)(a/b);
  a -= n*b;
  if (a < 0.f)
    a += b;
  return a;
}



class Vec4f {
public:
  float x,y,z,w;

  inline Vec4f() : x(0.0), y(0.0), z(0.0),w(0.0) {};
  inline Vec4f(float c) :
    x(c), y(c), z(c), w(c)
  {
  }

  inline Vec4f(float _x, float _y, float _z, float _w)
    : x(_x),y(_y),z(_z),w(_w)
  {};
  inline Vec4f(const Vec3f &b, float _w)
    : x(b.x),y(b.y),z(b.z),w(_w)
  {};
  inline Vec4f &operator=(const Vec4f &b) { x = b.x; y = b.y; z = b.z; w = b.w; return *this;};
};

class Matrix4f
{
public:
  float m00, m01, m02, m03;
  float m10, m11, m12, m13;
  float m20, m21, m22, m23;
  float m30, m31, m32, m33;

  inline Matrix4f()
{
             m00 = 1.f;
             m01 = 0.f;
             m02 = 0.f;
             m03 = 0.f;

             m10 = 0.f;
             m11 = 1.f;
             m12 = 0.f;
             m13 = 0.f;

             m20 = 0.f;
             m21 = 0.f;
             m22 = 1.f;
             m23 = 0.f;

             m30 = 0.f;
             m31 = 0.f;
             m32 = 1.f;
             m33 = 0.f;
  }

  inline Matrix4f(float fEntry00, float fEntry01, float fEntry02, float fEntry03,
                  float fEntry10, float fEntry11, float fEntry12, float fEntry13,
                  float fEntry20, float fEntry21, float fEntry22, float fEntry23,
                  float fEntry30, float fEntry31, float fEntry32, float fEntry33)
  {
             m00 = fEntry00;
             m01 = fEntry01;
             m02 = fEntry02;
             m03 = fEntry03;

             m10 = fEntry10;
             m11 = fEntry11;
             m12 = fEntry12;
             m13 = fEntry13;

             m20 = fEntry20;
             m21 = fEntry21;
             m22 = fEntry22;
             m23 = fEntry23;

             m30 = fEntry30;
             m31 = fEntry31;
             m32 = fEntry32;
             m33 = fEntry33;
  };
};


inline Vec4f operator*(const Matrix4f  &m, const Vec3f &v)
{ return Vec4f(m.m00*v.x+m.m01*v.y+m.m02*v.z+m.m03,
               m.m10*v.x+m.m11*v.y+m.m12*v.z+m.m13,
               m.m20*v.x+m.m21*v.y+m.m22*v.z+m.m23,
               m.m30*v.x+m.m31*v.y+m.m32*v.z+m.m33);
};
inline Vec4f operator*(const Matrix4f  &m, const Vec4f &v)
{ return Vec4f(m.m00*v.x+m.m01*v.y+m.m02*v.z+m.m03*v.w,
               m.m10*v.x+m.m11*v.y+m.m12*v.z+m.m13*v.w,
               m.m20*v.x+m.m21*v.y+m.m22*v.z+m.m23*v.w,
               m.m30*v.x+m.m31*v.y+m.m32*v.z+m.m33*v.w);
};
inline Vec4f operator/=(Vec4f v, const float f)
{v.x/=f; v.y/=f; v.z/=f; v.w/=f; return v;
};

//noise stuff
inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
inline float lerp(float t, float a, float b) { return a + t * (b - a); }
inline float grad(int hash, float x, float y, float z) {
//    int h = (hash & 15);                      // CONVERT LO 4 BITS OF HASH CODE
    float u = (hash & 15)<8 ? x : y,            // INTO 12 GRADIENT DIRECTIONS.
            v = (hash & 15)<4 ? y : (hash & 15)==12||(hash & 15)==14 ? x : z;
    return (((hash & 15)&1) == 0 ? u : -u) + (((hash & 15)&2) == 0 ? v : -v);
}
inline float Noise(float x, float y, float z) {
    static int __ptx_global_p[] = { 151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };

    int X = (int)floorf(x) & 255,                  // FIND UNIT CUBE THAT
        Y = (int)floorf(y) & 255,                  // CONTAINS POINT.
        Z = (int)floorf(z) & 255;
    x -= floorf(x);                                // FIND RELATIVE X,Y,Z
    y -= floorf(y);                                // OF POINT IN CUBE.
    z -= floorf(z);
    float u = fade(x),                                // COMPUTE FADE CURVES
            v = fade(y),                                // FOR EACH OF X,Y,Z.
            w = fade(z);
    int A = __ptx_global_p[X  ]+Y, AA = __ptx_global_p[A]+Z, AB = __ptx_global_p[A+1]+Z,      // HASH COORDINATES OF
        B = __ptx_global_p[X+1]+Y, BA = __ptx_global_p[B]+Z, BB = __ptx_global_p[B+1]+Z;      // THE 8 CUBE CORNERS,

    return
lerp(w,
    lerp(v,
        lerp(u, grad(__ptx_global_p[AA  ], x  , y  , z   ), grad(__ptx_global_p[BA  ], x-1, y  , z   )),
        lerp(u, grad(__ptx_global_p[AB  ], x  , y-1, z   ), grad(__ptx_global_p[BB  ], x-1, y-1, z   ))),// FROM  8
    lerp(v,
        lerp(u, grad(__ptx_global_p[AA+1], x  , y  , z-1 ), grad(__ptx_global_p[BA+1], x-1, y  , z-1 )), // OF CUBE
        lerp(u, grad(__ptx_global_p[AB+1], x  , y-1, z-1 ), grad(__ptx_global_p[BB+1], x-1, y-1, z-1 )))
    );
//        return 255;
}

inline float my_grad_2d(unsigned int hash, float x, float y)
{
        unsigned int u = *(unsigned int*) &x;
        unsigned int v = *(unsigned int*) &y;

        unsigned int mask = hash << 31;
        u ^= mask;
        mask = (hash >> 1) << 31;
        v ^= mask;

        float ru = *(float*) &u;
        float rv = *(float*) &v;

        return ru + rv;
}

inline unsigned int hash(unsigned int a)
{
        // noise needs reproducible hashes
        // based on the lower 8 bits of a
        a &= 0xff;

        a -= (a<<6);

        // original:  a ^= (a>>17);
        // since we only had 8 bits,
        // negating suffices
        a  = ~a;

        a -= (a<<9);
        a ^= (a<<4);
        a -= (a<<3);
        a ^= (a<<10);
        a ^= (a>>15);
        return a;
}



inline float my_noise_2d(float x, float y)
{
        float fx = floorf(x);
        float fy = floorf(y);

        int X = (int) fx & 255,
          Y = (int) fy & 255;

        x -= fx;
        y -= fy;

        float u = fade(x),
                  v = fade(y);

        unsigned int A = hash(X  )+Y, AA = hash(A), AB = hash(A+1),
                     B = hash(X+1)+Y, BA = hash(B), BB = hash(B+1);

        return lerp(v, lerp(u, my_grad_2d(hash(AA), x  , y  ),
                           my_grad_2d(hash(BA), x-1, y  )),
                   lerp(u, my_grad_2d(hash(AB), x  , y-1),
                           my_grad_2d(hash(BB), x-1, y-1)));
}


inline float Noise(const Vec3f & pos)
{
  return Noise(pos.x, pos.y, pos.z);
}

inline float Noise(float x, float y)
{
  return my_noise_2d(x, y);
}

#define Sqrt(x) sqrtf(x)
inline float Abs(float x) {
  return x < 0.f ? -x : x;
}

inline Color diffuseComponent(const Vector &L_dir_norm,
                              const Point &P,
                              const Normal &N,
                              const Color &Cl)
{
    float cosLight = Dot(L_dir_norm, N);
    if (cosLight < 0.0)
        return Color(0.0f, 0.0f, 0.0f);
    return Cl*cosLight;
}

inline Vec3f Normalized(const Vec3f &v) {
  Vec3f nv(v);
  Normalize(nv);
  return nv;
};

inline Color specularbrdf(const Vector &L_dir_norm,
                          const Vector &N,
                          const Vector &V,
                          float roughness,
                          float invRoughness)
{
    Vector H = Normalized(L_dir_norm + V);
    float NdotH = Dot(N, H);
    float val = NdotH > 0.0 ? powf(NdotH, invRoughness) : 0.0;
    return Color(val, val, val);
}

inline Color specularComponent(const Vector &L_dir_norm,
                               const Point &P,
                               const Normal &N,
                               const Vector &V,
                               const Color &Cl,
                               float roughness,
                               float invRoughness)
{
    return Cl * specularbrdf(L_dir_norm, N, V, roughness, invRoughness);
}

inline float Clamp(float val, float low, float high) {
  return val < low ? low : (val > high ? high : val);
}

inline float Floor(float x) {
  return floorf(x);
}

inline float Step(float min, float value) {
  return value < min ? 0.0f : 1.0f;
}

inline float Mix(float x, float y, float alpha) {
  return x*(1.0f - alpha) + y*alpha;
}

inline Vec3f Mix(const Vec3f &x, const Vec3f &y, float alpha) {
  return x*(1.0f - alpha) + y*alpha;
}

inline float SmoothStep(float min, float max, float value) {
  if (value < min) return 0.0f;
  if (value >= max) return 1.0f;

  float v = (value - min) / (max - min);
  return (-2.0f * v + 3.0f) * v * v;
}

inline Vector Reflect(const Vector &I, const Vector &N) {
  return I - 2.f * Dot(I, N) * N;
}

inline Vec3f Cross(const Vec3f &a, const Vec3f &b) {
  return Vec3f(a.y*b.z-a.z*b.y,
               a.z*b.x-a.x*b.z,
               a.x*b.y-a.y*b.x);
};

inline float Max(float a, float b) {
  return a > b ? a : b;
}

inline float Min(float a, float b) {
  return a < b ? a : b;
}

inline void GetFrame(const Vec3f &n, Vec3f &x, Vec3f &y) {
  Vec3f u(0);
  float n0 = n.x * n.x;
  float n1 = n.y * n.y;
  float n2 = n.z * n.z;

  if (n1 < n0) {
    if (n2 < n1) {
      u.z = 1;
    } else {
      u.y = 1;
    }
  } else if (n2 < n0) {
    u.z = 1;
  } else {
    u.x = 1;
  }

  x = Cross(u,n);
  y = Cross(n,x);

  Normalize(x);
  Normalize(y);
}

inline float clamp2One(float f)
{return (f > 1.f ? 1.f : (f < 0.f ? 0.f : f));}



typedef struct
{
  float a;
} ShadowTexture;

typedef struct
{
  float a,b,c,d;
} __ptx_global_Color32Bit;
typedef struct
{
  short a,b,c,d;
} __ptx_global_Color16Bit;
typedef struct
{
  char a,b,c,d;
} __ptx_global_Color8Bit;
//#define COLOR __ptx_global_Color32Bit
#define COLOR_IN __ptx_global_Color16Bit
#define COLOR_OUT __ptx_global_Color8Bit
#define SHADOW ShadowTexture
#define DeviceBitPOINTER int

typedef struct
{
  Point position;
  Vector direction;
  float spotlight_outerAngle;
  float spotlight_innerAngle;
  Color color_diffuse;
  Vector attenuation;
  float shadowFarClip;
  Matrix4f shadowTransform;
  DeviceBitPOINTER shadowTexture;
} PointLight_Host;

typedef struct
{
  Point position;
  Vector direction;
  float spotlight_outerAngle;
  float spotlight_innerAngle;
  Color color_diffuse;
  Vector attenuation;
  float shadowFarClip;
  Matrix4f shadowTransform;
  SHADOW* shadowTexture;
} PointLight;

typedef struct
{
  Point origin;
  float time;
  Matrix4f viewTransformInv;
  float farClipDistance;
  Vector farCorner;
  //  float shadowFarClip;
  /*
  Vector dir_topleft;
  Vector dir_topright;
  Vector dir_bottomleft;
  Vector dir_bottomright;
  */
  int width, height;
  COLOR_IN* texture0, *texture1;
  COLOR_OUT *textureOut;
  int lights_n;
} ShadingData;

typedef struct
{
  Point origin;
  float time;
  Matrix4f viewTransformInv;
  float farClipDistance;
  Vector farCorner;
  //  float shadowFarClip;
  /*
  Vector dir_topleft;
  Vector dir_topright;
  Vector dir_bottomleft;
  Vector dir_bottomright;
  */
  int width, height;
  DeviceBitPOINTER texture0, texture1;
  DeviceBitPOINTER textureOut;
  int lights_n;
} ShadingData_HOST;

#define THREAD_WIDTH_X 1
#define THREAD_WIDTH_Y 1

#define CTA_WIDTH 32
#define CTA_HEIGHT 2

//special register dummies
unsigned short __ptx_sreg_tid_x;
unsigned short __ptx_sreg_tid_y;
unsigned short __ptx_sreg_tid_z;

unsigned short __ptx_sreg_ntid_x;
unsigned short __ptx_sreg_ntid_y;
unsigned short __ptx_sreg_ntid_z;

unsigned short __ptx_sreg_ctaid_x;
unsigned short __ptx_sreg_ctaid_y;
unsigned short __ptx_sreg_ctaid_z;

unsigned short __ptx_sreg_nctaid_x;
unsigned short __ptx_sreg_nctaid_y;
unsigned short __ptx_sreg_nctaid_z;

unsigned short __ptx_sreg_gridid;
unsigned short __ptx_sreg_clock;

inline COLOR_IN* getPixelPtr(COLOR_IN *__ptx_global_data, int xpos, int ypos, int width, int height)
{
  return __ptx_global_data + ypos*width+xpos;
}

//void __attribute__((noinline))
inline void getPixelData(float &x, float &y, float &z, float &w,  COLOR_IN *pixel)
{
  x = (float)(pixel)->a;
  y = (float)(pixel)->b;
  z = (float)(pixel)->c;
  w = (float)(pixel)->d;
}

inline COLOR_IN* getPixel(COLOR_IN *__ptx_global_data, int x, int y, int width, int height)
{
    return (__ptx_global_data + y*width+x);
}

inline COLOR_OUT* getPixel(COLOR_OUT *__ptx_global_data, int x, int y, int width, int height)
{
    return (__ptx_global_data + y*width+x);
}


inline SHADOW* getPixel(SHADOW *__ptx_global_data, int x, int y, int width, int height)
{
    return (__ptx_global_data + y*width+x);
}

inline int rgbaToInt(float r, float g, float b, float a)
{
  return (int(a)<<24 | int(r)<<16) | (int(g)<<8) | int(b);
}

inline int getAlpha(int rgb)
{return (rgb&0xff000000)>>24;}

inline int getRed(int rgb)
{return (rgb&0x00ff0000)>>16;}

inline int getGreen(int rgb)
{return (rgb&0x0000ff00)>>8;}

inline int getBlue(int rgb)
{return rgb&0x000000ff;}


//texture references
unsigned int __ptx_texture_input0;
unsigned int __ptx_texture_input1;
unsigned int __ptx_texture_shadow;

//defined in llvm backend
/* extern v4sf __ptx_tex1D(float* ptr, float coordinate); */



/* inline int isShadow( */
/*        SHADOW* shadowMap, */
/*        Vec3f viewPos, */
/*        Matrix4f* invView, */
/*        Matrix4f* shadowViewProj, */
/*        float shadowFarClip, */
/*        float distanceFromLight, */
/*        Vec3f lightToPointDirNorm, */
/*        Vec3f lightDir */
/*        ) */
/* { */

/*   Vec4f worldPos = *invView * viewPos; */
/*   worldPos.w = 1; */
/*   Vec4f shadowProjPos = *shadowViewProj * worldPos; */
/*   shadowProjPos /= shadowProjPos.w; */


/*   float u = shadowProjPos.x; */
/*   float v = shadowProjPos.y; */
/*   float scalefix = (u*u+v*v)*0.7071067f*distanceFromLight; */

/*   Normalize(lightDir); */

/*   scalefix = (distanceFromLight) * Dot(lightDir,-lightToPointDirNorm); */

/*   float  xf = (shadowProjPos.x*511.f/scalefix); */
/*   float  yf = (shadowProjPos.y*511.f/scalefix); */

/*   int xx = (shadowProjPos.x*511.f/scalefix); */
/*   int yy = (shadowProjPos.y*511.f/scalefix); */

/*   //  float xWeigth = xf - xx; */
/*   //  float yWeigth = yf - yy; */

/*   if(xx < 1 || xx > 510 || yy<1 || yy>510) */
/*       return 0; */

/*   //SHADOW* shadowPixel  = getPixel(shadowMap,xx,yy,512,512); */
/*   //float shadowDepth = (float) shadowPixel->a; */

 /*   */
/*   SHADOW* shadowPixel  = getPixel(shadowMap,xx,yy,512,512); */
/*   float shadowDepth00 = (float) shadowPixel->a; */

/*   shadowPixel  = getPixel(shadowMap,xx+1,yy,512,512); */
/*   float shadowDepth10 = (float) shadowPixel->a; */

/*   shadowPixel  = getPixel(shadowMap,xx,yy+1,512,512); */
/*   float shadowDepth01 = (float) shadowPixel->a; */

/*   shadowPixel  = getPixel(shadowMap,xx+1,yy+1,512,512); */
/*   float shadowDepth11 = (float) shadowPixel->a; */

/*   float tmp1 = shadowDepth00 * (1 - xWeigth) + shadowDepth10 * xWeigth; */
/*   float tmp2 = shadowDepth01 * (1 - xWeigth) + shadowDepth11 * xWeigth; */

/*   float shadowDepth = tmp1 * (1 - yWeigth) + tmp2 * yWeigth; */
/*   */


/*   */
/*   SHADOW* shadowPixel  = getPixel(shadowMap,xx,yy,512,512); */
/*   float shadowDepth = (float) shadowPixel->a; */
/*   */

/*   //texture fetches */

/*   v4sf tmp_msse = __ptx_tex1D((float*)(&__ptx_texture_shadow), */
/*                                (yy*512+xx)); */
/*   float shadowDepth = ((float*)&tmp_msse)[0]; */


/*  //   shadowDepth = (shadowDepth+1.f)*0.5f; */
/*    float shadowDistance = shadowDepth * shadowFarClip * 0.99731f; */
/*    //   return shadowDepth; */

/*    return shadowDistance - distanceFromLight -0.1f < 0.f; //-0.1f */
/* } */





/* inline int isShadowOld( */
/*        SHADOW* shadowMap, */
/*        Vec3f viewPos, */
/*        Matrix4f* invView, */
/*        Matrix4f* shadowViewProj, */
/*        float shadowFarClip, */
/*        float distanceFromLight, */
/*        Vec3f lightToPointDirNorm, */
/*        Vec3f lightDir */
/*        ) */
/* { */

/*   Vec4f worldPos = (*invView) * viewPos; */
/*   worldPos.w = 1; */
/*   Vec4f shadowProjPos = (*shadowViewProj) * worldPos; */
/*   shadowProjPos /= shadowProjPos.w; */

/*   int xx = (shadowProjPos.x*511.f)+256; */
/*   int yy = (shadowProjPos.y*511.f)+256; */

/*   //  if(xx < 0 || xx > 512 || yy<0 || yy>511) */
/*   //     return 0; */

/*     ///    xx = ((xx) % 512);//(xx<0 ? 0 : (xx>511 ? 511 : xx)); */
/*     ///    yy = ((yy) % 512);//(yy<0 ? 0 : (yy>511 ? 511 : yy)); */

/*     ///    xx = 511 - xx; */

/*     v4sf tmp_msse = __ptx_tex1D((float*)(&__ptx_texture_shadow), */
/*                                (yy*512+xx)); */
/*   float shadowDepth = ((float*)&tmp_msse)[0]; */

/*   //    SHADOW* shadowPixel  = getPixel(shadowMap,xx,yy,512,512); */
/*   //   float shadowDepth = (float) shadowPixel->a; */

/*    float shadowDistance = shadowDepth * shadowFarClip; */

/*    return shadowDistance - distanceFromLight +0.0f < -99999.f; //0.f // 0.1f */
/* } */

#endif
