/**
 * @file   PTXTestFunctions.cpp
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
#include<math.h>
#include <xmmintrin.h>
#include "PTXShader.h"

#define ARRAY_N 64

//data structures
typedef struct
{
  float f;
  char c;
  int i;
  char cc;
} DataStructInternal1;

typedef struct
{
  float f;
  DataStructInternal1 s;
  DataStructInternal1 sa[3];
  int i;
} DataStructInternal0;

typedef struct
{
  float fa[ARRAY_N];
  float f;
  int i;
  unsigned int u;
  char c;
  char ca[19];
  int ia[ARRAY_N];
  DataStructInternal0 s;
  double d;
  short half;
} DataStruct;

/////////////////////////////////////////////////////////////
////////////////////////// test cases ///////////////////////
/////////////////////////////////////////////////////////////

extern "C" void test_GetElementPointer_constant(DataStruct* data)
{
  data->i=10;
  data->f=0.3;
  data->fa[3]=-4.1;
}

extern float __half2float(short x);

extern "C" void test_calculate(DataStruct* data)
{
  data->ia[0] = data->i + data->i;
  data->ia[1] = data->i - data->i;
  data->ia[2] = data->i * data->i;
  data->ia[3] = data->i / data->i;
  data->ia[4] = (unsigned)data->i << data->i;
  data->ia[5] = (unsigned)data->i >> data->i;
  data->ia[6] = data->i << data->i;
  data->ia[7] = data->i >> data->i;
  data->ia[8] = data->i % data->i;
  data->ia[9] = data->i & data->i;
  data->ia[10] = data->i | data->i;
  data->ia[11] = data->i ^ data->i;

  data->ia[20] = data->f + data->i;
  data->ia[21] = data->f - data->i;
  data->ia[22] = data->f * data->i;
  data->ia[23] = data->f / data->i;
  data->ia[24] = (unsigned)data->f << data->i;
  data->ia[25] = (unsigned)data->f >> data->i;


  data->fa[0] = data->f + data->f;
  data->fa[1] = data->f - data->f;
  data->fa[2] = data->f * data->f;
  data->fa[3] = data->f / data->f;

  data->fa[20] = data->f + data->i;
  data->fa[21] = data->f - data->i;
  data->fa[22] = data->f * data->i;
  data->fa[23] = data->f / data->i;
  data->fa[24] = (unsigned)data->f << data->i;
  data->fa[25] = (unsigned)data->f >> data->i;

  data->f = __half2float(data->half);
}

extern "C" void test_GetElementPointer_dyn(DataStruct* data)
{
  data->fa[data->u]= -7.1;
  data->fa[data->i]=  6.3;
}

extern "C" void test_branch_simple(DataStruct* data)
{
  data->i=10; //not run => fails!
  if(data->f<0)
    data->f = -data->f*2.3f;
  else
    data->f = data->f*2.1f;
}

extern "C" void test_branch_simplePHI(DataStruct* data)
{
  float tmp;
  if(data->f<0)
    tmp = -data->f*2;
  else
    tmp = data->f*2;
  data->f = tmp;
}

extern "C" void test_branch_loop(DataStruct* data)
{
  float tmp = 0;
  for(int i=0; i<data->i; i++)
    tmp += i;
  data->f = tmp;
}

//float floorf(float f);

extern "C" void test_math(DataStruct* data)
{
  float f = data->f;
  float fi = data->i;
  data->fa[0] = expf(f);
  data->fa[1] = logf(f);
  data->fa[2] = exp2f(f);
  data->fa[3] = log2f(f);
  data->fa[4] = sinf(f);
  data->fa[5] = cosf(f);
  data->fa[6] = sqrtf(f);
  data->fa[7] = tanf(f);
//  data->fa[8] = rsqrtf(f);
//  data->fa[9] = rcpf(f); // 1/x
  data->fa[10] = floorf(f);
  data->fa[11] = atanf(f);
  data->fa[12] = powf(f,fi);
//  data->fa[13] = powi(f,data->i);


}

extern "C" void test_signedOperands(DataStruct* data)
{
  int i = data->i;
  int j = i*2;
  data->ia[0] = j % i;
  data->ia[1] = j >> i;
  data->ia[2] = -7 / i;
  data->ia[3] = i - 5;
  data->ia[4] = data->f;
  data->i = (char)data->ia[3];
  data->f = data->ia[1];
}

extern "C" void test_constantOperands(DataStruct* data)
{
  float f = 0;
  data->ia[0] = 14 + data->i;
  data->ia[1] = char(3);
  data->ia[2] = char(-4);
  data->ia[3] = 7;

  data->fa[0] = data->f * 312.13f;
  data->fa[1] = data->f / -31213.123f;
  data->fa[2] = 234534 -3123.1123;
  data->fa[3] = expf(10.23);
  data->fa[4] = cosf(1.33);
  data->fa[5] = f;
}
extern "C" void test_binaryInst(DataStruct* data)
{
  int i = data->i;
  float f = data->f;
  data->ia[0] = data->i + data->i;
}

extern "C" void test_selp(DataStruct* data)
{
  int x = data->i == 0;
  data->fa[0] = (x? 100 : 12);
  data->fa[1] = (data->i != 0 ? 100 : 12);
}

extern "C" void test_branch_loop_semihard(DataStruct* data)
{
  int tmp = 0;
  int is = data->i;
  for(unsigned int i=0; i<is; i++)
  {
    tmp += i;
    if(i>10)
      break;
    tmp += i;
  }
  data->f = tmp;
}

extern "C" void test_branch_loop_hard(DataStruct* data)
{
  int tmp = 0;
  int is = data->i;
  unsigned int us = data->u;
  for(unsigned int i=0; i<is; i++)
  {
    tmp += i;
     if(i==5)
       continue;
     for(int k=0; k<us; k++)
     {
       if(i+k>15)
         return;
     }
     if(i>10)
       break;
    tmp += i;
  }
  data->f = tmp;
  data->i = 234234; //not run => fails!
}

extern "C" void test_GetElementPointer_complicated(DataStruct* data)
{
  data->fa[0] = data->s.s.f;
  data->fa[1] = data->s.sa[2].f;
  data->fa[2] = data->s.sa[data->i].f;
}

extern "C" void test_alloca(DataStruct* data)
{
  DataStruct data_local;
  int i = data->i;
  int f = data->f;
  data_local.fa[i] = f;
  data_local.fa[i*2] = f*2;
  float tmp0 = data_local.fa[i];
  float tmp1 = data_local.fa[i+1];
  data->fa[0] = tmp0;
  data->fa[1] = tmp1;
}

extern "C" void test_alloca_complicated(DataStruct* data)
{
  DataStruct data_local;
  int i = data->i;
  data_local.s.s.f = data->f;
  data_local.s.sa[2].f = data->f*2;
  data_local.s.sa[i].f = data->f*3;

  data->fa[0] = data_local.s.s.f;
  data->fa[1] = data_local.s.sa[2].f;
  data->fa[2] = data_local.s.sa[i].f;
}

extern "C" void test_specialRegisters_x(DataStruct* data)
{
  int i =  __ptx_sreg_tid_x
        + (__ptx_sreg_tid_y*__ptx_sreg_ntid_x)
        + (__ptx_sreg_tid_z*__ptx_sreg_ntid_x*__ptx_sreg_ntid_y);
  if(__ptx_sreg_ctaid_x>0)
    data->fa[i] = 0;
  else
    data->ia[i] = 1;
}

extern "C" void test_specialRegisters_y(DataStruct* data)
{
  int i =  __ptx_sreg_tid_x
        + (__ptx_sreg_tid_y*__ptx_sreg_ntid_x)
        + (__ptx_sreg_tid_z*__ptx_sreg_ntid_x*__ptx_sreg_ntid_y);
  if(__ptx_sreg_ctaid_y==0)
    data->fa[i] = 0;
  else if(__ptx_sreg_ctaid_y==1)
    data->ia[i] = 1;
}

DataStruct data_global;
int i_global = 0;
float f_global = 3.3;

int ia_global[10] = {217871,21123,2,3,4,5,6,7,8,9};
char ca_global[10] = {0,1,2,3,4,5,6,7,8,9};
float fa_global[10] =  {1.324,2.23,3.32,4.2342,5234.3,6.23,7234234234.1,8234,9324,2342410};

DataStruct __ptx_constant_data_global;
unsigned int __ptx_texture_dataf;
unsigned int __ptx_texture_datai;

extern "C" void test_globalVariables(DataStruct* data)
{
  data->i = i_global;
  data->f = f_global;

  data->ia[0] = ia_global[0];
  data->ia[1] = ia_global[1];
  data->ia[2] = ia_global[2];
  data->ia[3] = ia_global[3];
  data->ia[4] = ia_global[4];
  data->ia[5] = ia_global[5];
  data->ia[6] = ia_global[6];
  data->ia[7] = ia_global[7];
  data->ia[8] = ia_global[8];
  data->ia[9] = ia_global[9];
  data->ia[10] = ca_global[0];
  data->ia[11] = ca_global[1];
  data->ia[12] = ca_global[2];
  data->ia[13] = ca_global[3];
  data->ia[14] = ca_global[4];
  data->ia[15] = ca_global[5];
  data->ia[16] = ca_global[6];
  data->ia[17] = ca_global[7];
  data->ia[18] = ca_global[8];
  data->ia[19] = ca_global[9];

  data->fa[0] = fa_global[0];
  data->fa[1] = fa_global[1];
  data->fa[2] = fa_global[2];
  data->fa[3] = fa_global[3];
  data->fa[4] = fa_global[4];
  data->fa[5] = fa_global[5];
  data->fa[6] = fa_global[6];
  data->fa[7] = fa_global[7];
  data->fa[8] = fa_global[8];
  data->fa[9] = fa_global[9];
}
//////////////////////////////////////////////////
////////////// Ralfs tests ///////////////////////
// replace "return", "data->f ="
// replace "DataStruct* data", "DataStruct* data"
// replace "a", "data->f[0]"
// replace "b", "data->f[1]"
// replace "float test", "void test"

extern "C" void test_phi_scalar(DataStruct* data) {
  float a = data->fa[0];
  float b = data->fa[1];
    float x = a + b;
    float y = x * x - b;
    float z;

    if (x<y) {
        z = a+x;
    } else {
        z = a*a;
    }

    z = z+x;
    z = y-z;

    data->f = z;
}

extern "C" void test_phi2_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z;
    float r;

    if (x<y) {
        z = data->fa[0]+x;
        r = x*x;
    } else {
        z = data->fa[0]*data->fa[0];
        r = x-data->fa[0];
    }

    z = z+x;
    z = y-z;

    data->f = z * r;
}

extern "C" void test_phi3_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z;
    float r;

    if (x<y) {
        z = data->fa[0]+x;
        r = x*x;
    } else if (x>y) {
        z = data->fa[0]*data->fa[0];
        r = x-data->fa[0];
    } else {
        z = y-data->fa[0];
        r = y+data->fa[0];
    }

    z = z+x;
    z = y-z;

    data->f = z * r;
}

extern "C" void test_phi4_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z;
    float r;

    if (x>y) {
        z = data->fa[0]+x;
        r = x*x;
    } else if (y>x) {
        z = data->fa[0]*data->fa[0];
        if (z != y) r = x-data->fa[0];
        else r = x+data->fa[0];
    } else {
        z = y-data->fa[0];
        r = y+data->fa[0];
    }

    z = z+x;
    z = y-z;

    data->f = z * r;
}

extern "C" void test_phi5_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z = y;
    float r = 3;

    if (x<y) {
        z = data->fa[0]+x;
        r += z*z;
        float f = z-r;
        z -= f;
    }

    z = z+x;

    data->f = z * r;
}

extern "C" void test_phi6_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z = y;
    float r = 3;

    if ((data->fa[0] <= z && data->fa[1] > 4) || z>y) {
        z = data->fa[0]+x;
        r += z*z;
        float f = z-r;
        z -= f;
    }

    z = z+x;

    data->f = z * r;
}

extern "C" void test_phi7_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z;

    if (x<y) {
        z = data->fa[0]+x;
    } else {
        z = data->fa[0]*data->fa[0];
    }

    if (z > data->fa[0] && data->fa[0] < data->fa[1]) {
        z++;
    }

    z = z+x;
    z = y-z;

    data->f = z;
}

extern "C" void test_phi8_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z = y;

    if (y > data->fa[1]) {
        z *= z;

        if (x<y) {
            z = data->fa[0]+x;
        } else {
            z = data->fa[0]*data->fa[0];
        }

        z -= data->fa[0];

        if (z > data->fa[0] && data->fa[0] < data->fa[1]) {
            z++;
        }

        z += data->fa[1];
    }

    z = z+x;
    z = y-z;

    data->f = z;
}

extern "C" void test_phi9_scalar(DataStruct* data) {
    float x = data->fa[0] + data->fa[1];
    float y = x * x - data->fa[1];
    float z = y;

    if (data->fa[0] < data->fa[1]) {
        z += data->fa[0];
    } else if (data->fa[1] < data->fa[0]) {
        z += data->fa[0]*data->fa[0];
    }

    z = z+x;
    z = y-z;

    data->f = z;
}

extern "C" void test_loop13_scalar(DataStruct* data)
{
  float a = data->fa[0];
  float b = data->fa[1];
  float x = a + b;
  float y = x * x - b;
  float z = y;
  for (int i=0; i<1000; ++i) {
    z += a;
    if (z / a < x) z += a;
    else {
      z -= b;
      if (a > b) z -= b;
      else {
        z *= z-y;
        if (b == a) {
          for (int j=0; j<200; ++j) {
            if (i == j) z *= z;
            z += 13.2f;
          }
          z = a+3;
        } else {
          ++z;
        }
      }
      for (int j=0; j<100; ++j) {
        if (i < j) z += a;
        else z -= 13.2f;
      }
    }
  }
  z = z-y;
  data->f = z;
}

extern "C" void test_loop23_scalar(DataStruct* data) {
  float a = data->fa[0];
  float b = data->fa[1];
  float x = a + b;
  float y = x * x - b;
  float z = y;
  for (int i=0; i<1000; ++i) {
    z += a;
    if (z / a < x) break;
    else {
      z -= b;
      if (a > b) {
        for (int j=3; j<4500; ++j) {
          if (i == j) z /= -0.12f;
          if (z < -100.f) break;
          if (z < 0.f) {data->f = z; return;}
        }
        continue;
      }
      else {
        z *= z-y;
        if (b == a) {
          {data->f = z; return;}
        } else {
          ++z;
          break;
        }
      }
    }
  }
  z = z-y;
  data->f = z;
}

extern "C" void test_loopbad_scalar(DataStruct* data) {
  float a = data->fa[0];
  float b = data->fa[1];
  float x = a + b;
  float y = x * x - b;
  float z = y;
  //the 'butterfly' / 'lung'
  for (int i=0; i<100; ++i) {
    z += a;
    if (z > a) {
      a *= a;
      if (a > b) break;
    } else {
      a += a;
      if (a < b) break;
    }
    z += b;
  }
  z = z+x;
  z = y-z;
  data->f = z;
}

void callFun1(){}
int callFun2(){return 5;}
float callFun3(int i){return i*i;}

extern "C" void test_call(DataStruct* data)
{
  callFun1();
  data->ia[0] = callFun2();
  data->ia[1] = callFun3(data->i);
}

extern "C" void test_dualArgument(DataStruct* data0, DataStruct* data1)
{
  data0->i = 1;
  data1->i = 2;
  data0->fa[3] = 1.111;
  data1->fa[3] = 2.222;
}

extern "C" void test_vector(DataStruct* data0)
{
  //  __attribute__((aligned(16)) float arr[4];
  __m128 test = *(__m128*)data0->fa;
  float f = ((float*)&test)[0];
  float f2 = ((float*)&test)[1];
  float f3 = ((float*)&test)[2];
  data0->f = f+f2+f3;
}

extern "C" void test_constantMemory(DataStruct* data)
{
  data->f = __ptx_constant_data_global.f;
  __m128 test = *(__m128*)__ptx_constant_data_global.fa;
  float f = ((float*)&test)[0];
  float f2 = ((float*)&test)[1];
  float f3 = ((float*)&test)[2];
  data->fa[0] = f+f2+f3;

}

// extern v4sf __ptx_tex1D(float* ptr, float coordinate);//{return *(__m128*)ptr;}
// extern v4sf __ptx_tex2D(float* ptr, __m128 coordinates);//{return *(__m128*)ptr;}
// extern v4sf __ptx_tex3D(float* ptr, __m128 coordinates);//{return *(__m128*)ptr;}

/* m128i not supported??
extern __m128i __ptx_tex1D(int* ptr, float coordinate);//{return *(__m128*)ptr;}
extern __m128i __ptx_tex2D(int* ptr, __m64 coordinates);//{return *(__m128*)ptr;}
extern __m128i __ptx_tex3D(int* ptr, __m128 coordinates);//{return *(__m128*)ptr;}
*/
// TODO is this test correct? improve further..
// extern "C" void test_textureFetches(DataStruct* data)
// {

//   v4sf test = __ptx_tex1D((float*)(&__ptx_texture_dataf),0);
//   float f = ((float*)&test)[0];
//   float f2 = ((float*)&test)[1];
//   float f3 = ((float*)&test)[2];
//   data->fa[0] = f+f2+f3;

//   //  __m128 var = _mm_set_ps()
//   test = __ptx_tex2D((float*)(&__ptx_texture_dataf),_mm_set_ps(0,0,0,0));
//   f = ((float*)&test)[0];
//   f2 = ((float*)&test)[1];
//   f3 = ((float*)&test)[2];
//   data->fa[1] = f+f2+f3;

//   test = __ptx_tex3D((float*)(&__ptx_texture_dataf),_mm_set_ps(0,0,0,0));
//   f = ((float*)&test)[0];
//   f2 = ((float*)&test)[1];
//   f3 = ((float*)&test)[2];
//   data->fa[2] = f+f2+f3;
// }

const int warp_width = 16;
const int warp_width_h = warp_width/2;

short __ptx_shared_sharedData[(warp_width*2)*4];

// TODO: is test exhaustive
extern "C" void test_sharedMemory(DataStruct* data)
{
  //write to shared memory (no bank conflicts)
  float u = data->fa[__ptx_sreg_tid_x];
  // (__ptx_sreg_tid_x%2) == 0 => value a else b
  // __ptx_sreg_tid_x/16 == 0 => halfwarp 1 else halfwarp 2
  // __ptx_sreg_tid_x/2 index of a(or b): a_0,a_1.... and b_0, b_1 ...
  // put values a successively into array (because factor __ptx_sreg_tid_x%2 == 0)
  // put values b in the following order: b_8,b_9...b_15,b_0,b_1,...b_7
  // behind a values => initial offset +16
  // put first 8 values with offset 8(+initial offset) => offset+8 (done by halfwarp 1: threds 1,3,..15)
  // put last 8 values with offset 0(+initial offset), halfwarp 2 => (1 - __ptx_sreg_tid_x/16)==0 => no additional offset

  int index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2; //(a or b?)*
  *(float*)&(__ptx_shared_sharedData[index*2]) = u;

  //same as u, but offset 32
  float v = data->fa[__ptx_sreg_tid_x+32];
  index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2 + 32;
  *(float*)&(__ptx_shared_sharedData[index*2]) = v;

  //access shared memory (no bank conflicts)
  //both halfwarps acces successive adresses, offset of 32 for second halfwarp = (__ptx_sreg_tid_x>=16) + 16 additional offset
  index = __ptx_sreg_tid_x + (__ptx_sreg_tid_x/16)*16;
  float a = *(float*)&(__ptx_shared_sharedData[index*2]);
  data->fa[__ptx_sreg_tid_x] = a;

  // initial offset is 16 (first half wrap)
  // adresses begin with index 8 to 15 (first half (of halfwarp)) => offset 8
  // last 8 indexes start at offset 0 => %16   (threadid+8) el [8..23] => (threadid+8)%16 el [0...7]
  // offset for first index of second wrap is 32 + 8  = (((threadid==16)+8) % 16 == 8) + 32
  index = 16 + (__ptx_sreg_tid_x+8)%16 + (__ptx_sreg_tid_x/16)*32;
  float b = *(float*)&(__ptx_shared_sharedData[index*2]);
  data->fa[__ptx_sreg_tid_x+32] = b;

  //  for(int i = 0; i < ARRAY_N; i++)
  //    data->fa[i] = __ptx_shared_sharedData[i];
}

///////////////// shader tests //////////////
/*
inline float Mod(float a, float b) {
  // From Texturing & Modeling (a procedural approach) David S. Ebert
  int n = (int)(a/b);
  a -= n*b;
  if (a < 0.f)
    a += b;
  return a;
}
inline float Step(float min, float value) {
  return value < min ? 0.0f : 1.0f;
}
inline float SmoothStep(float min, float max, float value) {
  if (value < min) return 0.0f;
  if (value >= max) return 1.0f;

  float v = (value - min) / (max - min);
  return (-2.0f * v + 3.0f) * v * v;
}
inline float Mix(float x, float y, float alpha) {
  return x*(1.0f - alpha) + y*alpha;
}
*/

//debug single parquetShader parts
extern "C" void test_parquetShader(DataStruct* data)
{
  float m = data->f; //should be 1
  float l = m -10;
  float r = m +10;
  data->fa[0] = Mod(l,r);
  data->fa[1] = Mod(r,l);
  data->fa[2] = Mod(r,m);
  data->fa[3] = Step(l,r);
  data->fa[4] = Step(r,l);
  data->fa[5] = SmoothStep(l,r,m);
  data->fa[6] = SmoothStep(l,r,100*m);
  data->fa[7] = SmoothStep(l,r,-200*m);
  data->fa[8] = Mix(l,r,m);
}


Vec3f __ptx_position;
Matrix4f __ptx_matrix;
extern "C" void test_reg2Const(DataStruct* data)
{
  Vec4f worldPos = __ptx_matrix * __ptx_position;
  data->fa[0] = worldPos.x - worldPos.y - worldPos.z - worldPos.w;
}


//test light shader (no reference function is called)
// only serves for profiling purposes
#include "PTXShader.h"

#define dataS __ptx_data_dev
#define lightsS __ptx_lights_dev

ShadingData dataS;
PointLight lightsS[7];


short __ptx_shared_color[32*4];//thn.x*thn.y*(1/2float * 4)
short __ptx_shared_normal[32*4];//thn.x*thn.y*(1/2float * 4)

inline void initSharedMemory(int* ptr_color, int* ptr_normal)
{
  //write to shared memory (no bank conflicts)
  int u = ptr_color[__ptx_sreg_tid_x];
  int index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2; //(a or b?)
  *(int*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index*2]) = u;

  //same as u, but offset 32
  int v = ptr_color[__ptx_sreg_tid_x+32];
  index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2 + 32;
  *(int*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index*2]) = v;

  //write to shared memory (no bank conflicts)
  u = ptr_normal[__ptx_sreg_tid_x];
  index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2; //(a or b?)
  *(int*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2]) = u;

  //same as u, but offset 32
  v = ptr_normal[__ptx_sreg_tid_x+32];
  index = (__ptx_sreg_tid_x%2)*((1-__ptx_sreg_tid_x/16)*warp_width+8)+__ptx_sreg_tid_x/2 + 32;
  *(int*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2]) = v;
}

inline Point getPosition()
{
  //wiev coordinates
  int x = __ptx_sreg_ntid_x*__ptx_sreg_ctaid_x*THREAD_WIDTH_X+__ptx_sreg_tid_x*THREAD_WIDTH_X;
  int y = __ptx_sreg_ntid_y*__ptx_sreg_ctaid_y*THREAD_WIDTH_Y+__ptx_sreg_tid_y*THREAD_WIDTH_Y;
  float x_relative = (float)x/(float)dataS.width;
  float y_relative = (float)y/(float)dataS.height;
  Vector IN = Vector((x_relative-0.5f)*2.f, (y_relative-0.5f)*-2.f, 1) * dataS.farCorner;
  Normalize(IN);

  //load normal and dist of pixel from shared memory
  int index = 16 + (__ptx_sreg_tid_x+8)%16 + (__ptx_sreg_tid_x/16)*32;
  float d = (float)*(double*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2+1]);
  float hitDistance = d * dataS.farClipDistance;
  Point P = hitDistance * IN;// +dataS->origin //world view
  return P;
}

inline Point getNormalFF()
{
  //load normal and dist of pixel from shared memory
  int index = __ptx_sreg_tid_x + (__ptx_sreg_tid_x/16)*16;
  float a = (float)*(double*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2]);
  float b = (float)*(double*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2+1]);

  index = 16 + (__ptx_sreg_tid_x+8)%16 + (__ptx_sreg_tid_x/16)*32;
  float c = (float)*(double*)&(__ptx_shared_normal[__ptx_sreg_tid_y*32*2*2+index*2]);

  //normal
  Normal N(a,b,c);
  Normalize(N);

  //wiev coordinates
  int x = __ptx_sreg_ntid_x*__ptx_sreg_ctaid_x*THREAD_WIDTH_X+__ptx_sreg_tid_x*THREAD_WIDTH_X;
  int y = __ptx_sreg_ntid_y*__ptx_sreg_ctaid_y*THREAD_WIDTH_Y+__ptx_sreg_tid_y*THREAD_WIDTH_Y;
  float x_relative = (float)x/(float)dataS.width;
  float y_relative = (float)y/(float)dataS.height;
  Vector IN = Vector((x_relative-0.5f)*2.f, (y_relative-0.5f)*-2.f, 1) * dataS.farCorner;

  //dir vector
  Normalize(IN);

  //face forward normal
  return N;//FaceForward(N, IN);
}

extern "C" void test_lightShader(DataStruct* testData)
{
  //read from texture and copy to shared memory
  int x_shared = __ptx_sreg_ntid_x*__ptx_sreg_ctaid_x*THREAD_WIDTH_X;
  int y_shared = __ptx_sreg_ntid_y*__ptx_sreg_ctaid_y*THREAD_WIDTH_Y+__ptx_sreg_tid_y*THREAD_WIDTH_Y;
  initSharedMemory((int*)testData->ia, (int*)testData->fa);

  const  float Ka = 0.2f;//0.2
  const float Kd = 0.6f;//0.4
  const float Ks = 0.2f;//0.1
  const float roughness = 0.1f;
  const float invRoughness = 1.0f/roughness;

  Color C_diffuse(0.0f, 0.0f, 0.0f);
  Color C_specular(0.0f, 0.0f, 0.0f);

  // BEGIN_ILLUMINANCE_LOOP
  for(int l=0; l<dataS.lights_n; l++)
  {
    Point P = getPosition();
    Vector L_dir_norm = lightsS[l].position - P;
    float len_sq = Dot(L_dir_norm,L_dir_norm);
    float len = sqrtf(len_sq);
    L_dir_norm *= (1./len); //Normalize(L_dir_norm);

    //shadow?
//     if(isShadowOld(
//                lightsS[l].shadowTexture,//sampler2D shadowMap,
//                P,//Vec3f viewPos,
//                &(dataS.viewTransformInv),//Matrix4f invView,
//                &(lightsS[l].shadowTransform),//Matrix4f shadowViewProj,
//                 lightsS[l].shadowFarClip,
//                len,//distanceFromLight
//                L_dir_norm,
//                lightsS[l].direction
//                ))
//       {
//        continue;
//       }


    //recalculate
    //    L_dir_norm = lightsS[l].position - getPosition();
    //    Normalize(L_dir_norm);

    //spotlight falloff and attenuation
    float spotlightAngle = clamp2One(Dot(lightsS[l].direction, -L_dir_norm));
    float spotFalloff = clamp2One((+spotlightAngle - lightsS[l].spotlight_innerAngle) / (lightsS[l].spotlight_outerAngle - lightsS[l].spotlight_innerAngle));

    float attenuation = Dot(lightsS[l].attenuation, Vector(1.0, len, len*len))/(1-spotFalloff);

    float cosLight;
    //diffuse component
    Point Nf = getNormalFF();
    cosLight = Dot(L_dir_norm, Nf); //N

    if (cosLight >= 0.0)
      C_diffuse += lightsS[l].color_diffuse*cosLight/attenuation;

    //specular component
    L_dir_norm = L_dir_norm + Nf; //N
    Normalize(L_dir_norm);
    //        L_dir_norm *= (1./Length(L_dir_norm));
    cosLight = Dot(Nf, L_dir_norm);
    if(cosLight >= 0.0)
      C_specular += lightsS[l].color_diffuse  * powf(cosLight, invRoughness)/attenuation;//Color(val, val, val);
  }

  Color result = (Ka + Kd * C_diffuse + Ks * C_specular);

  //load color of pixel from shared memory
  int index2 = __ptx_sreg_tid_x + (__ptx_sreg_tid_x/16)*16;
  float a2 = (float)*(double*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index2*2]);
  float b2 = (float)*(double*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index2*2+1]);

  index2 = 16 + (__ptx_sreg_tid_x+8)%16 + (__ptx_sreg_tid_x/16)*32;
  float c2 = (float)*(double*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index2*2]);
  float d2 = (float)*(double*)&(__ptx_shared_color[__ptx_sreg_tid_y*32*2*2+index2*2+1]);

  Color color = Color(a2, b2, c2);
  float specular = d2;

  result = color * result;

  clamp2One(result.x);
  clamp2One(result.y);
  clamp2One(result.z);

  int x2 = __ptx_sreg_ntid_x*__ptx_sreg_ctaid_x*THREAD_WIDTH_X+__ptx_sreg_tid_x*THREAD_WIDTH_X;
  int y2 = __ptx_sreg_ntid_y*__ptx_sreg_ctaid_y*THREAD_WIDTH_Y+__ptx_sreg_tid_y*THREAD_WIDTH_Y;
  int out = rgbaToInt(255.f*result.x,255.f*result.y,255.f*result.z,0);

  testData->i = out;
}




/*
void test_atomic(DataStruct* data)
{
  int i =  __ptx_sreg_tid_x
        + (__ptx_sreg_tid_y*__ptx_sreg_ntid_x)
        + (__ptx_sreg_tid_z*__ptx_sreg_ntid_x*__ptx_sreg_ntid_y);
  data->fa[i] += __ptx_sreg_ctaid_x * __ptx_sreg_ctaid_y; //TODO: make atomic
}
*/
