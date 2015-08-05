
/*
* Copyright (c) 2014 - 2015 NVIDIA Corporation.  All rights reserved.
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

//-----------------------------------------------------------------------------
//
//  radical_inverse_jitter.cpp: Fills 2D space in a way that has symmetry on frames 1, 5, 9, 33, ...
//
//-----------------------------------------------------------------------------

#include "optixu/optixu_vector_functions.h"

using namespace optix;

struct sampleCoords_t
{
  unsigned int s;
  float x, y;
};

sampleCoords_t hardcodedSamples[] = {
  // Manual reordering of the first 64 sequence values so that they are all fairly symmetric sets of four.
  { 4, 0.5, 0.5 },
  { 7, 0.5, 0.25 },
  { 8, 0.5, 0.75 },
  { 10, 0.25, 0.5 },
  { 12, 0.75, 0.5 },
  { 13, 0.25, 0.25 },
  { 14, 0.25, 0.75 },
  { 15, 0.75, 0.25 },
  { 16, 0.75, 0.75 },
  { 20, 0.5, 0.625 },
  { 23, 0.5, 0.375 },
  { 36, 0.625, 0.5 },
  { 42, 0.375, 0.5 },
  { 19, 0.5, 0.125 },
  { 24, 0.5, 0.875 },
  { 34, 0.125, 0.5 },
  { 44, 0.875, 0.5 },
  { 26, 0.25, 0.625 },
  { 28, 0.75, 0.625 },
  { 29, 0.25, 0.375 },
  { 31, 0.75, 0.375 },
  { 39, 0.625, 0.25 },
  { 40, 0.625, 0.75 },
  { 45, 0.375, 0.25 },
  { 46, 0.375, 0.75 },
  { 25, 0.25, 0.125 },
  { 27, 0.75, 0.125 },
  { 30, 0.25, 0.875 },
  { 32, 0.75, 0.875 },
  { 37, 0.125, 0.25 },
  { 38, 0.125, 0.75 },
  { 47, 0.875, 0.25 },
  { 48, 0.875, 0.75 },
  { 52, 0.625, 0.625 },
  { 55, 0.625, 0.375 },
  { 58, 0.375, 0.625 },
  { 61, 0.375, 0.375 },
  { 51, 0.625, 0.125 },
  { 56, 0.625, 0.875 },
  { 57, 0.375, 0.125 },
  { 62, 0.375, 0.875 },
  { 50, 0.125, 0.625 },
  { 53, 0.125, 0.375 },
  { 60, 0.875, 0.625 },
  { 63, 0.875, 0.375 },
  { 49, 0.125, 0.125 },
  { 54, 0.125, 0.875 },
  { 59, 0.875, 0.125 },
  { 64, 0.875, 0.875 }
};

unsigned int compact(unsigned int y)
{
  y &= 0x55555555;
  y = (y | (y >> 1)) & 0x33333333;
  y = (y | (y >> 2)) & 0x0f0f0f0f;
  y = (y | (y >> 4)) & 0x00ff00ff;
  y = (y | (y >> 8)) & 0x0000ffff;
  return y;
}

unsigned int brev(unsigned int x)
{
  x = ((x >> 1) & 0x55555555) | ((x << 1) & 0xaaaaaaaa);
  x = ((x >> 2) & 0x33333333) | ((x << 2) & 0xcccccccc);
  x = ((x >> 4) & 0x0f0f0f0f) | ((x << 4) & 0xf0f0f0f0);
  x = ((x >> 8) & 0x00ff00ff) | ((x << 8) & 0xff00ff00);
  x = ((x >> 16)) | ((x << 16));
  return x;
}

float4 radical_inverse_jitter(unsigned int& sequence)
{
  float x = 0, y = 0;

  if (sequence <= 64) {
    if (sequence <= sizeof(hardcodedSamples) / sizeof(sampleCoords_t)) {
      x = hardcodedSamples[sequence - 1].x;
      y = hardcodedSamples[sequence - 1].y;
      sequence++;
    }

    if (sequence >= sizeof(hardcodedSamples) / sizeof(sampleCoords_t))
      sequence = 65;
  }
  else {
    unsigned int ix, iy;

    do {
      unsigned int ss = brev(sequence);
      ix = compact(ss);
      iy = compact(ss >> 1);
      sequence++;
    } while (ix == 0 || iy == 0);

    x = (float)ix / 65536.0f;
    y = (float)iy / 65536.0f;
  }

  float4 jitter = make_float4(x, y, x, y);

  return jitter;
}
