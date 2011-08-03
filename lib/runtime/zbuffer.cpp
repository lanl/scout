/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 * 
 */

#include <cassert>

#include "runtime/zbuffer.h"

using namespace scout;


// ----- zbuffer_rt
//
zbuffer_rt::zbuffer_rt(dim_t w, dim_t h, float depth)
{
  assert(w > 0 && h > 0);
  
  width  = w;
  height = h;

  values = new float[width * height];

  this->clear();
}


// ----- ~zbuffer_rt
//
zbuffer_rt::~zbuffer_rt()
{
  delete []values;
}


// ----- clear
//
void zbuffer_rt::clear()
{
  index_t size = width * height;
  
  #pragma omp for schedule(dynamic, width)
  for(index_t i = 0; i < size; ++i) {
    values[i] = depth;
  }
}

