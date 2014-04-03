/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *----- 
 * 
 */

#ifndef ISOSURFACE_CPP_H_
#define ISOSURFACE_CPP_H_

#include <scout/Runtime/isosurf/piston/marching_cube.h>
#include <scout/Runtime/isosurf/colmajor_isodata_field.h>
#include <scout/Runtime/isosurf/user_defined_color_func.h>

// ----- isosurface for CPP backend of Piston
// 

#define SPACE thrust::device_system_tag

using namespace piston;

namespace scout
{

  class Isosurface_CPP{

    public:

      Isosurface_CPP(int nx, int ny, int nz, float* input, float* source, float isovalue);
      ~Isosurface_CPP();
      float* getVertices();
      float* getNormals();
      float* getScalars();
      float getMinScalar();
      float getMaxScalar();
      int getNumVertices();
      void computeColors(thrust::host_vector<psfloat4>* colors_host_vector,
          const piston::hsv_color_map<float>& color_func);
      void computeColors2(thrust::host_vector<psfloat4>* colors_host_vector,
          const user_defined_color_func ud_color_func);
      void recompute(float isovalue);

    private:
      int             _nx, _ny, _nz;
      colmajor_isodata_field<int, float, SPACE>* _input;
      colmajor_isodata_field<int, float, SPACE>* _source;
      marching_cube<
        colmajor_isodata_field<int, float, SPACE>, 
        colmajor_isodata_field<int, float, SPACE> 
      > *_isosurface;

    public:

  };
}

extern scout::Isosurface_CPP* __sc_isosurface_cpp;
extern void __sc_isosurface(size_t nx, size_t ny, size_t nz, float* input, float* source, float isoval);
extern void __sc_recompute_isosurface(float isoval);

#endif
