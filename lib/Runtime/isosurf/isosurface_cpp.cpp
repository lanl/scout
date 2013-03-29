#include <scout/Runtime/isosurf/isosurface_cpp.h>

#include <stdio.h>

using namespace piston;
using namespace scout;

Isosurface_CPP* __sc_isosurface_cpp = 0;

Isosurface_CPP::Isosurface_CPP(int nx, int ny, int nz,
    float* input, float* source, float isovalue) 
  :_nx(nx), _ny(ny), _nz(nz) 
{
  _input = new colmajor_isodata_field<int, float, SPACE>(input, nx, ny, nz);
  _source = new colmajor_isodata_field<int, float, SPACE>(source, nx, ny, nz);

  _isosurface = new marching_cube<
      colmajor_isodata_field<int, float, SPACE>,  
      colmajor_isodata_field<int, float, SPACE> 
    >(*_input, *_source, isovalue);

    (*_isosurface)();
}

Isosurface_CPP::~Isosurface_CPP() {
  if (_isosurface) {
    delete _isosurface;
  }
}

void Isosurface_CPP::recompute(float isovalue) {
  _isosurface->set_isovalue(isovalue);
  (*_isosurface)();
}

float* Isosurface_CPP::getVertices() {
  return (float*)thrust::raw_pointer_cast(&_isosurface->vertices[0]);
}

float* Isosurface_CPP::getNormals() {
  return (float*)thrust::raw_pointer_cast(&_isosurface->normals[0]);
}

float* Isosurface_CPP::getScalars() {
  return thrust::raw_pointer_cast(&_isosurface->scalars[0]);
}

float Isosurface_CPP::getMinScalar() {
  return *thrust::min_element(_isosurface->scalars_begin(), _isosurface->scalars_end());
}

float Isosurface_CPP::getMaxScalar() {
  return *thrust::max_element(_isosurface->scalars_begin(), _isosurface->scalars_end());
}

int Isosurface_CPP::getNumVertices() {
  return thrust::distance(_isosurface->vertices_begin(), _isosurface->vertices_end());
}

void Isosurface_CPP::computeColors(thrust::host_vector<psfloat4>* colors_host_vector, 
    const piston::hsv_color_map<float>& color_func) 
{
  colors_host_vector->assign(
      thrust::make_transform_iterator(_isosurface->scalars_begin(), color_func),
      thrust::make_transform_iterator(_isosurface->scalars_end(), color_func));
}

// could take mesh ptr and set the fields of it instead of setting globals
void __sc_isosurface(size_t nx, size_t ny, size_t nz, float* input, float* source, float isoval){
  __sc_isosurface_cpp = new Isosurface_CPP(nx, ny, nz, input, source, isoval);
}

void __sc_recompute_isosurface(float isoval) {
  if (__sc_isosurface_cpp) {
    __sc_isosurface_cpp->recompute(isoval);
  }
}
