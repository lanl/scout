

#ifndef COLMAJOR_ISODATA_FIELD_H_
#define COLMAJOR_ISODATA_FIELD_H_

#include <scout/Runtime/isosurf/colmajor_image3d.h>
#include <scout/Runtime/isosurf/piston/choose_container.h>
#include <scout/Runtime/isosurf/piston/implicit_function.h>

using namespace piston;

namespace scout {

// TODO: should we parameterize the ValueType? only float makes sense.
template <typename IndexType, typename ValueType, typename Space>
struct colmajor_isodata_field : public piston::colmajor_image3d<IndexType, ValueType, Space>
{
  typedef piston::colmajor_image3d<IndexType, ValueType, Space> Parent;

  typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;

  PointDataContainer point_data_vector;

  typedef typename PointDataContainer::iterator PointDataIterator;

  colmajor_isodata_field(float *image, int nx, int ny, int nz) 
    :Parent(nx, ny, nz),
    point_data_vector((ValueType *) image, (ValueType *) image + this->NPoints) 
  {}

  void resize(int xdim, int ydim, int zdim) {
    Parent::resize(xdim, ydim, zdim);
    // TBD, is there resize in VTK?
  }

  PointDataIterator point_data_begin() {
    return point_data_vector.begin();
  }

  PointDataIterator point_data_end() {
    return point_data_vector.end();
  }

};

}

#endif /* COLMAJOR_ISODATA_FIELD_H_ */
