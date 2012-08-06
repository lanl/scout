

#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

namespace scout {

  extern "C"  {

    void
      getViewMatrix(double eyex, double eyey, double eyez, double centerx,
          double centery, double centerz, double upx, double upy,
          double upz, double m[4][4]);

    void
      getProjectionMatrix(double fovy, double aspect, double zNear, 
          double zFar, double m[4][4]);

    void
      transposed( double to[16], const double from[16] );

  }

} // end namespace scout

#endif
