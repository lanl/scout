/**
 * hpgv_utilmath.h
 *
 * Copyright (c) 2008 Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 * 
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */

#ifndef HPGV_UTILMATH_H
#define HPGV_UTILMATH_H


    
#include "scout/Runtime/volren/hpgv/hpgv_error.h"
#include <math.h>

namespace scout {

  extern "C" {

    /**
     * point_3d_t:
     *
     */
    typedef struct point_3d_t {
      double  x3d, y3d, z3d;
    } point_3d_t;

#define HPGV_HUGE              1e+10

#define HPGV_HUGE_INT          0x0FFFFFFF

#define BLOCK_LOW(id, p, n) \
    ((uint64_t)(((float)(id)) /(p) * (n)))

#define BLOCK_HIGH(id, p, n) \
    (BLOCK_LOW((id) + 1, p, n) - 1)

#define BLOCK_SIZE(id, p, n) \
    (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW(id, p, n))

#define log_2(x) (log((x)) / log(2))

#define SQR(x) ((x) * (x))

#define MAT_TIME_VEC(matrix, in, out) {\
  (out).x3d = (in).x3d * matrix[0] + (in).y3d * matrix[1] + \
  (in).z3d * matrix[2] + matrix[3]; \
  (out).y3d = (in).x3d * matrix[4] + (in).y3d * matrix[5] + \
  (in).z3d * matrix[6] + matrix[7]; \
  (out).z3d = (in).x3d * matrix[8] + (in).y3d * matrix[9] + \
  (in).z3d * matrix[10] + matrix[11];\
}

#define VEC_TIME_MAT(matrix, in, out) {\
  (out).x3d = (in).x3d * matrix[0] + (in).y3d * matrix[4] + \
  (in).z3d * matrix[8] + matrix[12]; \
  (out).y3d = (in).x3d * matrix[1] + (in).y3d * matrix[5] + \
  (in).z3d * matrix[9] + matrix[13]; \
  (out).z3d = (in).x3d * matrix[2] + (in).y3d * matrix[6] + \
  (in).z3d * matrix[10] + matrix[14];\
}

#define VEC_SET(des, x, y, z) {\
  (des).x3d = (x); \
  (des).y3d = (y); \
  (des).z3d = (z); \
}

#define VEC_CPY(src, des) {\
  (des).x3d = (src).x3d; \
  (des).y3d = (src).y3d; \
  (des).z3d = (src).z3d; \
}


#define VEC_MINUS(v1, v2, v3) {\
  (v3).x3d = (v1).x3d - (v2).x3d; \
  (v3).y3d = (v1).y3d - (v2).y3d; \
  (v3).z3d = (v1).z3d - (v2).z3d; \
}

#define VEC_ADD(v1, v2, v3) {\
  (v3).x3d = (v1).x3d + (v2).x3d; \
  (v3).y3d = (v1).y3d + (v2).y3d; \
  (v3).z3d = (v1).z3d + (v2).z3d; \
}

#define VEC_DOT_VEC(v1, v2) (\
    ((v1).x3d * (v2).x3d + (v1).y3d * (v2).y3d + (v1).z3d * (v2).z3d)\
    )

#define VEC_DOT_VAL(v1, v2) {\
  (v1).x3d *= (v2);\
  (v1).y3d *= (v2);\
  (v1).z3d *= (v2);\
}



#define DIST_SQR_VEC(v1, v2) \
  (SQR((v1).x3d - (v2).x3d) + \
   SQR((v1).y3d - (v2).y3d) + \
   SQR((v1).z3d - (v2).z3d))

#define MAX(x, y) ((x) > (y)? (x) : (y))

#define MIN(x, y) ((x) < (y)? (x) : (y))

#define CLAMP(x, minval, maxval) (MIN(MAX(x, (minval)), (maxval)))

void
mat_time_mat(const double a[16], const double b[16], double c[16]);

int
inverse_mat(const double a[16], double b[16]);

void
mat_time_vec(const double m[16], const double in[3], double out[3]);

int
normalize(point_3d_t *p3d);

double
length(point_3d_t p3d);


int 
is_mat_zero(const double m[16]); 


}

} // end namespace scout
    
#endif
