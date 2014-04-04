/**
 * hpgv_error.h
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

#ifndef HPGV_ERROR_H
#define HPGV_ERROR_H

namespace scout {

  extern "C" {

#define HPGV_TRUE                1
#define HPGV_FALSE               0

#define HPGV_SUCCESS             0
#define HPGV_ERROR              -1
#define HPGV_ERR_MEM            -2
#define HPGV_ERR_IO             -4
#define HPGV_ERR_COMM           -3
#define HPGV_ERR_OVERFLOW       -5

  }

} // end namespace scout

#endif
