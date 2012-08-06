/**
 * hpgv_util.h
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

#ifndef HPGV_UTIL_H
#define HPGV_UTIL_H

    
#include "scout/Runtime/volren/hpgv/hpgv_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>

namespace scout {
  extern "C" {

#if !defined(MAXLINE)
#define MAXLINE 4096      /* max line length */
#endif


#define HPGV_ERR_MSG(msg) {\
  fprintf(stderr, "<%s:%s:%d> : %s \n", \
      __FILE__, __func__, __LINE__, (msg));\
}

#define HPGV_ABORT(msg, errno) {\
  HPGV_ERR_MSG(msg);\
  exit(errno);\
}    

#define HPGV_ASSERT(condition, msg, errno) {\
  if (!(condition)) {\
    HPGV_ABORT(msg, errno);\
  }\
}

#define MPI_INCLUDED
#ifdef MPI_INCLUDED

#define HPGV_ERR_MSG_P(id, msg) {\
  fprintf(stderr, "Proc %d <%s:%s:%d> : %s \n", \
      (id), __FILE__, __func__, __LINE__, (msg));\
}

#define HPGV_ABORT_P(id, msg, errno) {\
  HPGV_ERR_MSG_P(id, msg); \
  MPI_Abort(MPI_COMM_WORLD, errno);\
  exit(errno);\
}

#define HPGV_ASSERT_P(id, condition, msg, errno) {\
  if (!(condition)) {\
    HPGV_ABORT_P(id, msg, errno);\
  }\
}

#endif

  void hpgv_msg(const char *fmt, ...);
  void hpgv_msg_p(int id, int root, const char *fmt, ...);


  }

} // end namespace scout
    
#endif
