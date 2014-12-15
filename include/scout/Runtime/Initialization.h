#ifndef __SC_INITIALIZATION_H__
#define __SC_INITIALIZATION_H__

#include "scout/sys/linux/unistd.h"

#if defined(__scout_cxx__) || defined(__cplusplus)
#include "scout/Runtime/opengl/glCamera.h"

extern const size_t __scrt_initial_window_width;
extern const size_t __scrt_initial_window_height;

extern "C" void __scrt_init_cpu();

#else
extern void __scrt_init_cpu();
#endif
#endif 

