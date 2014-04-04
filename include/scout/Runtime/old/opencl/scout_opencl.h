#ifndef SCOUT_OPENCL_H_
#define SCOUT_OPENCL_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#else // Linux
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

extern bool __sc_opencl;

extern cl_context __sc_opencl_context;
extern cl_program __sc_opencl_program;
extern cl_command_queue __sc_opencl_command_queue;
extern cl_mem __scrt_renderall_uniform_opencl_device;

void __sc_init_opencl();

#endif // SCOUT_OPENCL_H_
