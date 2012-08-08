#ifndef SCOUT_OPENCL_H_
#define SCOUT_OPENCL_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else // Linux
#include <CL/cl.h>
#endif

extern bool __sc_opencl;

extern cl_context __sc_opencl_context;
extern cl_program __sc_opencl_program;
extern cl_command_queue __sc_opencl_command_queue;

void __sc_init_opencl();

#endif // SCOUT_OPENCL_H_
