list of runtime functions currently being used:

We are calling __scrt_init() in Runtime/Initialization.cpp from CodeGen/Scout/CGScoutRuntime.cpp, 
but it is not doing anything useful at this point. It is setting up an CpuDevice
and a stub CpuRuntime (currently empty)

We are calling __scrt_malloc() in Runtime/cpu/MemAlloc.cpp from CodeGen/Scout/CGScoutRuntime.cpp 

We are calling __scrt_renderall_uniform_begin() in Runtime/renderall/RenderallUniform.cpp
from CodeGen/Scout/CGScoutRuntime.cpp

We are calling __scrt_renderall_end() in Runtime/renderall/RenderallBase.cpp from 
CodeGen/Scout/CGScoutRuntime.cpp

We are accessing the global  buffer __scrt_renderall_uniform_colors in  
Runtime/renderall/RenderallUniformImpl.h from CodeGen/Scout/CGScoutRuntime.cpp


