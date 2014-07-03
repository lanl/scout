list of runtime functions currently being used:

We are calling __scrt_init() in Runtime/Initialization.cpp from CodeGen/Scout/CGScoutRuntime.cpp, 
but it is not doing anything useful at this point. It is setting up an CpuDevice
and a stub CpuRuntime (currently empty)

We are calling __scrt_malloc() in Runtime/cpu/MemAlloc.cpp from CodeGen/Scout/CGScoutRuntime.cpp 

