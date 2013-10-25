#ifndef SCOUT_H_
#define SCOUT_H_


#include "scout/unistd.h"
#include "scout/Runtime/renderall/RenderallUniform.h"
#include "scout/Runtime/window.h"
#include "scout/Runtime/image.h"
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/Device.h"
#include "scout/Runtime/DeviceList.h"
#include "scout/Runtime/gpu.h"

#ifdef SC_ENABLE_MPI
#include "scout/Runtime/renderall/mpi/RenderallVolume.h"
extern MPI_Comm __volren_gcomm;
#endif

extern const size_t __scrt_initial_window_width;
extern const size_t __scrt_initial_window_height;

void __scrt_init(ScoutDeviceType gpuType);

void __scrt_end();

#endif // SCOUT_SCH_
