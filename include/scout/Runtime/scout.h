#ifndef SCOUT_H_
#define SCOUT_H_

#include "scout/Runtime/renderall/renderall_uniform.h"
#include "scout/Runtime/window.h"
#include "scout/Runtime/image.h"
#include "scout/Runtime/opengl/glCamera.h"
#include "scout/Runtime/Device.h"
#include "scout/Runtime/DeviceList.h"
#include "scout/Runtime/gpu.h"

#ifdef SC_ENABLE_MPI
#include "scout/Runtime/renderall/mpi/volume_renderall.h"
extern MPI_Comm __volren_gcomm;
#endif

extern scout::DeviceList DevList;
extern scout::glCamera* __sc_camera;
extern const size_t __sc_initial_width;
extern const size_t __sc_initial_height;

void __sc_init_sdl(size_t width, size_t height, scout::glCamera* cam = NULL);

void __sc_init(int argc, char** argv, ScoutDeviceType gpuType);

void __sc_init(ScoutDeviceType gpuType);

void __sc_end();

#endif // SCOUT_SCH_
