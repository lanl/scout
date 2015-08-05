# Copyright 2015 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

ifndef GASNET_ROOT
$(error GASNET_ROOT is not defined)
endif

ifndef QT_HOME
$(error QT_HOME is not defined)
endif

ifndef OPTIX_DIR
$(error OPTIX_DIR is not defined)
endif

#Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time print level
SHARED_LOWLEVEL ?= 0		# Use the shared low level
ALT_MAPPERS     ?= 0		# Compile the alternative mappers
CONDUIT := udp

# Put the binary file name here
OUTFILE		?= composite
# List all the application source files here
GEN_SRC		?= composite.cc DataMgr.cc radical_inverse_jitter.cc isosurface_render_kernel.cc#better_mapper.cc radical_inverse_jitter.cc	# .cc files
GEN_GPU_SRC	?= render_kernel.cu CUDAMarchingCubes.cu #circuit_gpu.cu				# .cu files
GEN_GPU_PTX_SRC ?= constantbg.cu matrix_camera.cu mesh_hit.cu triangle_mesh.cu

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=-I${QT_HOME} -I${OPTIX_DIR}/include -I${OPTIX_DIR}/include/optixu #-I${GASNET_ROOT}/include # -I${EIGEN_HOME} -I${QT_HOME}
CC_FLAGS	?= -std=c++11 -DREALM_BACKTRACE  
NVCC_FLAGS	?= --gpu-architecture=sm_30 -std=c++11 -I${OPTIX_DIR}/include -I${QT_HOME}  #p--keep
GASNET_FLAGS	?=
LD_FLAGS	?= -L${QT_HOME} -lcudart -lQtViewer -L${OPTIX_DIR}/lib64 -loptix -loptixu
GPU_ARCH	:= sm_30

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

