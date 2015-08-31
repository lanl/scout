/*
 * ###########################################################################
 * Copyright (c) 2014, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */ 

#include <stdint.h>

#include "legion_c.h"

#ifndef __SCLEGION_H__
#define __SCLEGION_H__

#ifdef __cplusplus
extern "C"{
#endif

#define NEW_OPAQUE_TYPE(T) typedef struct T { void* impl; } T  
  NEW_OPAQUE_TYPE(sclegion_uniform_mesh_t);
  NEW_OPAQUE_TYPE(sclegion_uniform_mesh_launcher_t);
#undef NEW_OPAQUE_TYPE
  
  typedef enum sclegion_field_kind_t{
    SCLEGION_INT32,
    SCLEGION_INT64,
    SCLEGION_FLOAT,
    SCLEGION_DOUBLE,
    SCLEGION_FIELD_MAX
  } sclegion_field_kind_t;

  typedef enum sclegion_element_kind_t{
    SCLEGION_CELL,
    SCLEGION_VERTEX,
    SCLEGION_EDGE,
    SCLEGION_FACE,
    SCLEGION_ELEMENT_MAX
  } sclegion_element_kind_t;

  void
  sclegion_init(const char* main_task_name,
                legion_task_pointer_void_t main_task_pointer);
  
  int
  sclegion_start(int argc, char** argv);

  void
  sclegion_register_task(legion_task_id_t task_id,
                         const char* task_name,
                         legion_task_pointer_void_t task_pointer);

  sclegion_uniform_mesh_t
  sclegion_uniform_mesh_create(legion_runtime_t runtime,
                               legion_context_t context,
                               size_t rank,
                               size_t width,
                               size_t height,
                               size_t depth,
                               void* topology);

  void
  sclegion_uniform_mesh_add_field(sclegion_uniform_mesh_t mesh,
                                  const char* field_name,
                                  sclegion_element_kind_t element_kind,
                                  sclegion_field_kind_t field_kind);

  void
  sclegion_uniform_mesh_init(sclegion_uniform_mesh_t mesh);

  void*
  sclegion_uniform_mesh_reconstruct(const legion_task_t task,
                                    const legion_physical_region_t* region,
                                    unsigned numRegions,
                                    legion_context_t context,
                                    legion_runtime_t runtime);

  sclegion_uniform_mesh_launcher_t
  sclegion_uniform_mesh_create_launcher(sclegion_uniform_mesh_t mesh,
                                        legion_task_id_t task_id);

  void
  sclegion_uniform_mesh_launcher_add_field(
    sclegion_uniform_mesh_launcher_t launcher,
    const char* field_name,
    legion_privilege_mode_t mode);

  void
  sclegion_uniform_mesh_launcher_execute(
    legion_context_t context,
    legion_runtime_t runtime,
    sclegion_uniform_mesh_launcher_t launcher);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __SCLEGION_H__

