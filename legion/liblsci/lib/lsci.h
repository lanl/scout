/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 */

#ifndef LSCI_H_INCLUDED
#define LSCI_H_INCLUDED

#include <stdbool.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* ideally, this value would be set during configure */
#define LSCI_RECT_1D_CXX_SIZE 8

typedef struct lsci_rect_1d_storage_t {
    uint8_t buffer[LSCI_RECT_1D_CXX_SIZE];
} lsci_rect_1d_storage_t;

enum {
    LSCI_SUCCESS = 0,
    LSCI_FAILURE
};
// TODO gather from actual
typedef enum lsci_privilege_mode_t {
    LSCI_NO_ACCESS       = 0x00000000,
    LSCI_READ_ONLY       = 0x00000001,
    LSCI_READ_WRITE      = 0x00000111,
    LSCI_WRITE_ONLY      = 0x00000010,
    LSCI_WRITE_DISCARD   = 0x00000010,
    LSCI_REDUCE          = 0x00000100,
    LSCI_PROMOTED        = 0x00001000
} lsci_privilege_mode_t;

typedef enum lsci_coherence_property_t {
    LSCI_EXCLUSIVE    = 0,
    LSCI_ATOMIC       = 1,
    LSCI_SIMULTANEOUS = 2,
    LSCI_RELAXED      = 3
} lsci_coherence_property_t;

/* if you update this, update lsci_dt_size_tab */
typedef enum {
    LSCI_TYPE_INT32,
    LSCI_TYPE_INT64,
    LSCI_TYPE_FLOAT,
    LSCI_TYPE_DOUBLE,
    /* sentinel */
    LSCI_TYPE_MAX
    /* TODO -- complete list -- */
} lsci_dt_t;

typedef void* lsci_runtime_t;
typedef void* lsci_context_t;
typedef void* lsci_logical_region_t;
typedef void* lsci_logical_partition_t;
typedef void* lsci_index_space_t;
typedef void* lsci_domain_handle_t;
typedef void* lsci_physical_regions_t;
typedef void* lsci_rect_1d_t;
typedef int   lsci_field_id_t;

size_t
lsci_sizeof_cxx_rect_1d(void);

lsci_rect_1d_t
lsci_subgrid_bounds_at(lsci_rect_1d_t rect_1d_array_basep,
                       size_t index);

typedef struct lsci_domain_t {
    // points to the Domain instance
    lsci_domain_handle_t hndl;
    // volume of the given Domain
    size_t volume;
} lsci_domain_t;

////////////////////////////////////////////////////////////////////////////////
// convenience vector abstraction
////////////////////////////////////////////////////////////////////////////////
typedef struct lsci_vector_t {
    size_t lr_len;
    lsci_field_id_t fid;
    lsci_index_space_t index_space;
    lsci_logical_region_t logical_region;
    lsci_logical_partition_t logical_partition;
    lsci_domain_t launch_domain;
    size_t subgrid_bounds_len;
    lsci_rect_1d_t subgrid_bounds;
} lsci_vector_t;


int
lsci_vector_dump(lsci_vector_t *vec,
                 lsci_dt_t type,
                 lsci_context_t context,
                 lsci_runtime_t runtime);

////////////////////////////////////////////////////////////////////////////////
typedef void* lsci_index_launcher_handle_t;
typedef void* lsci_task_argument_t; // TODO
typedef void* lsci_argument_map_handle_t;

typedef struct lsci_argument_map_t {
    lsci_argument_map_handle_t hndl;
} lsci_argument_map_t;

int
lsci_argument_map_create(lsci_argument_map_t *arg_map);

int
lsci_argument_map_set_point(lsci_argument_map_t *arg_map,
                            // not quite the same as legion, but this will work
                            size_t tid,
                            void *payload_base,
                            size_t payload_extent);

typedef struct lsci_index_launcher_t {
    lsci_index_launcher_handle_t hndl;
    int task_id;
    lsci_domain_t domain;
    // TODO add task args and argmap
} lsci_index_launcher_t;

typedef unsigned int lsci_projection_id_t;
typedef void* lsci_region_requirement_hndl_t;
typedef struct lsci_region_requirement_t {
    lsci_region_requirement_hndl_t hndl;
    lsci_logical_region_t region;
    lsci_projection_id_t projection_id;
    lsci_privilege_mode_t priv_mode;
    lsci_coherence_property_t coherence_prop;
    lsci_logical_partition_t parent;
} lsci_region_requirement_t;

int
lsci_index_launcher_create(lsci_index_launcher_t *il,
                           int task_id,
                           lsci_domain_t *ldom,
                           lsci_task_argument_t *task_arg,
                           lsci_argument_map_t *arg_map);

int
lsci_add_region_requirement(lsci_index_launcher_t *il,
                           lsci_logical_region_t lr,
                           lsci_projection_id_t proj_id,
                           lsci_privilege_mode_t priv_mode,
                           lsci_coherence_property_t coherence_prop,
                           lsci_logical_partition_t parent);

int
lsci_add_field(lsci_index_launcher_t *il,
               unsigned int idx,
               lsci_field_id_t field_id);

int
lsci_execute_index_space(lsci_runtime_t runtime,
                         lsci_context_t context,
                         lsci_index_launcher_t *il);

int
lsci_vector_create(lsci_vector_t *vec,
                   size_t len,
                   lsci_dt_t type,
                   lsci_context_t context,
                   lsci_runtime_t runtime);

int
lsci_vector_partition(lsci_vector_t *vec,
                      size_t n_parts,
                      lsci_context_t context,
                      lsci_runtime_t runtime);

////////////////////////////////////////////////////////////////////////////////
// convenience mesh abstraction
////////////////////////////////////////////////////////////////////////////////
typedef void* lsci_unimesh_handle_t;
typedef struct lsci_unimesh_t {
    // points to the underlying mesh instance
    lsci_unimesh_handle_t hndl;
    size_t dims;
    size_t width;
    size_t height;
    size_t depth;
} lsci_unimesh_t;

int
lsci_unimesh_create(lsci_unimesh_t *mesh,
                    size_t w,
                    size_t h,
                    size_t d,
                    lsci_context_t context,
                    lsci_runtime_t runtime);

int
lsci_unimesh_add_field(lsci_unimesh_t *mesh,
                       lsci_dt_t type,
                       char *field_name,
                       lsci_context_t context,
                       lsci_runtime_t runtime);

int
lsci_unimesh_partition(lsci_unimesh_t *mesh,
                       size_t n_parts,
                       lsci_context_t context,
                       lsci_runtime_t runtime);
int
lsci_unimesh_get_vec_by_name(lsci_unimesh_t *mesh,
                             char *name,
                             lsci_vector_t *vec,
                             lsci_context_t context,
                             lsci_runtime_t runtime);

////////////////////////////////////////////////////////////////////////////////
// convenience struct abstraction
////////////////////////////////////////////////////////////////////////////////
typedef void* lsci_struct_handle_t;
typedef struct lsci_struct_t {
    // points to the underlying struct instance
    lsci_struct_handle_t hndl;
} lsci_struct_t;

int
lsci_struct_create(lsci_struct_t *theStruct,
                   lsci_context_t context,
                   lsci_runtime_t runtime);

int
lsci_struct_add_field(lsci_struct_t *theStruct,
                      lsci_dt_t type,
                      size_t length,
                      char *field_name,
                      lsci_context_t context,
                      lsci_runtime_t runtime);

int
lsci_struct_partition(lsci_struct_t *theStruct,
                      size_t n_parts,
                      lsci_context_t context,
                      lsci_runtime_t runtime);

int
lsci_struct_get_vec_by_name(lsci_unimesh_t *theStruct,
                            char *name,
                            lsci_vector_t *vec,
                            lsci_context_t context,
                            lsci_runtime_t runtime);


/* arguments passed to tasks during their invocation */
typedef struct lsci_task_args_t {
    lsci_context_t context;
    lsci_runtime_t runtime;
    int task_id;
    /// physical region things ///
    size_t n_regions;
    lsci_physical_regions_t regions;
    void *local_argsp;
} lsci_task_args_t;

typedef struct lsci_mesh_task_args_t {
  size_t rank;
  size_t width;
  size_t height;
  size_t depth;
  lsci_rect_1d_t rect;
  size_t len;
} lsci_mesh_task_args_t;


// TODO add macro magic to create different ret type sigs
#define LSCI_FUNP_SIG(ret_type) \
    ret_type (*cbf)(struct lsci_task_args_t *task_args)

typedef struct lsci_reg_task_data_t {
    LSCI_FUNP_SIG(void);
} lsci_reg_task_data_t;

typedef unsigned long lsci_variant_id_t;
#define LSCI_AUTO_GENERATE_ID UINT_MAX

// TODO make a real mapping to the real underlying values
typedef enum lsci_proc_kind_t {
    LSCI_TOC_PROC, /* throughput core */
    LSCI_LOC_PROC, /* latency core */
    LSCI_UTIL_PROC /* utility core */
} lsci_proc_kind_t;

/**
 * wrapper for HighLevelRuntime::start()
 */
int
lsci_start(int argc,
           char **argv);

/**
 *
 */
void
lsci_set_top_level_task_id(int task_id);

/**
 * wrapper for: HighLevelRuntime::register_legion_task<>()
 */
int
lsci_register_void_legion_task(
    int task_id,
    lsci_proc_kind_t p_kind,
    bool single,
    bool index,
    bool leaf,
    lsci_variant_id_t vid,
    char *name,
    lsci_reg_task_data_t reg_task_data
);

int
lsci_register_void_legion_task_aux(
    int task_id,
    lsci_proc_kind_t p_kind,
    bool single,
    bool index,
    bool leaf,
    lsci_variant_id_t vid,
    char *name,
    void (*atask)(struct lsci_task_args_t *task_args)
);

void *
raw_rect_ptr_1d(lsci_physical_regions_t rgnp,
                lsci_dt_t type,
                size_t region_id,
                lsci_field_id_t fid,
                lsci_rect_1d_t subgrid_bounds);

/**
 * Print functions for debugging IR
 */

void lsci_print_mesh_task_args(lsci_mesh_task_args_t* mtargs);
void lsci_print_task_args_local_argsp(lsci_task_args_t* targs);

#ifdef __cplusplus
}
#endif

#endif
