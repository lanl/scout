/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 */

#include "lsci.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define APP_NAME_STR "c-unimesh"

enum {
    MAIN_TID = 0,
    INIT_VALS_TID,
    VECCP_TID
};

typedef struct mesh_task_args_t {
    // common mesh info (global)
    uint32_t rank;
    uint32_t global_width;
    uint32_t global_height;
    uint32_t global_depth;
    // mesh sub-grid bounds (per task). we only need one because all the sgb
    // should be the same across all fields.
    lsci_rect_1d_storage_t sgb;
    // length of sgb
    size_t sgb_len;
} mesh_task_args_t;

#if 0
void
veccp(lsci_vector_t *from,
      lsci_vector_t *to,
      lsci_context_t context,
      lsci_runtime_t runtime)
{
    assert(from && to && context && runtime);

    int idx = 0;
    printf("-- %s: from dom vol: %ld to dom vol: %ld\n",
           __func__,
           (long int)from->launch_domain.volume,
           (long int)to->launch_domain.volume);
    assert(from->launch_domain.volume == to->launch_domain.volume);
    lsci_argument_map_t arg_map;
    lsci_argument_map_create(&arg_map);
    // TODO add general task args. can pass common mesh info there
    for (size_t i = 0; i < to->launch_domain.volume; ++i) {
        veccp_args_t t_ar
    }
    // setup the index launcher
    lsci_index_launcher_t il;
    lsci_index_launcher_create(&il, VECCP_TID,
                               &to->launch_domain,
                               NULL, &arg_map);
    lsc_add_region_requirement(
        &il, from->logical_partition, 0,
        LSCI_READ_ONLY, LSCI_EXCLUSIVE, from->logical_region
    );
    lsci_add_field(&il, idx++, from->fid);
    lsc_add_region_requirement(
        &il, to->logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, to->logical_region
    );
    lsci_add_field(&il, idx++, to->fid);
    lsci_execute_index_space(runtime, context, &il);
}

void
veccp_task(lsci_task_args_t *task_args)
{
    assert(task_args && task_args->runtime && task_args->context);
    assert(task_args->regions);
    size_t rid = 0;
#if 0
    printf("-- %s (%d): got %lu physical regions\n",
            __func__, task_args->task_id,
            (unsigned long)task_args->n_regions);
#endif
    veccp_args_t t_args = *(veccp_args_t *)task_args->local_argsp;
    lsci_rect_1d_t from_sgb = (lsci_rect_1d_t)&t_args.vec_a_sgb;
    lsci_rect_1d_t to_sgb   = (lsci_rect_1d_t)&t_args.vec_b_sgb;
    assert(t_args.vec_a_sgb_len == t_args.vec_b_sgb_len);
    double *fromp = raw_rect_ptr_1d_double(
                        task_args->regions, rid++, 0, from_sgb
                    );
    double *top = raw_rect_ptr_1d_double(
                        task_args->regions, rid++, 0, to_sgb
                  );
    assert(fromp && top);
    for (size_t i = 0; i < t_args.vec_a_sgb_len; ++i) {
        top[i] = fromp[i];
    }
}
#endif

void
init_vals(lsci_unimesh_t *mesh,
          lsci_context_t context,
          lsci_runtime_t runtime)
{
    assert(mesh && context && runtime);
    lsci_vector_t field_a;
    lsci_vector_t field_b;

    lsci_unimesh_get_vec_by_name(mesh, "field-a", &field_a, context, runtime);
    lsci_unimesh_get_vec_by_name(mesh, "field-b", &field_b, context, runtime);

    assert(field_a.launch_domain.volume == field_b.launch_domain.volume);
    int idx = 0;
    lsci_argument_map_t arg_map;
    lsci_argument_map_create(&arg_map);
    for (size_t i = 0; i < field_a.launch_domain.volume; ++i) {
        mesh_task_args_t targs;
        targs.global_width = mesh->width;
        targs.global_height = mesh->height;
        targs.global_depth = mesh->depth;
        targs.rank = mesh->dims;
        targs.sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(field_a.subgrid_bounds, i);
        targs.sgb_len = field_a.subgrid_bounds_len;
        lsci_argument_map_set_point(&arg_map, i, &targs, sizeof(targs));
    }
    // setup the index launcher
    lsci_index_launcher_t il;
    lsci_index_launcher_create(&il, INIT_VALS_TID,
                               &field_a.launch_domain,
                               NULL, &arg_map);
    lsc_add_region_requirement(
        &il, field_a.logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, field_a.logical_region
    );
    lsci_add_field(&il, idx++, field_a.fid);
    lsc_add_region_requirement(
        &il, field_b.logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, field_b.logical_region
    );
    lsci_add_field(&il, idx++, field_b.fid);
    lsci_execute_index_space(runtime, context, &il);
}

void
init_vals_task(lsci_task_args_t *task_args)
{
    assert(task_args && task_args->runtime && task_args->context);
    assert(task_args->regions);
    size_t rid = 0;
    mesh_task_args_t targs = *(mesh_task_args_t *)task_args->local_argsp;
    lsci_rect_1d_t field_sgb = (lsci_rect_1d_t)&targs.sgb;
    double *fieldap = raw_rect_ptr_1d_double(
                        task_args->regions, rid++, 0, field_sgb
                    );
    double *fieldbp = raw_rect_ptr_1d_double(
                        task_args->regions, rid++, 0, field_sgb
                  );
    assert(fieldap && fieldbp);
    for (size_t i = 0; i < targs.sgb_len; ++i) {
        fieldap[i] = 1.23;
        fieldbp[i] = 4.56;
    }
}

void
main_task(lsci_task_args_t *task_args)
{
    assert(task_args && task_args->context && task_args->runtime);

    printf("-- starting %s\n", __func__);
    lsci_context_t context = task_args->context;
    lsci_runtime_t runtime = task_args->runtime;
    lsci_unimesh_t mesh_a;
    // 2D thing
    const size_t mesh_width = 16;
    const size_t mesh_height = 16;
    const size_t mesh_depth = 1;
    printf("-- %s: creating meshes\n", __func__);
    assert(LSCI_SUCCESS == lsci_unimesh_create(&mesh_a, mesh_width,
                                               mesh_height, mesh_depth,
                                               context, runtime));
    printf("-- %s: mesh_a rank: %d\n", __func__, (int)mesh_a.dims);
    printf("-- %s: adding fields\n", __func__);
    assert(LSCI_SUCCESS == lsci_unimesh_add_field(&mesh_a,
                                                  LSCI_TYPE_DOUBLE, "field-a",
                                                  context, runtime));
    printf("-- %s: adding fields\n", __func__);
    assert(LSCI_SUCCESS == lsci_unimesh_add_field(&mesh_a,
                                                  LSCI_TYPE_DOUBLE, "field-b",
                                                  context, runtime));
    // DO THIS ONLY AFTER ALL FIELDS HAVE BEEN ADDED
    assert(LSCI_SUCCESS == lsci_unimesh_partition(&mesh_a, 2,
                                                  context, runtime));
    printf("-- %s: calling %s\n", __func__, "init_vals");
    init_vals(&mesh_a, context, runtime);
}

int
main(int argc,
     char **argv)
{
    setbuf(stdout, NULL);
    printf("-- starting %s\n", APP_NAME_STR);
    /* let legion know what the first task will be invoked */
    lsci_set_top_level_task_id(MAIN_TID);

    lsci_reg_task_data_t main_task_data = {
        .cbf = main_task
    };
    /* register legion tasks */
    lsci_register_void_legion_task(
        MAIN_TID,
        LSCI_LOC_PROC,
        true,
        false,
        false,
        LSCI_AUTO_GENERATE_ID,
        "main-task",
        main_task_data
    );
    lsci_reg_task_data_t init_vals_task_data = {
        .cbf = init_vals_task
    };
    /* register legion tasks */
    lsci_register_void_legion_task(
        INIT_VALS_TID,
        LSCI_LOC_PROC,
        false,
        true,
        true,
        LSCI_AUTO_GENERATE_ID,
        "init-vals-task",
        init_vals_task_data
    );
#if 0
    lsci_reg_task_data_t veccp_task_data = {
        .cbf = veccp_task
    };
    /* register legion tasks */
    lsci_register_void_legion_task(
        VECCP_TID,
        LSCI_LOC_PROC,
        false,
        true,
        true,
        LSCI_AUTO_GENERATE_ID,
        "veccp-task",
        veccp_task_data
    );
#endif
    /* and go! */
    return lsci_start(argc, argv);
}
