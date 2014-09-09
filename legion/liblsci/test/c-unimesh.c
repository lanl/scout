/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 */

#include "../lib/lsci.h"

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
    lsci_add_region_requirement(
        &il, field_a.logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, field_a.logical_region
    );
    lsci_add_field(&il, idx++, field_a.fid);
    lsci_add_region_requirement(
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
    double *fieldap = (double *)raw_rect_ptr_1d(
                        task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, field_sgb
                      );
    double *fieldbp = (double *)raw_rect_ptr_1d(
                        task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, field_sgb
                      );
    assert(fieldap && fieldbp);
    for (size_t i = 0; i < targs.sgb_len; ++i) {
        fieldap[i] = -1;
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
    const size_t mesh_width = 4;
    const size_t mesh_height = 4;
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
    // sanity
    do {
        lsci_vector_t field_a;
        lsci_vector_t field_b;
        lsci_unimesh_get_vec_by_name(&mesh_a, "field-a", &field_a, context, runtime);
        lsci_unimesh_get_vec_by_name(&mesh_a, "field-b", &field_b, context, runtime);
        lsci_vector_dump(&field_a, LSCI_TYPE_DOUBLE, context, runtime);
    } while (0);
}

int
main(int argc,
     char **argv)
{
    setbuf(stdout, NULL);
    printf("-- starting %s\n", APP_NAME_STR);
    /* let legion know what the first task will be invoked */
    lsci_set_top_level_task_id(MAIN_TID);

    /* register legion tasks */
    lsci_register_void_legion_task_aux(
        MAIN_TID,
        LSCI_LOC_PROC,
        true,
        false,
        false,
        LSCI_AUTO_GENERATE_ID,
        "main-task",
        main_task
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
    /* and go! */
    return lsci_start(argc, argv);
}
