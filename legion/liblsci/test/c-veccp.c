/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 */

#include "lsci.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define APP_NAME_STR "c-veccp"

enum {
    MAIN_TID = 0,
    INIT_VALS_TID,
    VECCP_TID
};

typedef struct veccp_args_t {
    lsci_rect_1d_storage_t vec_a_sgb;
    size_t vec_a_sgb_len;
    lsci_rect_1d_storage_t vec_b_sgb;
    size_t vec_b_sgb_len;
} veccp_args_t;

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
    for (size_t i = 0; i < to->launch_domain.volume; ++i) {
        veccp_args_t t_args;
        // from
        t_args.vec_a_sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(from->subgrid_bounds, i);
        t_args.vec_a_sgb_len = from->subgrid_bounds_len;
        // to
        t_args.vec_b_sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(to->subgrid_bounds, i);
        t_args.vec_b_sgb_len = to->subgrid_bounds_len;
        lsci_argument_map_set_point(&arg_map, i, &t_args, sizeof(t_args));
    }
    // setup the index launcher
    lsci_index_launcher_t il;
    lsci_index_launcher_create(&il, VECCP_TID,
                               &to->launch_domain,
                               NULL, &arg_map);
    lsci_add_region_requirement(
        &il, from->logical_partition, 0,
        LSCI_READ_ONLY, LSCI_EXCLUSIVE, from->logical_region
    );
    lsci_add_field(&il, idx++, from->fid);
    lsci_add_region_requirement(
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
    double *fromp = (double *)raw_rect_ptr_1d(
                        task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, from_sgb
                    );
    double *top = (double *)raw_rect_ptr_1d(
                        task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, to_sgb
                  );
    assert(fromp && top);
    for (size_t i = 0; i < t_args.vec_a_sgb_len; ++i) {
        top[i] = fromp[i];
    }
}

void
init_vals(lsci_vector_t *from,
          lsci_vector_t *to,
          lsci_context_t context,
          lsci_runtime_t runtime)
{
    assert(from && to && context && runtime);

    int idx = 0;
    assert(from->launch_domain.volume == to->launch_domain.volume);
    lsci_argument_map_t arg_map;
    lsci_argument_map_create(&arg_map);
    for (size_t i = 0; i < to->launch_domain.volume; ++i) {
        veccp_args_t t_args;
        // from
        t_args.vec_a_sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(from->subgrid_bounds, i);
        t_args.vec_a_sgb_len = from->subgrid_bounds_len;
        // to
        t_args.vec_b_sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(to->subgrid_bounds, i);
        t_args.vec_b_sgb_len = to->subgrid_bounds_len;
        lsci_argument_map_set_point(&arg_map, i, &t_args, sizeof(t_args));
    }
    // setup the index launcher
    lsci_index_launcher_t il;
    lsci_index_launcher_create(&il, INIT_VALS_TID,
                               &to->launch_domain,
                               NULL, &arg_map);
    lsci_add_region_requirement(
        &il, from->logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, from->logical_region
    );
    lsci_add_field(&il, idx++, from->fid);
    lsci_add_region_requirement(
        &il, to->logical_partition, 0,
        LSCI_WRITE_DISCARD, LSCI_EXCLUSIVE, to->logical_region
    );
    lsci_add_field(&il, idx++, to->fid);
    lsci_execute_index_space(runtime, context, &il);
}

void
init_vals_task(lsci_task_args_t *task_args)
{
    assert(task_args && task_args->runtime && task_args->context);
    assert(task_args->regions);
    size_t rid = 0;
    veccp_args_t t_args = *(veccp_args_t *)task_args->local_argsp;
    lsci_rect_1d_t from_sgb = (lsci_rect_1d_t)&t_args.vec_a_sgb;
    lsci_rect_1d_t to_sgb   = (lsci_rect_1d_t)&t_args.vec_b_sgb;
    assert(t_args.vec_a_sgb_len == t_args.vec_b_sgb_len);
    double *fromp = (double *)raw_rect_ptr_1d(
                        task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, from_sgb
                    );
    double *top = (double *)raw_rect_ptr_1d(
                      task_args->regions, LSCI_TYPE_DOUBLE, rid++, 0, to_sgb
                  );
    assert(fromp && top);
    for (size_t i = 0; i < t_args.vec_a_sgb_len; ++i) {
        top[i] = 0.0;
        fromp[i] = -1.234;
    }
}

void
main_task(lsci_task_args_t *task_args)
{
    assert(task_args && task_args->context && task_args->runtime);

    printf("-- starting %s\n", __func__);
    lsci_context_t context = task_args->context;
    lsci_runtime_t runtime = task_args->runtime;
    lsci_vector_t vec_from;
    lsci_vector_t vec_to;
    printf("-- %s: creating vectors\n", __func__);
    lsci_vector_create(&vec_from, 8, LSCI_TYPE_DOUBLE, context, runtime);
    lsci_vector_create(&vec_to, 8, LSCI_TYPE_DOUBLE, context, runtime);
    printf("-- %s: partitioning vectors\n", __func__);
    lsci_vector_partition(&vec_from, 4, context, runtime);
    lsci_vector_partition(&vec_to, 4, context, runtime);
    printf("-- %s: calling %s\n", __func__, "init_vals");
    init_vals(&vec_from, &vec_to, context, runtime);
    printf("-- %s: calling %s\n", __func__, "veccp");
    veccp(&vec_from, &vec_to, context, runtime);
    lsci_vector_dump(&vec_to, LSCI_TYPE_DOUBLE, context, runtime);
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
        "init-vals--task",
        init_vals_task_data
    );
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
    /* and go! */
    return lsci_start(argc, argv);
}
