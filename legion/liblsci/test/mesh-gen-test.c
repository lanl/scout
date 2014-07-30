#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "lsci.h"

struct Mesh {
    float *a;
    float *b;
    uint32_t rank;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
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

// at compile-time, we know:
//
// -the width/height/depth - these are 1 in the case that the
// dimension is not used
//
// -the type of each field (and how to map it to an enum/id)
//
// -the name of each field
//
// -whether each field is read/write in the forall

// this is roughly what our current forall looks like
void
forall_ir(struct Mesh* m, uint32_t depth, uint32_t height, uint32_t width){
    size_t extent = width * height * depth;
    for (size_t i = 0; i < extent; ++i) {
        m->a[extent] += m->b[extent];
    }
}

enum {
    MAIN_TID = 0,
    FORALL_TID
};

    void
main_task(lsci_task_args_t* task_args)
{
    lsci_context_t context = task_args->context;
    lsci_runtime_t runtime = task_args->runtime;

    struct Mesh*m = malloc(sizeof(*m));
    m->rank = 2;
    m->width = 512;
    m->height = 512;
    m->depth = 1;
#if 0
    // no need to allocate here
    m->a = (float*)malloc(sizeof(float)*512*512);
    m->b = (float*)malloc(sizeof(float)*512*512);
#endif
    // create the lsci representation of the mesh
    lsci_unimesh_t lgn_m;
    lsci_unimesh_create(&lgn_m, m->width, m->height, m->depth,
                        context, runtime);
    // add the fields to it
    lsci_unimesh_add_field(&lgn_m, LSCI_TYPE_FLOAT, "a", context, runtime);
    lsci_unimesh_add_field(&lgn_m, LSCI_TYPE_FLOAT, "b", context, runtime);
    // partition?
    // only one partition for now -- FIXME
    lsci_unimesh_partition(&lgn_m, 1, context, runtime);
    // create the forall_task argument map
    // get access to the underlying vectors
    lsci_vector_t field_a;
    lsci_vector_t field_b;
    lsci_unimesh_get_vec_by_name(&lgn_m, "a", &field_a, context, runtime);
    lsci_unimesh_get_vec_by_name(&lgn_m, "b", &field_b, context, runtime);
    int idx = 0;
    lsci_argument_map_t arg_map;
    lsci_argument_map_create(&arg_map);
    for (size_t i = 0; i < field_a.launch_domain.volume; ++i) {
        mesh_task_args_t targs;
        targs.global_width = m->width;
        targs.global_height = m->height;
        targs.global_depth = m->depth;
        targs.rank = m->rank;
        targs.sgb = *(lsci_rect_1d_storage_t *)
            lsci_subgrid_bounds_at(field_a.subgrid_bounds, i);
        targs.sgb_len = field_a.subgrid_bounds_len;
        lsci_argument_map_set_point(&arg_map, i, &targs, sizeof(targs));
    }
    // create an index launcher
    lsci_index_launcher_t il;
    lsci_index_launcher_create(&il, FORALL_TID,
                               &field_a.launch_domain,
                               NULL, &arg_map);
    // add the region requirements for each of the fields
    lsc_add_region_requirement(
        &il, field_a.logical_partition, 0,
        LSCI_READ_WRITE, LSCI_EXCLUSIVE, field_a.logical_region
    );
    lsci_add_field(&il, idx++, field_a.fid);
    //
    lsc_add_region_requirement(
        &il, field_b.logical_partition, 0,
        LSCI_READ_WRITE, LSCI_EXCLUSIVE, field_b.logical_region
    );
    lsci_add_field(&il, idx++, field_b.fid);
    // execute the index launcher
    lsci_execute_index_space(runtime, context, &il);
}

void
forall_task(lsci_task_args_t* task_args)
{
    // extract the args
    printf("%s: hi from task %d\n", __func__, task_args->task_id);
    mesh_task_args_t targs = *(mesh_task_args_t *)task_args->local_argsp;
    lsci_rect_1d_t field_sgb = (lsci_rect_1d_t)&targs.sgb;
    struct Mesh m = {
        .a = NULL,
        .b = NULL,
        .rank = targs.rank,
        .width = targs.global_width,
        .height = targs.global_height,
        .depth = targs.global_depth
    };
    // take the IR from forall_ir and stitch in here
    // launch the forall_task's
}

int main(int argc, char** argv){
    lsci_set_top_level_task_id(MAIN_TID);

    lsci_reg_task_data_t main_task_data = {
        .cbf = main_task
    };

    lsci_register_void_legion_task(MAIN_TID,
            LSCI_LOC_PROC,
            true,
            false,
            false,
            LSCI_AUTO_GENERATE_ID,
            "main-task",
            main_task_data);

    lsci_reg_task_data_t forall_task_data = {
        .cbf = forall_task
    };

    lsci_register_void_legion_task(FORALL_TID,
            LSCI_LOC_PROC,
            false,
            true,
            true,
            LSCI_AUTO_GENERATE_ID,
            "forall-task",
            forall_task_data);

    return lsci_start(argc, argv);
}
