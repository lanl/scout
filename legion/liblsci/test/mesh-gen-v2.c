#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "lsci.h"

struct Mesh {
    float *a;
    float *b;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t rank;
};

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
    printf("%s: d: %lu h: %lu w: %lu\n", __func__,
           (unsigned long)depth, (unsigned long)height, (unsigned long)width);
    size_t extent = width * height * depth;
    for (size_t i = 0; i < extent; ++i) {
        m->a[i] += (m->b[i] + i);
    }
    printf("%s: done!\n", __func__);
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

    struct Mesh *m = malloc(sizeof(*m));
    m->rank = 2;
    m->width = 3;
    m->height = 2;
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
    // create an index launcher
    lsci_index_launcher_t il;
    lsci_mesh_task_args_t mtargs = {
        .global_width = m->width,
        .global_height = m->height,
        .global_depth = m->depth,
        .rank = m->rank
    };
    lsci_argument_map_t arg_map;
    lsci_argument_map_create(&arg_map);
    lsci_index_launcher_create(&il, FORALL_TID,
                               &field_a.launch_domain,
                               &mtargs, sizeof(mtargs), &arg_map);
    // add the region requirements for each of the fields
    lsci_add_region_requirement(
        &il, field_a.logical_partition, 0,
        LSCI_READ_WRITE, LSCI_EXCLUSIVE, field_a.logical_region
    );
    lsci_add_field(&il, idx++, field_a.fid);
    //
    lsci_add_region_requirement(
        &il, field_b.logical_partition, 0,
        LSCI_READ_ONLY, LSCI_EXCLUSIVE, field_b.logical_region
    );
    lsci_add_field(&il, idx++, field_b.fid);
    // execute the index launcher
    lsci_execute_index_space(runtime, context, &il);
    // sanity
    lsci_vector_dump(&field_a, LSCI_TYPE_FLOAT, context, runtime);
    // cleanup
    lsci_unimesh_free(&lgn_m, context, runtime);
}

void
forall_task(lsci_task_args_t* task_args)
{
    static const int aRID = 0;
    static const int bRID = 0;
    // extract the args
    printf("%s: hi from task %d\n", __func__, task_args->task_id);
    lsci_mesh_task_args_t *mtargs = (lsci_mesh_task_args_t *)task_args->argsp;
    //printf("subgrid_bounds_len: %lu\n", mtargs->sgb_len);
    // example of how to get domain to get bounds (volume)
    lsci_domain_t a_dom, b_dom;
    lsci_get_index_space_domain(task_args->runtime, task_args->context,
                                task_args->task, aRID, &a_dom);
    lsci_get_index_space_domain(task_args->runtime, task_args->context,
                                task_args->task, bRID, &b_dom);
    assert(a_dom.volume == b_dom.volume);
    printf("a_dom volume: %lu\n", (unsigned long)a_dom.volume);
    struct Mesh m = {
        .a = NULL,
        .b = NULL,
        .rank = mtargs->rank,
        // XXX not correct. cannot assume width, height, depth. only works
        // because we are dealing with one partition. will cause pain and
        // heartache at some point.
        .width = mtargs->global_width,
        .height = mtargs->global_height,
        .depth = mtargs->global_depth
    };
    m.a = (float *)lsci_raw_rect_ptr_1d(
              task_args->regions,
              LSCI_TYPE_FLOAT,
              aRID,
              0,
              task_args->task,
              task_args->context,
              task_args->runtime
          );
    m.b = (float *)lsci_raw_rect_ptr_1d(
              task_args->regions,
              LSCI_TYPE_FLOAT,
              bRID,
              0,
              task_args->task,
              task_args->context,
              task_args->runtime
          );
    // take the IR from forall_ir and stitch in here
    // launch the forall_task's
    forall_ir(&m, m.depth, m.height, m.width);
}

int main(int argc, char** argv){
    lsci_set_top_level_task_id(MAIN_TID);

    lsci_register_void_legion_task_aux(
            MAIN_TID,
            LSCI_LOC_PROC,
            true,
            false,
            false,
            LSCI_AUTO_GENERATE_ID,
            "main-task",
            main_task);

    lsci_register_void_legion_task_aux(
            FORALL_TID,
            LSCI_LOC_PROC,
            true,
            true,
            true,
            LSCI_AUTO_GENERATE_ID,
            "forall-task",
            forall_task);

    return lsci_start(argc, argv);
}
