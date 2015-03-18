#include <stdio.h>

#include "sclegion.h"
#include "legion_c.h"

static const size_t SIZE = 128;

struct MyMesh{
  float* a;
  float* b;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t rank;
  uint32_t xstart, xsize;
  uint32_t ystart, ysize;
  uint32_t zstart, zsize;
};

void MyTask(MyMesh* m){
  printf("width: %d\n", m->width);
  printf("height: %d\n", m->height);
  printf("depth: %d\n", m->depth);
  printf("rank: %d\n", m->rank);

  for(size_t i = 0; i < SIZE; ++i){
    m->a[i] = 3;
  }

  for(size_t i = 0; i < SIZE; ++i){
    m->b[i] = 9;
  }

  for(int i = 0; i < SIZE; ++i){
    printf("a[%d] = %f\n", i, m->a[i]);
    printf("b[%d] = %f\n", i, m->b[i]);
  }
}

void LegionTaskFunction(const legion_task_t task,
                        const legion_physical_region_t* region,
                        unsigned numRegions,
                        legion_context_t context,
                        legion_runtime_t runtime){
  MyMesh* m = 
    (MyMesh*)sclegion_uniform_mesh_reconstruct(task, region, numRegions,
                                               context, runtime);
  MyTask(m);
}

void LegionTaskInitFunction(sclegion_uniform_mesh_t mesh,
                            legion_context_t context,
                            legion_runtime_t runtime){

  sclegion_uniform_mesh_launcher_t launcher =
    sclegion_uniform_mesh_create_launcher(mesh, 1);

  sclegion_uniform_mesh_launcher_add_field(launcher, "a", READ_WRITE);
  sclegion_uniform_mesh_launcher_add_field(launcher, "b", READ_WRITE);

  sclegion_uniform_mesh_launcher_execute(context, runtime, launcher);
}

void main_task(const legion_task_t task,
               const legion_physical_region_t* region,
               unsigned numRegions,
               legion_context_t context,
               legion_runtime_t runtime){

  sclegion_uniform_mesh_t mesh = 
  sclegion_uniform_mesh_create(runtime, context, 1, SIZE, 0, 0);

  sclegion_uniform_mesh_add_field(mesh, "a", SCLEGION_CELL, SCLEGION_FLOAT);

  sclegion_uniform_mesh_add_field(mesh, "b", SCLEGION_CELL, SCLEGION_FLOAT);

  sclegion_uniform_mesh_init(mesh);

  LegionTaskInitFunction(mesh, context, runtime);
}

int lsci_main(int argc, char** argv){
  sclegion_init("main_task", main_task);

  sclegion_register_task(1, "LegionTaskFunction", LegionTaskFunction); 

  return sclegion_start(argc, argv);
}

int main(int argc, char** argv){
  return lsci_main(argc, argv);
}
