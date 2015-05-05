#include <assert.h>
#include <unistd.h>
#include <stdio.h>

#include "sclegion.h"
#include "legion_c.h"

static const size_t SIZE = 10;

struct MyMesh{
  float* a;
  float* b;
  float* c;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t rank;
  uint32_t xstart, ystart, zstart;
  uint32_t xsize, ysize, zsize;
};

void MyTask(MyMesh* m){
  printf("width: %d\n", m->width);
  printf("height: %d\n", m->height);
  printf("depth: %d\n", m->depth);
  printf("rank: %d\n", m->rank);
  printf("start: %d\n", m->xstart);
  printf("size: %d\n", m->xsize);
 
  for(size_t i = 0; i < m->xsize; ++i){
    m->a[i] = i + m->xstart;
  }

  for(size_t i = 0 ; i < m->xsize; ++i){
    m->c[i] = 2+(i + m->xstart);
  }
}

void MyTask2(MyMesh* m){
  printf("width2: %d\n", m->width);
  printf("height2: %d\n", m->height);
  printf("depth2: %d\n", m->depth);
  printf("rank2: %d\n", m->rank);
  printf("start2: %d\n", m->xstart);
  printf("size2: %d\n", m->xsize);
 
  for(int i = 0; i < m->xsize; ++i){
    assert(m->a[i] == i + m->xstart);
    assert(m->c[i] == 2+(i + m->xstart));
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

void LegionTaskFunction2(const legion_task_t task,
                        const legion_physical_region_t* region,
                        unsigned numRegions,
                        legion_context_t context,
                        legion_runtime_t runtime){
  MyMesh* m = 
    (MyMesh*)sclegion_uniform_mesh_reconstruct(task, region, numRegions,
                                               context, runtime);
  MyTask2(m);
}

void LegionTaskInitFunction(int task_id, legion_privilege_mode_t* mode,
                            sclegion_uniform_mesh_t mesh,
                            legion_context_t context,
                            legion_runtime_t runtime){

  sclegion_uniform_mesh_launcher_t launcher =
    sclegion_uniform_mesh_create_launcher(mesh, task_id);

  sclegion_uniform_mesh_launcher_add_field(launcher, "a", mode[0]);
  sclegion_uniform_mesh_launcher_add_field(launcher, "c", mode[1]);

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
  sclegion_uniform_mesh_add_field(mesh, "c", SCLEGION_CELL, SCLEGION_FLOAT);

  sclegion_uniform_mesh_init(mesh);

  legion_privilege_mode_t p1[] = {WRITE_ONLY, WRITE_ONLY};
  LegionTaskInitFunction(1, p1, mesh, context, runtime);
  legion_privilege_mode_t p2[] = {READ_ONLY, READ_ONLY};
  LegionTaskInitFunction(2, p2, mesh, context, runtime);
}

int lsci_main(int argc, char** argv){
  sclegion_init("main_task", main_task);

  sclegion_register_task(1, "LegionTaskFunction", LegionTaskFunction); 
  sclegion_register_task(2, "LegionTaskFunction2", LegionTaskFunction2); 

  return sclegion_start(argc, argv);
}

int main(int argc, char** argv){
  return lsci_main(argc, argv);
}
