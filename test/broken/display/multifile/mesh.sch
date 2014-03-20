//
// The mesh structure for our simple heat transfer
// example. 
//
#ifndef MESH_SCH_
#define MESH_SCH_

//
// Note that a uniform mesh declaration is independent of a specific
// rank or any dimensions.  We are simply describing the attributes of
// the mesh (what fields are stored within the mesh and where within
// the mesh structure) -- we worry about the specific mesh instance at
// a later point in time.
//
uniform mesh UniMesh {
 cells:
  float t1, t2;
};

#endif
