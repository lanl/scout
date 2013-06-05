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


//
// We can use C/C++ style typedefs to give us a bit of a hand in
// making specific mesh instances (1d, 2d, or 3d) a bit more friendly
// in appearance. 
//
typedef UniMesh[] UniMesh1D;
typedef UniMesh[:] UniMesh2D;
typedef UniMesh[::] UniMesh3D;

#endif
