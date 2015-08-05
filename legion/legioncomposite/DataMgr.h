/**
 * Xin Tong - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Ray Trace Rendering Code
 */

#ifndef DATA_MGR_H
#define DATA_MGR_H

#include "cstdlib"

class DataMgr
{
    //Q_OBJECT
public:
    DataMgr();

    ~DataMgr();

    void* loadRawFile(const char *filename, int nx, int ny, int nz, int type_size);

//    size_t GetDatasize();

    void GetDataDim(size_t dim[3]);

    void GetDataDim(int dim[3]);

    int GetDataDim(int i) {return col_dims[i];}

    void GetDataDim(int &nx, int &ny, int &nz) {nx = col_dims[0];ny = col_dims[1];nz = col_dims[2];}

    int GetNumCells() {return col_dims[0] * col_dims[1] * col_dims[2];}

    int GetMaxDim();

    void* GetData();

    void DumpData();

    void SaveImage(uint* image, int width, int height, const char* filename);

    void SaveRawImage(uint* image, int width, int height, const char* filename);

    int GetNumTriangles(){ return TotalConnectedTriangles;}

    float* GetTriangles() {return Faces_Triangles;}

    float* GetNormals() {return Normals;}

    float *data = NULL;

    void SetStride(int sx, int sy, int sz){stride[0] = sx; stride[1] = sy; stride[2] = sz;}

private:
//    size_t datasize;
    size_t col_dims[3];

    int TotalConnectedTriangles;
    //int TotalConnectedQuads;
    int TotalConnectedPoints;
    int TotalFaces;

    float* Faces_Triangles;
    float* Normals;

    size_t stride[3];
};


#endif //DATA_MGR_H
