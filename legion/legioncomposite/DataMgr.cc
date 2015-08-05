/**
 * Xin Tong - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Ray Trace Rendering Code
 */

#include "DataMgr.h"
#include "iostream"
#include "float.h"
#include "string.h"
#include "math.h"
#include <stdio.h>



//TODO: this file is adopted from Hongfeng Yu's program
//It should be rewritten completely or we notify him for permission

using namespace std;


float* calculateNormal( float *coord1, float *coord2, float *coord3 )
{
    /* calculate Vector1 and Vector2 */
    float va[3], vb[3], vr[3], val;
    va[0] = coord1[0] - coord2[0];
    va[1] = coord1[1] - coord2[1];
    va[2] = coord1[2] - coord2[2];

    vb[0] = coord1[0] - coord3[0];
    vb[1] = coord1[1] - coord3[1];
    vb[2] = coord1[2] - coord3[2];

    /* cross product */
    vr[0] = va[1] * vb[2] - vb[1] * va[2];
    vr[1] = vb[0] * va[2] - va[0] * vb[2];
    vr[2] = va[0] * vb[1] - vb[0] * va[1];

    /* normalization factor */
    val = sqrt( vr[0]*vr[0] + vr[1]*vr[1] + vr[2]*vr[2] );

    float *norm = new float[3];
    norm[0] = vr[0]/val;
    norm[1] = vr[1]/val;
    norm[2] = vr[2]/val;


    return norm;
}

int DataMgr::GetMaxDim()
{
    return std::max(std::max(col_dims[0], col_dims[1]), col_dims[2]);
}

// Load raw data from disk
void* DataMgr::loadRawFile(const char *filename, int nx, int ny, int nz, int type_size)
{
    FILE *fp = fopen(filename, "rb");

    size_t size = nx * ny * nz * type_size;



    col_dims[0] = nx;
    col_dims[1] = ny;
    col_dims[2] = nz;


    int nCells = nx * ny * nz;

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    data = (float*)malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    for(int i = 0; i < nCells; i++) {
        data[i] /= 500000.0;
        //cout<<data[i]<<" ";
        //data[i] = 0.9;
    }

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif

    return data;
}

DataMgr::~DataMgr()
{
    if (NULL != data) {
        delete [] data;
    }
}

//size_t DataMgr::GetDatasize()
//{
//    return datasize;
//}

void DataMgr::GetDataDim(size_t dim[3])
{
    for(int i = 0; i < 3; i++)  {
        dim[i] = col_dims[i];
    }
}

void DataMgr::GetDataDim(int dim[3])
{
    for(int i = 0; i < 3; i++)  {
        dim[i] = col_dims[i];
    }
}


void* DataMgr::GetData()
{
    return (void *)data;
}

void DataMgr::DumpData()
{
    size_t totalDim = col_dims[0] * col_dims[1] * col_dims[2];
    for(unsigned int i = 0;i < totalDim; i++)
        cout<<(float)data[i]<<" ";
    cout<<endl;
}

DataMgr::DataMgr()
{
    SetStride(1, 1, 1);
}
