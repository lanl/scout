/**
 * hpgv_block.c
 *
 * Copyright (c) 2008 Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 * 
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */

#include "runtime/volren/hpgv/hpgv_block.h"

namespace scout{

#define TAG_DATA_RECV_LOW   0x8000
#define TAG_DATA_SEND_HIGH  0x8000

#define TAG_DATA_RECV_HIGH  0x8001
#define TAG_DATA_SEND_LOW   0x8001



/* function pointer for quanization */
static quantize_t    *data_quantize_user = 0;


/**
 * block_set_quantize
 *
 */
void block_set_quantize(quantize_t *quantize)
{
    data_quantize_user = quantize;
}

/**
 * data_quantize
 *
 */
static float data_quantize(float value, int varname)
{
    if (data_quantize_user != 0) {
        return data_quantize_user(value, varname);
    }

    return value;
}

void printnf(float* field, int n) 
{ 
  int i;
  for (i = 0; i < n; i++) {
    fprintf(stderr, " %f ", field[i]);
  }
  fprintf(stderr, "\n");
}

void printni(int* field, int n) 
{
  int i;
  for (i = 0; i < n; i++) {
    fprintf(stderr, " %d ", field[i]);
  }
  fprintf(stderr, "\n");
}

void hpgv_block_print(int mpiid, int root, block_t *block) 
{
  blk_header_t* header = &block->blk_header;
  //if (mpiid == root) {
  if (1) {
    fprintf(stderr, "\n blk_num: ");
    printni(header->blk_num, 3);
    fprintf(stderr, "\n blk_id: ");
    printni(header->blk_id, 3);
    fprintf(stderr, "\n blk_neighbor_x: ");
    printni(header->blk_neighbor_x, 3);
    fprintf(stderr, "\n domain_obj_near: ");
    printnf(header->domain_obj_near, 3);
    fprintf(stderr, "\n domain_obj_far: ");
    printnf(header->domain_obj_far, 3);
    fprintf(stderr, "\n domain_obj_size: ");
    printnf(header->domain_obj_size, 3);
    fprintf(stderr, "\n domain_obj_center: ");
    printnf(header->domain_obj_center, 3);
    fprintf(stderr, "\n domain_obj_maxsize: %f", header->domain_obj_maxsize);
    fprintf(stderr, "\n blk_obj_near: ");
    printnf(header->blk_obj_near, 3);
    fprintf(stderr, "\n blk_obj_far: ");
    printnf(header->blk_obj_far, 3);
    fprintf(stderr, "\n blk_obj_size: ");
    printnf(header->blk_obj_size, 3);
    fprintf(stderr, "\n blk_obj_center: ");
    printnf(header->blk_obj_center, 3);
    fprintf(stderr, "\n blk_obj_packnear: ");
    printnf(header->blk_obj_packnear, 3);
    fprintf(stderr, "\n blk_obj_packfar: ");
    printnf(header->blk_obj_packfar, 3);
    fprintf(stderr, "\n blk_obj_packsize: ");
    printnf(header->blk_obj_packsize, 3);
    fprintf(stderr, "\n blk_obj_raynear: ");
    printnf(header->blk_obj_raynear, 3);
    fprintf(stderr, "\n blk_obj_rayfar: ");
    printnf(header->blk_obj_rayfar, 3);
    fprintf(stderr, "\n blk_obj_centridiff: %f", header->blk_obj_centridiff);
    fprintf(stderr, "\n domain_grid_near: ");
    printni(header->domain_grid_near, 3);
    fprintf(stderr, "\n domain_grid_far: ");
    printni(header->domain_grid_far, 3);
    fprintf(stderr, "\n domain_grid_size: ");
    printni(header->domain_grid_size, 3);
    fprintf(stderr, "\n blk_grid_near: ");
    printni(header->blk_grid_near, 3);
    fprintf(stderr, "\n blk_grid_far: ");
    printni(header->blk_grid_far, 3);
    fprintf(stderr, "\n blk_grid_size: ");
    printni(header->blk_grid_size, 3);
    fprintf(stderr, "\n blk_grid_boundary: ");
    printni(header->blk_grid_boundary[0], 2);
    printni(header->blk_grid_boundary[1], 2);
    printni(header->blk_grid_boundary[2], 2);
    fprintf(stderr, "\n blk_grid_packnear: ");
    printni(header->blk_grid_packnear, 3);
    fprintf(stderr, "\n blk_grid_packfar: ");
    printni(header->blk_grid_packfar, 3);
    fprintf(stderr, "\n blk_grid_packsize: ");
    printni(header->blk_grid_packsize, 3);
    fprintf(stderr, "\n blk_coord_lookup: %d", header->blk_coord_lookup);
/*
    fprintf(stderr, "\n blk_coord_obj2grid: ");
    printnf(header->blk_coord_obj2grid[0], GRID_SAMPLE);
    printnf(header->blk_coord_obj2grid[1], GRID_SAMPLE);
    printnf(header->blk_coord_obj2grid[2], GRID_SAMPLE);
    fprintf(stderr, "\n blk_coord_grid2obj: ");
    printnf(header->blk_coord_obj2grid[0], GRID_SAMPLE);
    printnf(header->blk_coord_obj2grid[1], GRID_SAMPLE);
    printnf(header->blk_coord_obj2grid[2], GRID_SAMPLE);
*/
 } 
}

/**
* header_new -- make a new block header
*
*/
void
header_new(int id, MPI_Comm mpicomm, int groupsize,
           double *x, double *y, double *z,
           int nx, int ny, int nz,
           int npx, int npy, int npz,
           blk_header_t *header)
{
    HPGV_ASSERT_P(id, header != NULL, "Empty data", HPGV_ERR_MEM);

    int i, j;

    /* partition data domain */
    int mypz = id /(npx * npy);
    int mypx = (id - mypz * npx * npy) % npx;
    int mypy = (id - mypz * npx * npy) / npx;

    header->blk_id[0]      = mypx;
    header->blk_id[1]      = mypy;
    header->blk_id[2]      = mypz;

    header->blk_num[0]     = npx;
    header->blk_num[1]     = npy;
    header->blk_num[2]     = npz;

    /* neighbor */
    block_init_neighbor(header);

    /* domain grid */
    int size[3];

    size[0] = npx * nx;
    size[1] = npy * ny;
    size[2] = npz * nz;

    for (i = 0; i < 3; i++) {
        header->domain_grid_near[i]   = 0;
        header->domain_grid_far[i]    = size[i] - 1;
        header->domain_grid_size[i]   = size[i];
    }

    /* block grid */
    header->blk_grid_size[0]    = nx;
    header->blk_grid_size[1]    = ny;
    header->blk_grid_size[2]    = nz;

    int boundary = 2;
    for (i = 0; i < 3; i++) {

        header->blk_grid_near[i]
            = header->blk_id[i] * header->blk_grid_size[i];

        header->blk_grid_far[i]
            = (header->blk_id[i] + 1) * header->blk_grid_size[i] -1;

        HPGV_ASSERT(header->blk_grid_size[i]
                    == (header->blk_grid_far[i] - header->blk_grid_near[i] + 1),
                    "Inconsistent block size",
                    HPGV_ERROR);

        /* near */
        header->blk_grid_boundary[i][0] = boundary;

        header->blk_grid_packnear[i]
            = header->blk_grid_near[i] - header->blk_grid_boundary[i][0];

        if (header->blk_grid_packnear[i] < header->domain_grid_near[i]) {
            header->blk_grid_packnear[i] = header->blk_grid_near[i];
        }

        /* real near boundary size */
        header->blk_grid_boundary[i][0]
            = header->blk_grid_near[i] - header->blk_grid_packnear[i];

        /* far */
        header->blk_grid_boundary[i][1] = boundary;

        header->blk_grid_packfar[i]
            = header->blk_grid_far[i] + header->blk_grid_boundary[i][1];

        if (header->blk_grid_packfar[i] > header->domain_grid_far[i]) {
            header->blk_grid_packfar[i] = header->blk_grid_far[i];
        }

        /* real far boundary size */
        header->blk_grid_boundary[i][1]
            = header->blk_grid_packfar[i] - header->blk_grid_far[i];

        header->blk_grid_packsize[i]
            = header->blk_grid_packfar[i] - header->blk_grid_packnear[i] + 1;
    }


    /* block obj */
    header->blk_obj_near[0] = x[0];
    header->blk_obj_near[1] = y[0];
    header->blk_obj_near[2] = z[0];

    header->blk_obj_far[0] = x[nx-1];
    header->blk_obj_far[1] = y[ny-1];
    header->blk_obj_far[2] = z[nz-1];
    float boundary_near[3] = {x[1] - x[0], y[1] - y[0], z[1] - z[0]};
    float boundary_far[3] = {x[nx-1] - x[nx-2],
            y[ny-1] - y[ny-2],
            z[nz-1] - z[nz-2]};

    for (i = 0; i < 3; i++) {
        header->blk_obj_size[i]
            = (header->blk_obj_far[i] - header->blk_obj_near[i]);

        header->blk_obj_center[i]
            = (header->blk_obj_far[i] + header->blk_obj_near[i]) * 0.5f;

        header->blk_obj_packnear[i] = header->blk_obj_near[i] -
            header->blk_grid_boundary[i][0] * boundary_near[i];

        header->blk_obj_packfar[i] = header->blk_obj_far[i] +
            header->blk_grid_boundary[i][1] * boundary_far[i];

        header->blk_obj_packsize[i] = header->blk_obj_packfar[i]
            - header->blk_obj_packnear[i];
    }

    /* domain obj */
    for (i = 0; i < 3; i++) {
        MPI_Allreduce(&(header->blk_obj_near[i]),
                      &(header->domain_obj_near[i]),
                      1, MPI_FLOAT, MPI_MIN, mpicomm);

        MPI_Allreduce(&(header->blk_obj_far[i]),
                      &(header->domain_obj_far[i]),
                      1, MPI_FLOAT, MPI_MAX, mpicomm);

        header->domain_obj_size[i] =
            header->domain_obj_far[i] - header->domain_obj_near[i];

        header->domain_obj_center[i] =
            (header->domain_obj_far[i] + header->domain_obj_near[i]) * 0.5;
    }

    float domain_maxsize;
    domain_maxsize = MAX(header->domain_obj_size[0], header->domain_obj_size[1]);
    domain_maxsize = MAX(domain_maxsize, header->domain_obj_size[2]);
    header->domain_obj_maxsize = domain_maxsize;

    /* ray near and far */
    header->blk_obj_raynear[0] = x[0];
    header->blk_obj_raynear[1] = y[0];
    header->blk_obj_raynear[2] = z[0];

    header->blk_obj_rayfar[0] = x[nx-1] + boundary_far[0];
    header->blk_obj_rayfar[1] = y[ny-1] + boundary_far[1];
    header->blk_obj_rayfar[2] = z[nz-1] + boundary_far[2];

    for (i = 0; i < 3; i++) {
        if (header->blk_obj_raynear[i] < header->domain_obj_near[i]) {
            header->blk_obj_raynear[i] = header->domain_obj_near[i];
        }

        if (header->blk_obj_rayfar[i] > header->domain_obj_far[i]) {
            header->blk_obj_rayfar[i] = header->domain_obj_far[i];
        }
    }

    /* lookup table */
    header->blk_coord_lookup = HPGV_TRUE;

    int axis = 0;
    double *gridptr[3] = {x, y, z};
    int gridsize[3] = {nx, ny, nz};

    /* fill the obj2grid look up table */

    for (axis = 0; axis < 3; axis++) {

        int grid_packsize = header->blk_grid_packsize[axis];

        float min = header->blk_obj_packnear[axis];

        float max = header->blk_obj_packfar[axis];

        int boundary_nearnum = header->blk_grid_boundary[axis][0];

        int boundary_farnum = header->blk_grid_boundary[axis][1];


       for (i = 0; i < grid_packsize - 1; i++) {

            float start_v, end_v;

            if (boundary_nearnum > 0 && i < boundary_nearnum)
            {
                start_v = min + i * boundary_near[axis];
                end_v = min + (i + 1) * boundary_near[axis];

            } else if(boundary_farnum > 0 &&
                      i >= (grid_packsize - boundary_farnum - 1))
            {
                int delta = grid_packsize - 1 - i;
                start_v = max -  delta * boundary_far[axis];
                end_v = max - (delta - 1) * boundary_far[axis];
            } else {
                start_v = gridptr[axis][i-boundary_nearnum];
                end_v = gridptr[axis][i+1-boundary_nearnum];
            }

            int start
                = (int)((start_v - min) / (max - min) * (GRID_SAMPLE - 1));

            int end
                = (int)((end_v - min) /(max - min) * (GRID_SAMPLE - 1));

            HPGV_ASSERT_P(id,
                          start >= 0 &&
                          end <= GRID_SAMPLE - 1
                          && start <= end,
                          "Over flow.",
                          HPGV_ERR_MEM);

            for (j = start; j <= end; j++) {

                float factor = ((float) (j - start)) / (end - start);
                float value = i * (1-factor) + (i+1) * (factor);

                header->blk_coord_obj2grid[axis][j] = value;
            }
        }
    }

    /* fill the grid2obj look up table */
    for (axis = 0; axis < 3; axis++) {

        int grid_packsize = header->blk_grid_packsize[axis];

        float min = header->blk_obj_packnear[axis];

        float max = header->blk_obj_packfar[axis];

        int boundary_nearnum = header->blk_grid_boundary[axis][0];

        int boundary_farnum = header->blk_grid_boundary[axis][1];

        for (i = 0; i < grid_packsize - 1; i++) {

            float start_v, end_v;

            if (boundary_nearnum > 0 && i < boundary_nearnum)
            {
                start_v = min + i * boundary_near[axis];
                end_v = min + (i + 1) * boundary_near[axis];

            } else if(boundary_farnum > 0 &&
                      i >= (grid_packsize - boundary_farnum - 1))
            {
                int delta = grid_packsize - 1 - i;
                start_v = max -  delta * boundary_far[axis];
                end_v = max - (delta - 1) * boundary_far[axis];
            } else {
                start_v = gridptr[axis][i-boundary_nearnum];
                end_v = gridptr[axis][i+1-boundary_nearnum];
            }

            int start
                = i * (GRID_SAMPLE - 1) / (grid_packsize - 1);

            int end
                =  (i + 1) * (GRID_SAMPLE - 1) / (grid_packsize - 1);

            for (j = start; j <= end; j++) {

                float factor = ((float) (j - start)) / (end - start);
                float value = start_v * (1-factor) + end_v * (factor);

                header->blk_coord_grid2obj[axis][j] = value;
            }
        }
    }

    /* find the space for centrical difference */
    float space;
    float min_space = x[1] - x[0];
    for (i = 0; i < nx - 1; i++) {
        space = x[i+1] - x[i];
        if (min_space > space) {
            min_space = space;
        }
    }

    for (i = 0; i < ny - 1; i++) {
        space = y[i+1] - y[i];
        if (min_space > space) {
            min_space = space;
        }
    }

    for (i = 0; i < nz - 1; i++) {
        space = z[i+1] - z[i];
        if (min_space > space) {
            min_space = space;
        }
    }


    MPI_Allreduce(&(min_space),
                  &(header->blk_obj_centridiff),
                  1, MPI_FLOAT, MPI_MIN, mpicomm);
}



/**
 * volume_init
 *
 */
static void volume_init(volume_data_t *volume,
                 int mpiid,
                 MPI_Comm mpicomm,
                 blk_header_t *header,
                 int datatype,
                 void *data,
                 int varname)
{
    int i = 0;
    
    HPGV_ASSERT(volume != NULL, "Empty data", HPGV_ERR_MEM);

    volume->mpiid = mpiid;
    volume->mpicomm = mpicomm;

    volume->data_type = datatype;
    volume->data_typesize = hpgv_typesize(datatype);
    volume->data_varname = varname;
    
    volume->data_original = data;
    volume->exg_maxbuffsize = 0;
    
    int bs = 0;
    
    for (i = 0; i < 2; i++) {
        /* x boundary */
        bs = header->blk_grid_boundary[0][i] *
             header->blk_grid_packsize[1] *
             header->blk_grid_packsize[2];

        if (volume->exg_maxbuffsize < bs) {
            volume->exg_maxbuffsize = bs;
        }

        if (bs > 0) {
            volume->data_boundary_x[i]
                = (void *)calloc(volume->data_typesize, bs);
            
            HPGV_ASSERT(volume->data_boundary_x[i],
                        "Out of memory",
                        HPGV_ERR_MEM);
        } else {
            volume->data_boundary_x[i] = NULL;
        }

        /* y boundary */
        bs = header->blk_grid_packsize[0] *
             header->blk_grid_boundary[1][i] *
             header->blk_grid_packsize[2];

        if (volume->exg_maxbuffsize < bs) {
            volume->exg_maxbuffsize = bs;
        }

        if (bs > 0) {
            volume->data_boundary_y[i]
                = (void *)calloc(volume->data_typesize, bs);

            HPGV_ASSERT(volume->data_boundary_y[i],
                        "Out of memory",
                        HPGV_ERR_MEM);
        } else {
            volume->data_boundary_y[i] = NULL;
        }

        /* z boundary */
        bs = header->blk_grid_packsize[0] *
             header->blk_grid_packsize[1] *
             header->blk_grid_boundary[2][i];
        
        if (volume->exg_maxbuffsize < bs) {
            volume->exg_maxbuffsize = bs;
        }

        if (bs > 0) {
            volume->data_boundary_z[i]
                = (void *)calloc(volume->data_typesize, bs);
        
            HPGV_ASSERT(volume->data_boundary_z[i],
                        "Out of memory",
                        HPGV_ERR_MEM);
        } else {
            volume->data_boundary_z[i] = NULL;
        }
    }

    int maxbuffsize = 0;
    MPI_Allreduce(&(volume->exg_maxbuffsize),
                  &(maxbuffsize),
                  1,
                  MPI_INT,
                  MPI_MAX,
                  volume->mpicomm);

    volume->exg_maxbuffsize = maxbuffsize;
}

/**
 * volume_finalize
 *
 */
static void volume_finalize(volume_data_t *volume)
{
    int i = 0;
    
    if (volume == NULL) {
        return;
    }
    
    for (i = 0; i < 2; i++) {
        if (volume->data_boundary_x[i] != NULL) {
            free(volume->data_boundary_x[i]);
        }
    
        if (volume->data_boundary_y[i] != NULL) {
            free(volume->data_boundary_y[i]);
        }
        
        if (volume->data_boundary_z[i] != NULL) {
            free(volume->data_boundary_z[i]);
        }
        
        if (volume->exg_boundary_send[i] != NULL) {
            free(volume->exg_boundary_send[i]);
        }
        
        if (volume->exg_boundary_recv[i] != NULL) {
            free(volume->exg_boundary_recv[i]);
        }
    }

    if (volume->data_from_file) {
        free(volume->data_original);
    }
    
    free(volume);
    volume = NULL;
}

/**
 * volume_set_emulatesphere
 *
 */
static void volume_set_emulatesphere(volume_data_t *volume, int b)
{
    volume->data_sphere = b;
}


/**
 * volume_get_value_grid
 *
 */
static void volume_get_value_grid(volume_data_t *volume, blk_header_t *header,
                                  int x, int y, int z, void *v)
{    
    HPGV_ASSERT(volume != NULL, "Empty data", HPGV_ERR_MEM);
    
    if (x < 0 || x >= header->blk_grid_packsize[0] ||
        y < 0 || y >= header->blk_grid_packsize[1] ||
        z < 0 || z >= header->blk_grid_packsize[2] )
    {
        memset(v, 0, volume->data_typesize);
        return;
    }
    
    int ts; 
    ts = volume->data_typesize;
    
    int s[3], b[3][2], p[3];
    memcpy(s, header->blk_grid_size, sizeof(int) * 3);
    memcpy(b, header->blk_grid_boundary, sizeof(int) * 3 * 2);
    memcpy(p, header->blk_grid_packsize, sizeof(int) * 3);

    int offset = 0;
    
    /* the original data contains all the block;
       don't need to check the ghost area */
    if (volume->data_from_file) {
        /* original data  */
        offset = x + y * p[0] + z * p[0] * p[1];
        memcpy(v,
               ((char *)(volume->data_original)) + ts * offset,
               ts);
        return;
    }

    /* otherwise, we need to check which regions this position falls in */ 
    int i;
    int d[3] = {0, 0, 0};
    int pos[3];
    pos[0] = x, pos[1] = y, pos[2] = z;
    
    for (i = 0; i < 3; i++) {
        if (pos[i] < b[i][0]) {
            d[i] = -1;
        } else if (pos[i] >= s[i] + b[i][0]) {
            d[i] = 1;
        }
    }
    
    if (d[0] == 0 && d[1] ==0 && d[2] == 0) {
        /* original data  */
        offset = (x - b[0][0])
               + (y - b[1][0]) * s[0]
               + (z - b[2][0]) * s[0] * s[1];
        
        memcpy(v,
               ((char *)(volume->data_original)) + ts * offset,
               ts);
        return;
    } else if (d[2] == -1) {
        /* low z boundary */
        offset = x + y * p[0] + z * p[0] * p[1];
        memcpy(v,
               ((char *)(volume->data_boundary_z[0])) + ts * offset,
               ts);
        return;
    } else if (d[2] == 1) {
        /* high z boundary */
        offset = x + y * p[0] + (z - b[2][0] - s[2]) * p[0] * p[1];
        memcpy(v,
               ((char *)(volume->data_boundary_z[1])) + ts * offset,
               ts);
        return;
    } else if (d[1] == -1) {
        /* low y boundary */
        offset = x + y * p[0] + z * p[0] * b[1][0];
        memcpy(v,
               ((char *)(volume->data_boundary_y[0])) + ts * offset,
               ts);
        return;
    } else if (d[1] == 1) {
        /* high y boundary */
        offset = x + (y - b[1][0] - s[1]) * p[0] + z * p[0] * b[1][1];
        memcpy(v,
               ((char *)(volume->data_boundary_y[1])) + ts * offset,
               ts);
        return;
    } else if (d[0] == -1) {
        /* low x boundary */
        offset = x + y * b[0][0] + z * b[0][0] * p[1];
        memcpy(v,
               ((char *)(volume->data_boundary_x[0])) + ts * offset,
               ts);
        return;
    } else if (d[0] == 1) {
        /* high x boundary */
        offset = (x - b[0][0] - s[0])  + y * b[0][1] + z * b[0][1] * p[1];
        memcpy(v,
               ((char *)(volume->data_boundary_x[1])) + ts * offset,
               ts);
        return;
    } else {
        HPGV_ABORT("Impossible case", HPGV_ERROR);
    }
}

/**
 * volume_get_value_grid_f
 *
 */
static float volume_get_value_grid_f(volume_data_t *volume,
                                     blk_header_t *header,
                                     int x, int y, int z)
{
    float v = 0;

    if (volume->data_type == HPGV_FLOAT) {
        float v_f;
        volume_get_value_grid(volume, header, x, y, z, &v_f);
        v = v_f;
    } else if (volume->data_type == HPGV_DOUBLE) {
        double v_d;
        volume_get_value_grid(volume, header, x, y, z, &v_d);
        v = (float)v_d;
    } else if (volume->data_type == HPGV_UNSIGNED_BYTE) {
        unsigned char v_c;
        volume_get_value_grid(volume, header, x, y, z, &v_c);
        v = v_c / 255.0f;
    } else if (volume->data_type == HPGV_UNSIGNED_SHORT) {
        unsigned short v_s;
        volume_get_value_grid(volume, header, x, y, z, &v_s);
        v = v_s / 65535.0f;
    } else if (volume->data_type == HPGV_UNSIGNED_INT) {
        unsigned int v_n;
        volume_get_value_grid(volume, header, x, y, z, &v_n);
        v = v_n / 4294967295.0f;
    } else {
        HPGV_ABORT("Unsupported format", HPGV_ERROR);
    }

    return v;
}



/**
 * volume_get_value_local
 *
 */
static int volume_get_value_local(volume_data_t *volume,
                                  blk_header_t *header,
                                  float x, float y, float z,
                                  float *value)
{
    int sx = header->blk_grid_packsize[0];
    int sy = header->blk_grid_packsize[1];
    int sz = header->blk_grid_packsize[2];

    if (x < 0 || x > sx - 1 ||
        y < 0 || y > sy - 1 ||
        z < 0 || z > sz - 1 ||
        volume->data_original == NULL)
    {
        *value = 0;
        return HPGV_TRUE;
        //return HPGV_FALSE;
    }

    float t = 0;
    if ( x == (int)(x) && y == (int)(y) && z == (int)(z)) {
        
        t = volume_get_value_grid_f(volume, header, x, y, z);
        
    } else {
         
        int cx = (int)(x);
        int cy = (int)(y);
        int cz = (int)(z);
    
        int px[8], py[8], pz[8];
        int i;
        for (i = 0; i < 8; i++) {
            px[i] = cx + (i & 0x01);
            py[i] = cy + ((i >> 1) & 0x01);
            pz[i] = cz + ((i >> 2) & 0x01);
        }
    
        float fx = x - cx;
        float fy = y - cy;
        float fz = z - cz;
        float v[8];
        
        for (i = 0; i < 8; i++) {
            v[i] = volume_get_value_grid_f(volume, header, px[i], py[i], pz[i]);
        }
    
        t = ((v[0] * (1 - fx) + v[1] * fx) * (1- fy) +
             (v[2] * (1 - fx) + v[3] * fx) * fy) * (1 - fz) +
            ((v[4] * (1 - fx) + v[5] * fx) * (1- fy) +
             (v[6] * (1 - fx) + v[7] * fx) * fy) * fz;
    }

    *value = (float) data_quantize(t, volume->data_varname);

    return HPGV_TRUE;
}


/**
 * block_coord_lookup
 *
 */
int block_coord_obj2grid(blk_header_t *header,
                         float ox, float oy, float oz,
                         float *gx, float *gy, float *gz)
{
    int tx, ty, tz;
    int samplesize = GRID_SAMPLE - 1;
    
    tx = (int)((ox - header->blk_obj_packnear[0]) * samplesize
               / header->blk_obj_packsize[0]);
    
    if (tx < 0 || tx >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }

    ty = (int)((oy - header->blk_obj_packnear[1]) * samplesize
               / header->blk_obj_packsize[1]);
               

    if (ty < 0 || ty >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }

    tz = (int)((oz - header->blk_obj_packnear[2]) * samplesize
               / header->blk_obj_packsize[2]);
               

    if (tz < 0 || tz >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }
    
    *gx = header->blk_coord_obj2grid[0][tx];
    *gy = header->blk_coord_obj2grid[1][ty];
    *gz = header->blk_coord_obj2grid[2][tz];

    return HPGV_TRUE;
}

/**
 * blcok_coord_lookup
 *
 */
int block_coord_grid2obj(blk_header_t *header,
                         float gx, float gy, float gz,
                         float *ox, float *oy, float *oz)
{
    int tx, ty, tz;
    int samplesize = GRID_SAMPLE - 1;
    
    tx = (int)((gx - header->blk_grid_packnear[0]) * samplesize
               / (header->blk_grid_packsize[0] - 1));
               
    
    if (tx < 0 || tx >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }
    
    ty = (int)((gy - header->blk_grid_packnear[1]) * samplesize
               / (header->blk_grid_packsize[1] - 1));
               
    
    if (ty < 0 || ty >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }
    
    tz = (int)((gz - header->blk_grid_packnear[2]) * samplesize
               / (header->blk_grid_packsize[2] - 1));
               
    
    if (tz < 0 || tz >= GRID_SAMPLE) {
        return HPGV_FALSE;
    }
    
    *ox = header->blk_coord_grid2obj[0][tx];
    *oy = header->blk_coord_grid2obj[1][ty];
    *oz = header->blk_coord_grid2obj[2][tz];
    
    return HPGV_TRUE; 
}


/**
 * volume_get_value_global
 *
 */
static int volume_get_value_global(volume_data_t *volume,
                                   blk_header_t *header,
                                   float x, float y, float z, float *v,
                                   float grid_off_x,
                                   float grid_off_y,
                                   float grid_off_z)
{
    float px, py, pz;
    
    if (volume->data_sphere) {

        float center[3];
        memcpy(center, header->domain_obj_center, sizeof(float) * 3);
        
        px = x + grid_off_x / header->domain_obj_maxsize;
        py = y + grid_off_y / header->domain_obj_maxsize;
        pz = z + grid_off_z / header->domain_obj_maxsize;
        
        float value;
        value = sqrt((px - center[0]) * (px - center[0]) +
                     (py - center[1]) * (py - center[1]) +
                     (pz - center[2]) * (pz - center[2]));

        float max_radius = (header->domain_obj_maxsize) * 0.5f;

        if (value > max_radius) {
            value = max_radius;
        }

        *v = (max_radius - value) / max_radius;
        
        return HPGV_TRUE;
        
    } else {

        if (header->blk_coord_lookup == HPGV_TRUE) {
            if (!block_coord_obj2grid(header, x, y, z,
                                      &px, &py, &pz))
            {
                return HPGV_FALSE;
            }
        } else {
            px = x;
            py = y;
            pz = z;
        }
        
        px += grid_off_x;
        py += grid_off_y;
        pz += grid_off_z;
        
        return volume_get_value_local(volume,
                                      header,
                                      px,
                                      py,
                                      pz,
                                      v);
    }

    return HPGV_FALSE;
}

/**
 * volume_get_gradient_global
 *
 */
static int volume_get_gradient_global(volume_data_t *volume,
                                      blk_header_t *header,
                                      float x, float y, float z,
                                      point_3d_t *gradient)
{
    float a, b;

    /* grident in grid coordinate */
//     if (volume_get_value_global(volume, header, x, y, z, &a, -1, 0, 0) &&
//         volume_get_value_global(volume, header, x, y, z, &b,  1, 0, 0))
//     {
//         gradient->x3d = a - b;
//     } else {
//         gradient->x3d = 0;
//     }
// 
//     if (volume_get_value_global(volume, header, x, y, z, &a, 0, -1, 0) &&
//         volume_get_value_global(volume, header, x, y, z, &b, 0,  1, 0))
//     {
//         gradient->y3d = a - b;
//     } else {
//         gradient->y3d = 0;
//     }
// 
//     if (volume_get_value_global(volume, header, x, y, z, &a, 0, 0, -1) &&
//         volume_get_value_global(volume, header, x, y, z, &b, 0, 0,  1))
//     {
//         gradient->z3d = a - b;
//     } else {
//         gradient->z3d = 0;
//     }
    

    /* grident in obj coordinate */
    float d = header->blk_obj_centridiff;

    if (volume_get_value_global(volume, header, x + d, y, z, &a, 0, 0, 0) &&
        volume_get_value_global(volume, header, x - d, y, z, &b, 0, 0, 0))
    {
        gradient->x3d = a - b;
    } else {
        return HPGV_FALSE;
    }

    if (volume_get_value_global(volume, header, x, y + d, z, &a, 0, 0, 0) &&
        volume_get_value_global(volume, header, x, y - d, z, &b, 0, 0, 0))
    {
        gradient->y3d = a - b;
    } else {
        return HPGV_FALSE;
    }

    if (volume_get_value_global(volume, header, x, y, z + d, &a, 0, 0, 0) &&
        volume_get_value_global(volume, header, x, y, z - d, &b, 0, 0, 0))
    {
        gradient->z3d = a - b;
    } else {
        return HPGV_FALSE;
    }

    
    normalize(gradient);

    return HPGV_TRUE;
}


/**
 * volume_exchange_boundary
 *
 */
static void volume_exchange_boundary(volume_data_t *volume,
                                     blk_header_t *header)
{
    HPGV_ASSERT(volume != NULL, "Empty data", HPGV_ERR_MEM);

    
    if (volume->data_original == NULL || volume->data_sphere) {
        return;
    }
    
    MPI_Request recvreq[2];
    MPI_Request sendreq[2];
    MPI_Status  recvsta[2];
    MPI_Status  sendsta[2];
    int recvcount, sendcount;
    
    int ts = volume->data_typesize;
    int bs = volume->exg_maxbuffsize;

    if (bs == 0) {
        /* no ghost area */
        return;
    }
    
    int i = 0;
    void *sendbuf[2], *recvbuf[2];

    for (i = 0; i < 2; i++) {
        if ( volume->exg_boundary_send[i] == NULL) {
            volume->exg_boundary_send[i] = (void *)calloc(bs, ts);

            HPGV_ASSERT(volume->exg_boundary_send[i],
                        "Out of memory",
                        HPGV_ERR_MEM); 
        }

        if (volume->exg_boundary_recv[i] == NULL) {
            volume->exg_boundary_recv[i] = (void *)calloc(bs, ts);
        
            HPGV_ASSERT(volume->exg_boundary_recv[i],
                        "Out of memory",
                        HPGV_ERR_MEM);
        }

        sendbuf[i] = volume->exg_boundary_send[i];
        recvbuf[i] = volume->exg_boundary_recv[i];
    }

    int s[3], b[3][2], p[3], n[3], f[3];
    memcpy(s, header->blk_grid_size, sizeof(int) * 3);
    memcpy(b, header->blk_grid_boundary, sizeof(int) * 3 * 2);
    memcpy(p, header->blk_grid_packsize, sizeof(int) * 3);

    for (i = 0; i < 3; i++) {
        n[i] = b[i][0];
        f[i] = b[i][0] + s[i];
    }

    int x, y, z, offset;
    
    /* ===================================== */
    /* fill buffer for exchanging x boundary */
    for (i = 0; i < 2; i++) {
        memset(sendbuf[i], 0, ts * bs);
        memset(recvbuf[i], 0, ts * bs);
    }

    /* low */
    for (z = n[2]; z < f[2]; z++) {
        for (y = n[1]; y < f[1]; y++) {
            for (x = 0; x < b[0][0]; x++) {
                
                offset = x + y * b[0][0] + z * b[0][0] * p[1];
                
                volume_get_value_grid(volume,
                                      header,
                                      n[0] + x, y, z,
                                      ((char *)(sendbuf[0]) + ts * offset));

            }
        }
    }
    
    /* high */
    for (z = n[2]; z < f[2]; z++) {
        for (y = n[1]; y < f[1]; y++) {
            for (x = 0; x < b[0][1]; x++) {
                
                offset = x + y * b[0][1] + z * b[0][1] * p[1];
                
                volume_get_value_grid(volume,
                                      header,
                                      f[0] - b[0][1] + x, y, z,
                                      ((char *)(sendbuf[1]) + ts * offset));

            }
        }
    }

    /* exchange */
    recvcount = sendcount = 0;

    if (header->blk_neighbor_x[0] != -1) {
        MPI_Irecv(recvbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_x[0],
                  TAG_DATA_RECV_LOW,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }

    if (header->blk_neighbor_x[1] != -1) {
        MPI_Irecv(recvbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_x[1],
                  TAG_DATA_RECV_HIGH,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }
    
    if (header->blk_neighbor_x[0] != -1) {
        MPI_Isend(sendbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_x[0],
                  TAG_DATA_SEND_LOW,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }

    if (header->blk_neighbor_x[1] != -1) {
        MPI_Isend(sendbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_x[1],
                  TAG_DATA_SEND_HIGH,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }

    if (recvcount > 0) {
        MPI_Waitall(recvcount, recvreq, recvsta);
    }
    
    if (sendcount > 0) {
        MPI_Waitall(sendcount, sendreq, sendsta);
    }
    
    /* post processing */
    memcpy(volume->data_boundary_x[0], recvbuf[0], ts * b[0][0] * p[1] * p[2]);
    memcpy(volume->data_boundary_x[1], recvbuf[1], ts * b[0][1] * p[1] * p[2]);

    /* ===================================== */
    /* fill buffer for exchanging y boundary */
    for (i = 0; i < 2; i++) {
        memset(sendbuf[i], 0, ts * bs);
        memset(recvbuf[i], 0, ts * bs);
    }

    /* low */
    for (z = n[2]; z < f[2]; z++) {
        for (y = 0; y < b[1][0]; y++) {
            for (x = 0; x < p[0]; x++) {
    
                offset = x + y * p[0] + z * p[0] * b[1][0];
                
                volume_get_value_grid(volume,
                                      header,
                                      x, n[1] + y, z,
                                      ((char *)(sendbuf[0]) + ts * offset));
            }
        }
    }

    /* high */
    for (z = n[2]; z < f[2]; z++) {
        for (y = 0; y < b[1][1]; y++) {
            for (x = 0; x < p[0]; x++) {

                offset = x + y * p[0] + z * p[0] * b[1][1];
                
                volume_get_value_grid(volume,
                                      header,
                                      x, f[1] - b[1][1] + y, z,
                                      ((char *)(sendbuf[1]) + ts * offset));
            }
        }
    }
    
    /* exchange */
    recvcount = sendcount = 0;
    
    if (header->blk_neighbor_y[0] != -1) {
        MPI_Irecv(recvbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_y[0],
                  TAG_DATA_RECV_LOW,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }
    
    if (header->blk_neighbor_y[1] != -1) {
        MPI_Irecv(recvbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_y[1],
                  TAG_DATA_RECV_HIGH,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }
    
    if (header->blk_neighbor_y[0] != -1) {
        MPI_Isend(sendbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_y[0],
                  TAG_DATA_SEND_LOW,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }
    
    if (header->blk_neighbor_y[1] != -1) {
        MPI_Isend(sendbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_y[1],
                  TAG_DATA_SEND_HIGH,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }
    
    if (recvcount > 0) {
        MPI_Waitall(recvcount, recvreq, recvsta);
    }
    
    if (sendcount > 0) {
        MPI_Waitall(sendcount, sendreq, sendsta);
    }
    
    /* post processing */
    memcpy(volume->data_boundary_y[0], recvbuf[0], ts * p[0] * b[1][0] * p[2]);
    memcpy(volume->data_boundary_y[1], recvbuf[1], ts * p[0] * b[1][1] * p[2]);

    /* ===================================== */
    /* fill buffer for exchanging z boundary */
    for (i = 0; i < 2; i++) {
        memset(sendbuf[i], 0, ts * bs);
        memset(recvbuf[i], 0, ts * bs);
    }

    /* low */
    for (z = 0; z < b[2][0]; z++) {
        for (y = 0; y < p[1]; y++) {
            for (x = 0; x < p[0]; x++) {
                
                offset = x + y * p[0] + z * p[0] * p[1];
                
                volume_get_value_grid(volume,
                                      header,
                                      x, y, n[2] + z,
                                      ((char *)(sendbuf[0]) + ts * offset));
            }
        }
    }

    /* high */
    for (z = 0; z < b[2][1]; z++) {
        for (y = 0; y < p[1]; y++) {
            for (x = 0; x < p[0]; x++) {
                
                offset = x + y * p[0] + z * p[0] * p[1];
                
                volume_get_value_grid(volume,
                                      header,
                                      x, y, f[2] - b[2][1] + z,
                                      ((char *)(sendbuf[1]) + ts * offset));
            }
        }
    }
    
    /* exchange */
    recvcount = sendcount = 0;
    
    if (header->blk_neighbor_z[0] != -1) {
        MPI_Irecv(recvbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_z[0],
                  TAG_DATA_RECV_LOW,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }
    
    if (header->blk_neighbor_z[1] != -1) {
        MPI_Irecv(recvbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_z[1],
                  TAG_DATA_RECV_HIGH,
                  volume->mpicomm,
                  &(recvreq[recvcount++]));
    }
    
    if (header->blk_neighbor_z[0] != -1) {
        MPI_Isend(sendbuf[0],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_z[0],
                  TAG_DATA_SEND_LOW,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }
    
    if (header->blk_neighbor_z[1] != -1) {
        MPI_Isend(sendbuf[1],
                  ts * bs,
                  MPI_BYTE,
                  header->blk_neighbor_z[1],
                  TAG_DATA_SEND_HIGH,
                  volume->mpicomm,
                  &(sendreq[sendcount++]));
    }
    
    if (recvcount > 0) {
        MPI_Waitall(recvcount, recvreq, recvsta);
    }
    
    if (sendcount > 0) {
        MPI_Waitall(sendcount, sendreq, sendsta);
    }
    
    /* post processing */
    memcpy(volume->data_boundary_z[0], recvbuf[0], ts * p[0] * p[1] * b[2][0]);
    memcpy(volume->data_boundary_z[1], recvbuf[1], ts * p[0] * p[1] * b[2][1]);
    
}

/**
 * block_init
 *
 */
void block_init(block_t *block,
                int mpiid,
                MPI_Comm mpicomm,
                blk_header_t header)
{
    HPGV_ASSERT(block != NULL, "Empty data", HPGV_ERR_MEM);
    block->mpiid = mpiid;
    block->mpicomm = mpicomm;
    memcpy(&(block->blk_header), &(header), sizeof(blk_header_t));
}


/**
 * block_finalize
 *
 */
void block_finalize(block_t *block)
{
    if (block == NULL) {
        return;
    }

    int i = 0;
    if (block->volume_num > 0 && block->volume_data != NULL) {
        for (i = 0; i < block->volume_num; i++) {
            if (block->volume_data[i] != NULL) {
                volume_finalize(block->volume_data[i]);
            }
        }

        free(block->volume_data);
    }

    if (block->particle_data != NULL) {
        free(block->particle_data);
    }

    free(block);
}

/**
 * block_add_volume
 *
 */
void block_add_volume(block_t *block, int datatype, void *data, int varname)
{
    HPGV_ASSERT(block != NULL, "Empty data", HPGV_ERR_MEM);

    volume_data_t *volume = (volume_data_t *)calloc(1, sizeof(volume_data_t));
    HPGV_ASSERT_P(block->mpiid, volume != NULL, "Out of memory", HPGV_ERR_MEM);

    volume_init(volume, block->mpiid, block->mpicomm, &(block->blk_header),
                datatype, data, varname);

    block->volume_num++;

    block->volume_data
        = (volume_data_t **)realloc(block->volume_data,
                               block->volume_num * sizeof(volume_data_t *));

    HPGV_ASSERT_P(block->mpiid, block->volume_data != NULL,
                  "Out of memory", HPGV_ERR_MEM);
    
    block->volume_data[block->volume_num - 1] = volume;
}

/**
 * block_get_value
 *
 */
int block_get_value(block_t *block, int vol, float x, float y, float z,
                    float *v)
{
    return volume_get_value_global(block->volume_data[vol],
                                   &(block->blk_header),
                                   x, y, z, v, 0, 0, 0);
}

/**
 * block_get_gradient
 *
 */
int block_get_gradient(block_t *block, int vol, float x, float y, float z,
                       point_3d_t *gradient)
{
    return volume_get_gradient_global(block->volume_data[vol],
                                      &(block->blk_header),
                                      x, y, z,
                                      gradient);
}


/**
 * block_set_emulatesphere
 *
 */
void block_set_emulatesphere(block_t *block, int b)
{
    int i;

    for (i = 0; i < block->volume_num; i++) {
        volume_set_emulatesphere(block->volume_data[i], b);
    }
}

/**
 * block_get_particlevalue
 *
 */
int block_get_particlevalue(block_t *block, int vol, float x, float y, float z,
                            float *v)
{
    return volume_get_value_global(block->volume_data[vol],
                                   &(block->blk_header),
                                   x, y, z, v, 0, 0, 0);
}


/**
 * block_read_data
 *
 */
void block_read_data(block_t *block, int vol, char *filename)
{
    HPGV_ASSERT(block, "block is null", HPGV_ERR_MEM);
    HPGV_ASSERT(vol < block->volume_num, "excess the maximum volume number",
                HPGV_ERROR);
    
    static MPI_Datatype filetype;
    static int sizes[3] = {0, 0, 0};
    static int subsizes[3] = {0, 0, 0};
    static int starts[3] = {0, 0, 0};
    static int totalsize = 0;
    
    MPI_File fd;
    MPI_Status status;
    MPI_Datatype datatype;

    volume_data_t * volume = block->volume_data[vol];
    blk_header_t * header = &(block->blk_header);
    
    switch (volume->data_type) {
    case HPGV_FLOAT: datatype = MPI_FLOAT;
        break;
    case HPGV_DOUBLE: datatype = MPI_DOUBLE;
        break;
    case HPGV_UNSIGNED_INT: datatype = MPI_UNSIGNED;
        break;
    case HPGV_UNSIGNED_SHORT: datatype = MPI_UNSIGNED_SHORT;
        break;
    case HPGV_UNSIGNED_BYTE: datatype = MPI_UNSIGNED_CHAR;
        break;
    default:
        HPGV_ABORT("Unsupported data type.", HPGV_ERROR);
    }
    
    
    if (memcmp(sizes, header->domain_grid_size, sizeof(int) * 3) != 0 ||
        memcmp(subsizes, header->blk_grid_packsize, sizeof(int) * 3) != 0 ||
        memcmp(starts, header->blk_grid_packnear, sizeof(int) * 3) != 0 )
    {
        memcpy(sizes, header->domain_grid_size, sizeof(int) * 3);
        memcpy(subsizes, header->blk_grid_packsize, sizeof(int) * 3);
        memcpy(starts, header->blk_grid_packnear, sizeof(int) * 3);
        
        if (MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                     MPI_ORDER_FORTRAN,
                                     //MPI_ORDER_C,
                                     datatype, &filetype) != MPI_SUCCESS )
        {
            HPGV_ABORT("Can not create subarray", HPGV_ERR_IO);
        }
        
        if (MPI_Type_commit(&filetype) != MPI_SUCCESS) {
            HPGV_ABORT("Can not commit file type", HPGV_ERR_IO);
        }
        
        totalsize = subsizes[0] * subsizes[1] * subsizes[2];
        volume->data_original = realloc(volume->data_original,
                                        hpgv_typesize(volume->data_type) *
                                        totalsize);
        
        HPGV_ASSERT(volume->data_original, "Out of memory", HPGV_ERR_MEM);
    }
    
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &fd) != MPI_SUCCESS)
    {
        HPGV_ABORT("Can not open file", HPGV_ERR_IO);
    }
    
    MPI_File_set_view(fd, 0, datatype, filetype, "native", MPI_INFO_NULL);
    
    if (MPI_File_read_all(fd, volume->data_original, totalsize, datatype,
                          &status) != MPI_SUCCESS)
    {
        HPGV_ABORT("Can not read file", HPGV_ERR_IO);
    }
    
    HPGV_ASSERT_P(block->mpiid,
                  status.count
                  == hpgv_typesize(volume->data_type) * totalsize,
                  "Inconsistent read",
                  HPGV_ERR_IO);
    
    MPI_File_close(&fd);
    volume->data_from_file = HPGV_TRUE;
}


/**
 * block_write_data
 *
 */
void block_write_data(block_t *block, int vol, char *filename)
{
    int i;
    
    HPGV_ASSERT(block, "block is null", HPGV_ERR_MEM);
    HPGV_ASSERT(vol < block->volume_num, "excess the maximum volume number",
                HPGV_ERROR);
    
    static MPI_Datatype filetype;
    static int sizes[3] = {0, 0, 0};
    static int subsizes[3] = {0, 0, 0};
    static int starts[3] = {0, 0, 0};
    static int totalsize = 0;
    
    MPI_File fd;
    MPI_Status status;
    MPI_Datatype datatype;
    
    volume_data_t * volume = block->volume_data[vol];
    blk_header_t * header = &(block->blk_header);

    /* We only save the data into float format at this moment. */
    datatype = MPI_FLOAT;
    
    if (memcmp(sizes, header->domain_grid_size, sizeof(int) * 3) != 0 ||
        memcmp(subsizes, header->blk_grid_size, sizeof(int) * 3) != 0 ||
        memcmp(starts, header->blk_grid_near, sizeof(int) * 3) != 0 )
    {
        memcpy(sizes, header->domain_grid_size, sizeof(int) * 3);
        memcpy(subsizes, header->blk_grid_size, sizeof(int) * 3);
        memcpy(starts, header->blk_grid_near, sizeof(int) * 3);
        
        if (MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                     MPI_ORDER_FORTRAN,
                                     //MPI_ORDER_C,
                                     datatype, &filetype) != MPI_SUCCESS )
        {
            HPGV_ABORT("Can not create subarray", HPGV_ERR_IO);
        }
        
        if (MPI_Type_commit(&filetype) != MPI_SUCCESS) {
            HPGV_ABORT("Can not commit file type", HPGV_ERR_IO);
        }
        
        totalsize = subsizes[0] * subsizes[1] * subsizes[2];
    }

    /* convert the data into float format */
    float *tempbuf = (float *)calloc(totalsize, sizeof(float));
    HPGV_ASSERT(tempbuf, "Out of memory", HPGV_ERR_MEM);
    
    for (i = 0; i < totalsize; i++) {
        switch (volume->data_type) {
        case HPGV_FLOAT:
            tempbuf[i] = ((float *)(volume->data_original))[i];
            break;
        case HPGV_DOUBLE:
            tempbuf[i] = ((double *)(volume->data_original))[i];
            break;
        case HPGV_UNSIGNED_INT:
            tempbuf[i] = ((uint32_t *)(volume->data_original))[i] * 1.0f
                / ((uint32_t)0xFFFFFFFF);
            break;
        case HPGV_UNSIGNED_SHORT:
            tempbuf[i] = ((uint16_t *)(volume->data_original))[i] * 1.0f
                / ((uint16_t)0xFFFF);
            break;
        case HPGV_UNSIGNED_BYTE: 
            tempbuf[i] = ((uint8_t *)(volume->data_original))[i] * 1.0f
                / ((uint8_t)0xFF);
            break;
        default:
            HPGV_ABORT("Unsupported data type.", HPGV_ERROR);
        }
        
    }
    
    
    

    
    if (MPI_File_open(MPI_COMM_WORLD,
                      filename,
                      MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL,
                      &fd) != MPI_SUCCESS)
    {
        HPGV_ABORT("Can not open file", HPGV_ERR_IO);
    }
    
    MPI_File_set_view(fd, 0, datatype, filetype, "native", MPI_INFO_NULL);
    
    if (MPI_File_write_all(fd, tempbuf, totalsize, datatype,
                           &status) != MPI_SUCCESS)
    {
        HPGV_ABORT("Can not read file", HPGV_ERR_IO);
    }
    
    HPGV_ASSERT_P(block->mpiid,
                  status.count
                  == hpgv_typesize(HPGV_FLOAT) * totalsize,
                  "Inconsistent read",
                  HPGV_ERR_IO);
    
    MPI_File_close(&fd);

    free(tempbuf);
}


/**
 * block_init_neighbor
 *
 */
void block_init_neighbor(blk_header_t *header)
{
    int xx, yy, zz, mx, my, mz;
    int count = 0;

    /* 26 neighbors */
    for (zz = -1; zz <= 1; zz++) {
        for (yy = -1; yy <= 1; yy++) {
            for (xx = -1; xx <= 1; xx++) {
                /* skip myself */
                if (xx == 0 && yy == 0 && zz == 0) {
                    continue;
                }
                
                mx = header->blk_id[0] + xx;
                my = header->blk_id[1] + yy;
                mz = header->blk_id[2] + zz;

                int t = -1;
                
                if (mx >= 0 && mx < header->blk_num[0] &&
                    my >= 0 && my < header->blk_num[1] &&
                    mz >= 0 && mz < header->blk_num[2])
                {
                    t = mx + my * header->blk_num[0] +
                        mz * header->blk_num[0] * header->blk_num[1];
                }

                header->blk_neighbor[count] = t;
                count++;

                /* 6 neighbors */
                if (xx == -1 && yy == 0 && zz == 0) {
                    header->blk_neighbor_x[0] = t;
                } else if (xx == 1  && yy ==  0 && zz == 0) {
                    header->blk_neighbor_x[1] = t;
                } else if (xx == 0  && yy == -1 && zz == 0) {
                    header->blk_neighbor_y[0] = t;
                } else if (xx == 0  && yy ==  1 && zz == 0) {
                    header->blk_neighbor_y[1] = t;
                } else if (xx == 0  && yy ==  0 && zz == -1) {
                    header->blk_neighbor_z[0] = t;
                } else if (xx == 0  && yy ==  0 && zz == 1) {
                    header->blk_neighbor_z[1] = t;
                } 
            }
        }
    }
}


/**
 * block_exchange_all_boundary
 *
 */
void block_exchange_all_boundary(block_t *block)
{
    int i = 0;
    for (i = 0; i < block->volume_num; i++) {
        volume_exchange_boundary(block->volume_data[i], &(block->blk_header));
    }
}

/**
 * block_exchange_boundary
 *
 */
void block_exchange_boundary(block_t *block, int vol)
{
    volume_exchange_boundary(block->volume_data[vol], &(block->blk_header));
}


}
