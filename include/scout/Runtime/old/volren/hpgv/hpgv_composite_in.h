/**
 * hpgv_composite_in.h
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


#ifndef HPGV_COMPOSITE_IN_H
#define HPGV_COMPOSITE_IN_H

    
#include "scout/Runtime/volren/hpgv/hpgv_composite.h"

namespace scout {

  extern "C" {

    /**
     * count_t
     *
     */
    typedef struct count_t {
      int   key;
      int   count;
    } count_t;


    /**
     * depth_t
     *
     */
    typedef struct depth_t {
      int     mpiid;
      float   d;
      uint8_t init;
    } depth_t;


    /**
     * swap_message_t
     *
     */
    typedef struct swap_message_t {
      uint32_t mpiid;                 /* Remote processor id */
      uint16_t pixelsize;             /* byte size of a pixel */
      uint64_t recordcount;           /* number of send pixels */
      uint64_t offset;                /* offset of send pixels */
      uint8_t  *partial_image;        /* null if it's a send message. */
      /* the actual send image is stored in */
      /* the swap controller */
      struct swap_message_t *next;
    } swap_message_t;


    /**
     * swap_schedule_t
     *
     */
    typedef struct swap_schedule_t {
      uint32_t mpiid;
      uint32_t curstage;

      /* the pixel I responsible for */
      uint32_t bldoffset;
      uint32_t bldcount;

      uint32_t send_count;
      swap_message_t *first_send;
      swap_message_t *last_send;
      MPI_Request *isendreqs;
      MPI_Status  *isendstats;
      uint64_t send_recordcount;

      uint32_t recv_count;
      swap_message_t *first_recv;
      swap_message_t *last_recv;
      MPI_Request *irecvreqs;
      MPI_Status  *irecvstats;
      uint64_t recv_recordcount;

    } swap_schedule_t;


    /**
     * array_t
     *
     */
    typedef struct array_t{
      int count;
      int cursor;
      uint32_t *array;
    } array_t;


    /**
     * swapnode_t
     *
     */
    typedef struct swapnode_t {
      uint32_t workernum;             /* number of processors which is equal to */
      /* number of image partition */
      uint16_t level;                 /* stage in swap process */
      uint32_t childnum;              /* number of children */
      uint32_t firstworker;           /* first processor id */
      uint32_t lastworker;            /* last processor id */
      uint32_t *assignorder;          /* children's assignment order */
      struct swapnode_t *parent;      /* parent */
      struct swapnode_t **children;   /* children */
    } swapnode_t;



    /**
     * swaptree_t
     *
     */
    typedef struct swaptree_t {
      swapnode_t *root;
      uint16_t maxlevel;
    } swaptree_t;


    /**
     * split_t
     *
     */
    typedef struct split_t {
      int count;
      int *split;
    } split_t;


    /**
     * update_t
     *
     */
    typedef struct update_t {
      uint8_t worker;
      uint8_t workersplit;
      uint8_t tree;
      uint8_t order;
      uint8_t schedule;
    } update_t;


    /**
     * swap_control_t
     *
     */
    typedef struct swap_control_t {
      /* rendering parameters */
      uint32_t image_size_x, image_size_y;
      composite_t composite_type;
      depth_t depth;
      char prefix[64];

      /* mpi */
      int mpiid, workid;
      int *workid_2_mpiid;
      int *workers, workernum;
      int totalprocnum;
      MPI_Comm mpicomm;
      int root;
      int roothasimage;

      /* swap tree */
      swaptree_t *swaptree;

      /* schedule responsible for sending and receiving data */
      swap_schedule_t **schedule;
      uint32_t send_count;
      uint64_t send_recordcount;
      uint32_t recv_count;
      uint64_t recv_recordcount;

      /* the number of swap */
      int stagenum;

      /* partial image that I am responsible for. Ping-pong buffer for
         avoiding to copy my image during multiply stages*/
      uint8_t *partial_image[2];

      /* Use the third buffer for collecting the finished pixels for
         optimization */
      uint8_t *collect_image;

      /* final image, only significant at root */
      uint8_t *final_image;

      /* image format */
      int imageformat;

      /* image type */
      int imagetype;

      /* total pixels of the final image */
      uint64_t totalpixels;

      /* useful image */
      uint64_t usefulpixels;

      /* byte size of a pixel */
      uint16_t pixelsize;

      /* total byte size */
      uint64_t totalpixelbytes;

      /* assignment of the last stage which is used to
         gather the final image*/
      int64_t *finalassign;

      /* cpu split table */
      split_t **workersplit;

    } swap_control_t;


    /**
     * swapaux_splitworker_qkswap
     *
     */
    void swapaux_splitworker_qkswap(swap_control_t *swapctrl);


  }

} // end namespace scout

#endif

